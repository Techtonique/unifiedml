#' Cross-Validation for Model Objects
#'
#' Perform k-fold cross-validation with consistent scoring metrics
#' across different model types. The scoring metric is automatically
#' selected based on the detected task type.
#'
#' @param model A Model object
#' @param X Feature matrix or data.frame
#' @param y Target vector (type determines regression vs classification)
#' @param cv Number of cross-validation folds (default: 5)
#' @param scoring Scoring metric: "rmse", "mae", "accuracy", "f1", or a
#'   custom function with signature \code{function(true, pred)} returning
#'   a scalar. Default: auto-detected based on task type.
#' @param show_progress Whether to show progress bar (default: TRUE) in sequential mode
#' @param verbose logical flag enabling verbose messages (default: TRUE) in parallel mode
#' @param cl Optional number of clusters for parallel processing
#' If using \code{cl} for parallel execution, custom scoring functions must
#' be self-contained (no dependencies on the calling environment).
#' @param seed Reproducibility seed
#' @param fit_params A list of additional arguments passed to model$fit()  
#' @param predict_params A list of additional arguments passed to model$predict()
#'
#' @return Vector of cross-validation scores for each fold
#'
#' @examples
#' \dontrun{
#' library(glmnet)
#' X <- matrix(rnorm(100), ncol = 4)
#' y <- 2*X[,1] - 1.5*X[,2] + rnorm(25)  # numeric -> regression
#'
#' mod <- Model$new(glmnet::glmnet)
#' (cv_scores <- cross_val_score(mod, X, y, cv = 5))  # auto-uses RMSE
#' mean(cv_scores)  # Average RMSE
#'
#' cross_val_score(mod, X, y,
#' fit_params     = list(alpha = 0, lambda = 0.1),
#' predict_params = list(type = "response"))
#' 
#' cross_val_score(mod, X, y,
#' fit_params     = list(alpha = 0.5, lambda = 0.1),
#' predict_params = list(type = "response"))
#' 
#' # Custom scoring: R-squared
#' r2 <- function(true, pred) {
#'   ss_res <- sum((true - pred)^2)
#'   ss_tot <- sum((true - mean(true))^2)
#'   1 - ss_res / ss_tot
#' }
#' 
#' (cv_scores4 <- cross_val_score(mod, X, y, cv = 5, scoring = r2))
#' mean(cv_scores4)  # Average R²
#' 
#' # Classification with accuracy scoring
#' data(iris)
#' X_class <- iris[, 1:4]
#' y_class <- iris$Species  # factor -> classification
#' mod2 <- Model$new(e1071::svm)
#' (cv_scores2 <- cross_val_score(mod2, X_class, y_class, cv = 5))  # auto-uses accuracy
#' mean(cv_scores2)  # Average accuracy
#' 
#' iris_bin <- iris[iris$Species != "virginica", ]
#' X_bin <- iris_bin[, 1:4]
#' y_bin <- droplevels(iris_bin$Species)
#' (cv_scores3 <- cross_val_score(mod2, X_bin, y_bin, cv = 3, 
#' scoring="f1", fit_params=list(kernel="polynomial")))  
#' mean(cv_scores3)  # Average F1
#' }
#'
#' @export
cross_val_score <- function(model,
                            X,
                            y,
                            cv = 5,
                            scoring = NULL,
                            show_progress = TRUE,
                            verbose = TRUE,
                            cl = NULL,
                            seed = 123,
                            fit_params = NULL, 
                            predict_params = NULL) {
  X <- as.matrix(X)
  n <- nrow(X)
  if (length(y) != n)
    stop("Must have: length(y) == nrow(X)")
  if (!(floor(cv) == cv))
    stop("'cv' must be an integer")
  
  set.seed(seed)
  
  folds <- base::split(base::sample(seq_len(n)), rep(1:cv, length.out = n))
  scores <- numeric(cv)
  names(scores) <- paste0("fold", seq_len(cv))
  
  # Auto-detect task based on y
  task_type <- ifelse(is.factor(y), "classification", "regression")
  
  # Auto-detect scoring metric if not provided
  if (is.null(scoring)) {
    scoring <- ifelse(task_type == "regression", "rmse", "accuracy")
  }
  
  # resolve scoring to a function once, before the fold loop
  score_fn <- if (is.function(scoring)) {
    scoring
  } else {
    switch(scoring,
           rmse     = function(true, pred) sqrt(mean((true - pred)^2, na.rm = TRUE)),
           mae      = function(true, pred) mean(abs(true - pred), na.rm = TRUE),
           accuracy = function(true, pred) mean(pred == true, na.rm = TRUE),
           f1       = function(true, pred) compute_f1(true, pred), # see implementation below
           stop("Unknown scoring metric: ", scoring)
    )
  }
  
  if (is.null(cl))
  {
    if (show_progress)
      pb <- utils::txtProgressBar(max = cv, style = 3)
    
    for (i in seq_len(cv)) {
      val_idx   <- folds[[i]]
      train_idx <- setdiff(seq_len(n), val_idx)
      
      m <- model$clone_model()
      do.call(m$fit, c(list(X = X[train_idx,, drop=FALSE], y = y[train_idx]), fit_params))
      
      pred <- do.call(m$predict, c(list(X = X[val_idx,,  drop=FALSE]), predict_params))
      if (is.factor(y))
      {
        true <- droplevels(y[val_idx])
      } else {
        true <- y[val_idx]
      }
      
      scores[i] <- score_fn(true, pred)
      
      if (show_progress)
        utils::setTxtProgressBar(pb, i)
    }
    
    if (show_progress)
      close(pb)
    
    return(scores)
    
  } else {
    cl_SOCK <- parallel::makeCluster(cl, type = "SOCK")
    doParallel::registerDoParallel(cl_SOCK)
    `%op%` <-  foreach::`%dopar%`
    
    # KEY FIX: Store the result and use .combine to collect scores
    scores <- foreach::foreach(
      i = seq_len(cv),
      .packages = c("unifiedml"),
      .combine = 'c',
      .errorhandling = "stop",
      .verbose = verbose
    ) %op% {
      val_idx   <- folds[[i]]
      train_idx <- setdiff(seq_len(n), val_idx)
      
      m <- model$clone_model()
      do.call(m$fit,    c(list(X = X[train_idx,, drop=FALSE], y = y[train_idx]), fit_params))
      
      pred <- do.call(m$predict, c(list(X = X[val_idx,,  drop=FALSE]), predict_params))
      if (is.factor(y))
      {
        true <- droplevels(y[val_idx])
      } else {
        true <- y[val_idx]
      }
      
      # RETURN the score (not assign to scores[i])
      score_fn(true, pred)
    }
    
    parallel::stopCluster(cl_SOCK)
    names(scores) <- paste0("fold", seq_len(cv))
    return(scores)
  }
  
}

compute_f1 <- function(true, pred)
{
  if (is.factor(true)) {
    # For binary classification, assume first two levels
    if (nlevels(true) == 2) {
      tp <- sum(pred == levels(true)[2] & true == levels(true)[2])
      fp <- sum(pred == levels(true)[2] &
                  true == levels(true)[1])
      fn <- sum(pred == levels(true)[1] &
                  true == levels(true)[2])
      precision <- tp / (tp + fp + 1e-10)
      recall <- tp / (tp + fn + 1e-10)
      score <- 2 * precision * recall / (precision + recall + 1e-10)
    } else {
      warning(
        "F1 score currently only supports binary classification. Using accuracy instead."
      )
      score <- mean(pred == true, na.rm = TRUE)
    }
  } else {
    # For numeric binary classification (0/1)
    tp <- sum(pred == 1 &
                true == 1)
    fp <- sum(pred == 1 &
                true == 0)
    fn <- sum(pred == 0 &
                true == 1)
    precision <- tp / (tp + fp + 1e-10)
    recall <- tp / (tp + fn + 1e-10)
    score <- 2 * precision * recall / (precision + recall + 1e-10)
  }
  return(score)
}
