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
#' @param scoring Scoring metric: "rmse", "mae", "accuracy", or "f1" 
#'               (default: auto-detected based on task)
#' @param show_progress Whether to show progress bar (default: TRUE)
#' @param cl Optional cluster for parallel processing (not yet implemented)
#' @param ... Additional arguments passed to model$fit()
#' 
#' @return Vector of cross-validation scores for each fold
#' 
#' @examples
#' \donttest{
#' library(glmnet)
#' X <- matrix(rnorm(100), ncol = 4)
#' y <- 2*X[,1] - 1.5*X[,2] + rnorm(25)  # numeric -> regression
#' 
#' mod <- Model$new(glmnet::glmnet)
#' mod$fit(X, y, alpha = 0, lambda = 0.1)
#' cv_scores <- cross_val_score(mod, X, y, cv = 5)  # auto-uses RMSE
#' mean(cv_scores)  # Average RMSE
#' 
#' # Classification with accuracy scoring
#' data(iris)
#' X_class <- iris[, 1:4]
#' y_class <- iris$Species  # factor -> classification
#' 
#' mod2 <- Model$new(e1071::svm)
#' cv_scores2 <- cross_val_score(mod2, X_class, y_class, cv = 5)  # auto-uses accuracy
#' mean(cv_scores2)  # Average accuracy
#' }
#' 
#' @export
cross_val_score <- function(model, X, y, cv = 5, scoring = NULL, 
                            show_progress = TRUE, cl = NULL, ...) {
  X <- as.matrix(X)
  n <- nrow(X)
  folds <- split(sample(seq_len(n)), rep(1:cv, length.out = n))
  scores <- numeric(cv)
  
  # Auto-detect task based on y
  task_type <- ifelse(is.factor(y), "classification", "regression")
  
  # Auto-detect scoring metric if not provided
  if (is.null(scoring)) {
    scoring <- ifelse(task_type == "regression", "rmse", "accuracy")
  }
  
  if (is.null(cl))
  {
    if (show_progress)
      pb <- utils::txtProgressBar(max = cv, style = 3)
    
    for (i in seq_len(cv)) {
      val_idx   <- folds[[i]]
      train_idx <- setdiff(seq_len(n), val_idx)
      
      m <- model$clone_model()
      m$fit(X = X[train_idx, , drop = FALSE], y = y[train_idx], ...)
      
      pred <- m$predict(X = X[val_idx, , drop = FALSE], ...)
      true <- y[val_idx]
      
      if (scoring == "rmse") {
        scores[i] <- sqrt(mean((true - pred)^2, na.rm = TRUE))
      } else if (scoring == "mae") {
        scores[i] <- mean(abs(true - pred), na.rm = TRUE)
      } else if (scoring == "accuracy") {
        scores[i] <- mean(pred == true, na.rm = TRUE)
      } else if (scoring == "f1") {
        # Binary F1 score - handle multi-class later
        if (is.factor(true)) {
          # For binary classification, assume first two levels
          if (nlevels(true) == 2) {
            tp <- sum(pred == levels(true)[2] & true == levels(true)[2])
            fp <- sum(pred == levels(true)[2] & true == levels(true)[1])
            fn <- sum(pred == levels(true)[1] & true == levels(true)[2])
            precision <- tp / (tp + fp + 1e-10)
            recall <- tp / (tp + fn + 1e-10)
            scores[i] <- 2 * precision * recall / (precision + recall + 1e-10)
          } else {
            warning("F1 score currently only supports binary classification. Using accuracy instead.")
            scores[i] <- mean(pred == true, na.rm = TRUE)
          }
        } else {
          # For numeric binary classification (0/1)
          tp <- sum(pred == 1 & true == 1)
          fp <- sum(pred == 1 & true == 0)
          fn <- sum(pred == 0 & true == 1)
          precision <- tp / (tp + fp + 1e-10)
          recall <- tp / (tp + fn + 1e-10)
          scores[i] <- 2 * precision * recall / (precision + recall + 1e-10)
        }
      }
      
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
    
    if (show_progress)
    {
      pb <- txtProgressBar(min = 0,
                           max = cv,
                           style = 3)
      progress <- function(n)
        utils::setTxtProgressBar(pb, n)
      opts <- list(progress = progress)
      
    } else {
      
      opts <- NULL
      
    }
    
    # KEY FIX: Store the result and use .combine to collect scores
    scores <- foreach::foreach(i = seq_len(cv), 
                               .packages = c("unifiedml"),
                               .combine = 'c',  # ADDED: Combine results into vector
                               .errorhandling = "stop",
                               .options.snow = opts, 
                               .verbose = FALSE) %op% {  # Changed to FALSE for cleaner output
                                 
                                 val_idx   <- folds[[i]]
                                 train_idx <- setdiff(seq_len(n), val_idx)
                                 
                                 m <- model$clone_model()
                                 m$fit(X = X[train_idx, , drop = FALSE], y = y[train_idx], ...)
                                 
                                 pred <- m$predict(X = X[val_idx, , drop = FALSE], ...)
                                 true <- y[val_idx]
                                 
                                 score <- NA  # Initialize score
                                 
                                 if (scoring == "rmse") {
                                   score <- sqrt(mean((true - pred)^2, na.rm = TRUE))
                                 } else if (scoring == "mae") {
                                   score <- mean(abs(true - pred), na.rm = TRUE)
                                 } else if (scoring == "accuracy") {
                                   score <- mean(pred == true, na.rm = TRUE)
                                 } else if (scoring == "f1") {
                                   # Binary F1 score - handle multi-class later
                                   if (is.factor(true)) {
                                     # For binary classification, assume first two levels
                                     if (nlevels(true) == 2) {
                                       tp <- sum(pred == levels(true)[2] & true == levels(true)[2])
                                       fp <- sum(pred == levels(true)[2] & true == levels(true)[1])
                                       fn <- sum(pred == levels(true)[1] & true == levels(true)[2])
                                       precision <- tp / (tp + fp + 1e-10)
                                       recall <- tp / (tp + fn + 1e-10)
                                       score <- 2 * precision * recall / (precision + recall + 1e-10)
                                     } else {
                                       warning("F1 score currently only supports binary classification. Using accuracy instead.")
                                       score <- mean(pred == true, na.rm = TRUE)
                                     }
                                   } else {
                                     # For numeric binary classification (0/1)
                                     tp <- sum(pred == 1 & true == 1)
                                     fp <- sum(pred == 1 & true == 0)
                                     fn <- sum(pred == 0 & true == 1)
                                     precision <- tp / (tp + fp + 1e-10)
                                     recall <- tp / (tp + fn + 1e-10)
                                     score <- 2 * precision * recall / (precision + recall + 1e-10)
                                   }
                                 }
                                 
                                 # RETURN the score (not assign to scores[i])
                                 score
                               }
    
    if (show_progress)
    {
      close(pb)
    }
    
    parallel::stopCluster(cl_SOCK)
    
    return(scores)      
  }
  
}
