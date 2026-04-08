#' Benchmark Multiple Models with Cross-Validation and Model-Specific Parameters
#'
#' Perform k-fold cross-validation on a list of models, using model-specific parameters.
#' Supports verbose messages and a progress bar.
#'
#' @param models A named list of \code{Model$new(...)} objects to benchmark.
#' @param X A data frame or matrix of predictors.
#' @param y A vector of outcomes (factor for classification, numeric for regression).
#' @param cv Integer, number of cross-validation folds (default 5).
#' @param scoring Scoring metric: "rmse", "mae", "accuracy", or "f1" 
#'               (default: auto-detected based on task)
#' @param params Optional named list of lists, each sublist containing extra arguments
#'   to pass to the corresponding model's \code{fit()} call. Names must match `models`.
#' @param cl Optional cluster for parallel processing (not yet implemented)
#' @param show_progress Logical, whether to show a progress bar (default TRUE).
#' @param verbose Logical, whether to print messages about each model (default TRUE).
#'
#' @return A named numeric vector containing the mean CV score for each model.
#'
#' @examples
#' \dontrun{
#' library(randomForest)
#'
#' X <- iris[, 1:4]
#' y <- iris$Species
#'
#' models <- list(
#'   glm  = Model$new(caret::train),
#'   rf   = Model$new(randomForest::randomForest),
#'   xgb  = Model$new(caret::train)
#' )
#'
#' params <- list(
#'   glm = list(method = "glmnet",
#'              tuneGrid = data.frame(alpha = 0, lambda = 0.01),
#'              trControl = trainControl(method = "none")),
#'   rf  = list(ntree = 150),
#'   xgb = list(method = "xgbTree",
#'              tuneGrid = data.frame(nrounds = 150, max_depth = 3, eta = 0.3,
#'                                    gamma = 0, colsample_bytree = 1,
#'                                    min_child_weight = 1, subsample = 1),
#'              trControl = trainControl(method = "none"))
#' )
#'
#' results <- benchmark(models, X, y, cv = 5, params = params,
#'                      show_progress = TRUE, verbose = TRUE)
#' print(results)
#' }
#' @export
benchmark <- function(models, X, y, cv = 5L, scoring = NULL, params = NULL, cl=NULL, show_progress = FALSE, verbose = TRUE) {
  n_models <- length(models)
  results <- numeric(n_models)
  names(results) <- names(models)
  
  if (show_progress) {
    pb <- utils::txtProgressBar(min = 0, max = n_models, style = 3)
  }
  
  for (i in seq_along(models)) {
    model_name <- names(models)[i]
    mod <- models[[i]]
    
    if (verbose) cat(sprintf("\n[%d/%d] Fitting model: %s\n", 
                             i, length(models), model_name))
    
    # Extract model-specific parameters if provided
    extra_args <- if (!is.null(params) && model_name %in% names(params)) {
      params[[model_name]]
    } else {
      list()
    }
    
    # Run cross-validation using your cross_val_score function
    args <- c(
      list(
        model = mod,
        X = X,
        y = y,
        cv = cv,
        scoring = scoring,
        show_progress = FALSE,
        cl = cl
      ),
      extra_args
    )
    
    scores <- do.call(cross_val_score, args)
    
    results[i] <- mean(scores)
    
    if (verbose) cat(sprintf("Mean CV score for %s: %.4f\n", model_name, results[i]))
    
    if (show_progress) utils::setTxtProgressBar(pb, i)
  }
  
  if (show_progress) close(pb)
  
  return(results)
}