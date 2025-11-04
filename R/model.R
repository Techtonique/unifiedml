#' Unified Machine Learning Interface using R6
#' 
#' Provides a consistent interface for various machine learning models in R,
#' with automatic detection of formula vs matrix interfaces, built-in
#' cross-validation, model interpretability, and visualization.
#' 
#' @name Model
#' @docType class
#' @author Your Name
#' 
#' @import R6
#' @importFrom stats predict pt sd cor
#' @importFrom utils txtProgressBar setTxtProgressBar
NULL

#' @title Unified Machine Learning Model
#' @description
#' An R6 class that provides a unified interface for regression and classification
#' models with automatic interface detection, cross-validation, and interpretability
#' features. The task type (regression vs classification) is automatically detected
#' from the response variable type.
#' 
#' @field model_fn The modeling function (e.g., glmnet::glmnet, randomForest::randomForest)
#' @field fitted The fitted model object
#' @field task Type of task: "regression" or "classification" (automatically detected)
#' @field X_train Training features matrix
#' @field y_train Training target vector
#' 
#' @examples
#' \dontrun{
#' # Regression example with glmnet
#' library(glmnet)
#' X <- matrix(rnorm(100), ncol = 4)
#' y <- 2*X[,1] - 1.5*X[,2] + rnorm(25)  # numeric → regression
#' 
#' mod <- Model$new(glmnet::glmnet)
#' mod$fit(X, y, alpha = 0, lambda = 0.1)
#' mod$summary()
#' predictions <- mod$predict(X)
#' 
#' # Classification example  
#' data(iris)
#' iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#' X_class <- as.matrix(iris_binary[, 1:4])
#' y_class <- iris_binary$Species  # factor → classification
#' 
#' mod2 <- Model$new(e1071::svm)
#' mod2$fit(X_class, y_class, kernel = "radial")
#' mod2$summary()
#' 
#' # Cross-validation
#' cv_scores <- cross_val_score(mod, X, y, cv = 5)
#' }
#' 
#' @export
Model <- R6::R6Class(
  "Model",
  public = list(
    model_fn = NULL,
    fitted   = NULL,
    task     = NULL,  # Will be set automatically in fit()
    X_train  = NULL,
    y_train  = NULL,
    
    #' @description Initialize a new Model
    #' @param model_fn A modeling function (e.g., glmnet, randomForest, svm)
    #' @return A new Model object
    initialize = function(model_fn) {
      stopifnot(is.function(model_fn))
      self$model_fn <- model_fn
    },
    
    #' @description Fit the model to training data
    #' 
    #' Automatically detects task type (regression vs classification) based on
    #' the type of the response variable y. Numeric y → regression, 
    #' factor y → classification.
    #' 
    #' @param X Feature matrix or data.frame
    #' @param y Target vector (numeric for regression, factor for classification)
    #' @param ... Additional arguments passed to the model function
    #' @return self (invisible) for method chaining
    fit = function(X, y, ...) {
      X <- as.matrix(X)
      
      # Store training data for later use
      self$X_train <- X
      self$y_train <- y
      
      # Auto-detect task type based on y
      if (is.factor(y)) {
        self$task <- "classification"
        # Ensure y is factor for classification
        y <- as.factor(y)
      } else {
        self$task <- "regression"
        # Ensure y is numeric for regression
        y <- as.numeric(y)
      }
      
      # 1. Try formula + data interface
      df <- data.frame(y = y, X)
      fit_args <- c(list(formula = y ~ ., data = df), list(...))
      self$fitted <- tryCatch(
        do.call(self$model_fn, fit_args),
        error = function(e) NULL
      )
      
      # 2. If failed → try matrix interface
      if (is.null(self$fitted)) {
        fit_args <- c(list(x = X, y = y), list(...))
        self$fitted <- tryCatch(
          do.call(self$model_fn, fit_args),
          error = function(e) {
            #misc::debug_print(X)
            #misc::debug_print(y)
            fit_args <- c(list(x = as.matrix(X), y = as.integer(y)), list(...))
            self$fitted <- tryCatch(do.call(self$model_fn, fit_args),
                                    error = function(e) {
                                      print(e)
                                      stop("Model fit failed.")
                                    }
                                    )
          }
        )
      }
      
      invisible(self)
    },
    
    #' @description Generate predictions from fitted model
    #' @param X Feature matrix for prediction
    #' @param type Type of prediction ("response", "class", "probabilities")
    #' @param ... Additional arguments passed to predict function
    #' @return Vector of predictions
    predict = function(X, type = NULL, ...) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      X <- as.matrix(X)
      
      # Set default type based on task
      if (is.null(type)) {
        type <- ifelse(self$task == "classification", "response", "response")
      }
      
      # 1. Try newdata (formula models)
      df <- data.frame(X)
      pred <- tryCatch(
        predict(self$fitted, newdata = df, type = type, ...),
        error = function(e) NULL
      )
      
      # 2. Fallback: newx (matrix models)
      if (is.null(pred)) {
        pred <- tryCatch(
          predict(self$fitted, newx = X, type = type, ...),
          error = function(e) {
            stop("Predict failed with both newdata and newx.")
          }
        )
      }
      
      # Clean output
      if (is.matrix(pred) && ncol(pred) == 1) pred <- drop(pred)
      if (is.list(pred)) pred <- unlist(pred)
      
      # For classification, ensure factors are returned as original levels if possible
      if (self$task == "classification" && type == "class" && is.factor(self$y_train)) {
        if (is.numeric(pred)) {
          # Convert numeric predictions back to factor levels
          pred <- factor(levels(self$y_train)[pred + 1], levels = levels(self$y_train))
        }
      }
      
      pred
    },
    
    #' @description Print model information
    #' @return self (invisible) for method chaining
    print = function() {
      cat("Model Object\n")
      cat("------------\n")
      cat("Model function:", deparse(substitute(self$model_fn))[1], "\n")
      cat("Fitted:", !is.null(self$fitted), "\n")
      if (!is.null(self$fitted)) {
        cat("Task:", self$task, "\n")
        cat("Training samples:", nrow(self$X_train), "\n")
        cat("Features:", ncol(self$X_train), "\n")
        if (self$task == "classification") {
          cat("Classes:", paste(levels(self$y_train), collapse = ", "), "\n")
          cat("Class distribution:\n")
          print(table(self$y_train))
        }
      }
      invisible(self)
    },
    
    #' @description Compute numerical derivatives and statistical significance
    #' 
    #' Uses finite differences to compute approximate partial derivatives
    #' for each feature, providing model-agnostic interpretability.
    #' 
    #' @param h Step size for finite differences (default: 0.01)
    #' @param alpha Significance level for p-values (default: 0.05)
    #' @return A data.frame with derivative statistics (invisible)
    #' 
    #' @details
    #' The method computes numerical derivatives using central differences.
    #' 
    #' Statistical significance is assessed using t-tests on the derivative
    #' estimates across samples.
    summary = function(h = 0.01, alpha = 0.05) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      
      n <- nrow(self$X_train)
      p <- ncol(self$X_train)
      
      # Compute numerical derivatives for each feature
      derivatives <- matrix(0, nrow = n, ncol = p)
      
      for (j in 1:p) {
        X_plus <- X_minus <- self$X_train
        X_plus[, j] <- X_plus[, j] + h
        X_minus[, j] <- X_minus[, j] - h
        
        pred_plus <- self$predict(X_plus)
        pred_minus <- self$predict(X_minus)
        
        derivatives[, j] <- (pred_plus - pred_minus) / (2 * h)
      }
      
      # Compute mean derivatives and standard errors
      mean_deriv <- colMeans(derivatives)
      se_deriv <- apply(derivatives, 2, sd) / sqrt(n)
      
      # Compute t-statistics and p-values
      t_stat <- mean_deriv / se_deriv
      p_values <- 2 * pt(-abs(t_stat), df = n - 1)
      
      # Create summary table
      feature_names <- colnames(self$X_train)
      if (is.null(feature_names)) {
        feature_names <- paste0("X", 1:p)
      }
      
      summary_table <- data.frame(
        Feature = feature_names,
        Mean_Derivative = mean_deriv,
        Std_Error = se_deriv,
        t_value = t_stat,
        p_value = p_values,
        Significance = ifelse(p_values < alpha, "***", 
                              ifelse(p_values < alpha * 2, "**",
                                     ifelse(p_values < alpha * 4, "*", "")))
      )
      
      cat("\nModel Summary - Numerical Derivatives\n")
      cat("======================================\n")
      cat("Task:", self$task, "\n")
      cat("Samples:", n, "| Features:", p, "\n")
      cat("Step size (h):", h, "\n\n")
      
      print(summary_table, row.names = FALSE)
      cat("\nSignificance codes: 0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1\n")
      
      invisible(summary_table)
    },
    
    #' @description Create partial dependence plot for a feature
    #' 
    #' Visualizes the relationship between a feature and the predicted outcome
    #' while holding other features at their mean values.
    #' 
    #' @param feature Index or name of feature to plot
    #' @param n_points Number of points for the grid (default: 100)
    #' @return self (invisible) for method chaining
    plot = function(feature = 1, n_points = 100) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      
      p <- ncol(self$X_train)
      if (feature < 1 || feature > p) {
        stop("feature must be between 1 and ", p)
      }
      
      feature_name <- colnames(self$X_train)[feature]
      if (is.null(feature_name)) feature_name <- paste0("X", feature)
      
      # Create a grid for the selected feature
      x_range <- range(self$X_train[, feature])
      x_grid <- seq(x_range[1], x_range[2], length.out = n_points)
      
      # Hold other features at their means
      X_grid <- matrix(rep(colMeans(self$X_train), each = n_points), 
                       nrow = n_points)
      X_grid[, feature] <- x_grid
      
      # Get predictions
      y_pred <- self$predict(X_grid)
      
      # Create plot
      par(mfrow = c(1, 1))
      plot(self$X_train[, feature], self$y_train,
           xlab = feature_name,
           ylab = ifelse(self$task == "regression", "y", "Class"),
           main = paste("Partial Dependence Plot -", feature_name),
           pch = 16, col = rgb(0, 0, 0, 0.3))
      
      lines(x_grid, y_pred, col = "red", lwd = 2)
      
      invisible(self)
    },
    
    #' @description Create a deep copy of the model
    #' 
    #' Useful for cross-validation and parallel processing where
    #' multiple independent model instances are needed.
    #' 
    #' @return A new Model object with same configuration
    clone_model = function() {
      Model$new(self$model_fn)
    }
  )
)

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
#' \dontrun{
#' library(glmnet)
#' X <- matrix(rnorm(100), ncol = 4)
#' y <- 2*X[,1] - 1.5*X[,2] + rnorm(25)  # numeric → regression
#' 
#' mod <- Model$new(glmnet::glmnet)
#' mod$fit(X, y, alpha = 0, lambda = 0.1)
#' cv_scores <- cross_val_score(mod, X, y, cv = 5)  # auto-uses RMSE
#' mean(cv_scores)  # Average RMSE
#' 
#' # Classification with accuracy scoring
#' data(iris)
#' X_class <- as.matrix(iris[, 1:4])
#' y_class <- iris$Species  # factor → classification
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
  
  scores
}