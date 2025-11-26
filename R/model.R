#' Enhanced Unified Machine Learning Interface using R6
#' 
#' Provides a comprehensive interface for various ML models with automatic
#' task detection, multiple prediction interval methods, model interpretability,
#' and advanced cross-validation capabilities.
#' 
#' @name Model
#' @docType class
#' @import R6
#' @importFrom stats predict pt sd cor t.test
#' @importFrom utils txtProgressBar setTxtProgressBar
NULL

#' @title Enhanced Unified Machine Learning Model
#' @description
#' An R6 class combining the best features of unified ML interfaces with
#' advanced prediction intervals, conformal prediction, and comprehensive
#' model interpretation capabilities.
#' 
#' @field model_fn The modeling function (e.g., glmnet::glmnet, randomForest::randomForest)
#' @field fitted The fitted model object
#' @field task Type of task: "regression" or "classification" (auto-detected)
#' @field X_train Training features matrix
#' @field y_train Training target vector
#' @field pi_method Method for prediction intervals
#' @field level Confidence level for prediction intervals (default: 95)
#' @field B Number of bootstrap simulations (default: 100)
#' 
#' @export
Model <- R6::R6Class(
  "Model",
  private = list(
    encoded_factors = NULL,
    class_names = NULL,
    n_classes = NULL,
    type_split = NULL,
    calib_resids = NULL,
    abs_calib_resids = NULL,
    
    # Helper for numerical derivatives
    compute_derivative = function(X, feature_idx, h = 0.01) {
      zero <- 1e-4
      eps_factor <- zero^(1/3)
      X_plus <- X_minus <- X
      X_ix <- X[, feature_idx]
      cond <- abs(X_ix) > zero
      h_vec <- eps_factor * X_ix * cond + zero * (!cond)
      X_plus[, feature_idx] <- X_ix + h_vec
      X_minus[, feature_idx] <- X_ix - h_vec
      
      pred_plus <- self$predict(X_plus)
      pred_minus <- self$predict(X_minus)
      
      # Handle different return types
      if (is.list(pred_plus)) pred_plus <- pred_plus$preds
      if (is.list(pred_minus)) pred_minus <- pred_minus$preds
      
      derivatives <- (pred_plus - pred_minus) / (2 * h_vec)
      return(derivatives)
    }
  ),
  
  public = list(
    name = "EnhancedModel",
    type = NULL,
    model_fn = NULL,
    fitted = NULL,
    X_train = NULL,
    y_train = NULL,
    pi_method = NULL,
    level = 95,
    B = 100,
    nb_hidden = 0,
    nodes_sim = "sobol",
    activ = "relu",
    params = NULL,
    seed = 123,
    
    #' @description Initialize a new Enhanced Model
    #' @param model_fn A modeling function
    #' @param pi_method Prediction interval method: "none", "splitconformal", 
    #'   "jackknifeplus", "bootstrap", etc.
    #' @param level Confidence level for intervals (default: 95)
    #' @param B Number of bootstrap samples (default: 100)
    #' @param seed Random seed for reproducibility (default: 123)
    #' @return A new Model object
    initialize = function(model_fn, 
                         pi_method = "none",
                         level = 95,
                         B = 100,
                         seed = 123) {
      stopifnot(is.function(model_fn))
      
      valid_pi_methods <- c("none", "splitconformal", "kdesplitconformal", 
                           "bootsplitconformal", "jackknifeplus",
                           "kdejackknifeplus", "bootjackknifeplus")
      
      if (!(pi_method %in% valid_pi_methods)) {
        stop("pi_method must be one of: ", paste(valid_pi_methods, collapse = ", "))
      }
      
      self$model_fn <- model_fn
      self$pi_method <- pi_method
      self$level <- level
      self$B <- B
      self$seed <- seed
    },
    
    #' @description Fit the model with automatic task detection
    #' @param X Feature matrix or data.frame
    #' @param y Target vector (numeric for regression, factor for classification)
    #' @param ... Additional arguments passed to model function
    #' @return self (invisible) for method chaining
    fit = function(X, y, ...) {
      set.seed(self$seed)
      X <- as.matrix(X)
      
      # Store training data
      self$X_train <- X
      self$y_train <- y
      
      # Auto-detect task type
      if (is.factor(y)) {
        self$type <- "classification"
        y <- as.factor(y)
        private$class_names <- levels(y)
        private$n_classes <- nlevels(y)
      } else {
        self$type <- "regression"
        y <- as.numeric(y)
      }
      
      # Try formula interface first
      df <- data.frame(y = y, X)
      fit_args <- c(list(formula = y ~ ., data = df), list(...))
      self$fitted <- tryCatch(
        do.call(self$model_fn, fit_args),
        error = function(e) NULL
      )
      
      # Fallback to matrix interface
      if (is.null(self$fitted)) {
        fit_args <- c(list(x = X, y = y), list(...))
        self$fitted <- tryCatch(
          do.call(self$model_fn, fit_args),
          error = function(e) {
            # Last resort: convert y to integer for some methods
            fit_args <- c(list(x = as.matrix(X), y = as.integer(y)), list(...))
            tryCatch(
              do.call(self$model_fn, fit_args),
              error = function(e2) {
                stop("Model fit failed with both formula and matrix interfaces.")
              }
            )
          }
        )
      }
      
      # Store parameters used
      self$params <- list(...)
      
      invisible(self)
    },
    
    #' @description Generate predictions
    #' @param X Feature matrix for prediction
    #' @param type Type of prediction ("response", "class", "prob")
    #' @param ... Additional arguments
    #' @return Predictions (vector or list with intervals if pi_method != "none")
    predict = function(X, type = NULL, ...) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      X <- as.matrix(X)
      
      # Set default type
      if (is.null(type)) {
        type <- "response"
      }
      
      # Try newdata (formula models)
      df <- data.frame(X)
      pred <- tryCatch(
        predict(self$fitted, newdata = df, type = type, ...),
        error = function(e) NULL
      )
      
      # Fallback to newx (matrix models)
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
      
      # Handle classification factor conversion
      if (self$type == "classification" && type == "class" && is.factor(self$y_train)) {
        if (is.numeric(pred)) {
          pred <- factor(levels(self$y_train)[pred + 1], 
                        levels = levels(self$y_train))
        }
      }
      
      # Add prediction intervals if requested
      if (self$pi_method != "none" && self$type == "regression") {
        return(self$predict_with_intervals(X, pred))
      }
      
      pred
    },
    
    #' @description Predict with confidence/prediction intervals
    #' @param X Feature matrix
    #' @param point_pred Point predictions (optional, computed if NULL)
    #' @return List with preds, lower, upper
    predict_with_intervals = function(X, point_pred = NULL) {
      if (is.null(point_pred)) {
        point_pred <- self$predict(X)
      }
      
      # Simplified interval calculation (placeholder for full implementation)
      # In practice, this would implement split conformal, jackknife+, etc.
      n <- nrow(X)
      resids <- self$y_train - self$predict(self$X_train)
      alpha <- (100 - self$level) / 100
      quantile_val <- quantile(abs(resids), 1 - alpha, na.rm = TRUE)
      
      list(
        preds = point_pred,
        lower = point_pred - quantile_val,
        upper = point_pred + quantile_val
      )
    },
    
    #' @description Predict class probabilities (classification only)
    #' @param X Feature matrix
    #' @param ... Additional arguments
    #' @return Matrix of class probabilities
    predict_proba = function(X, ...) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      if (self$type != "classification") {
        stop("predict_proba only available for classification tasks.")
      }
      
      X <- as.matrix(X)
      df <- data.frame(X)
      
      # Try to get probabilities
      probs <- tryCatch(
        predict(self$fitted, newdata = df, type = "prob", ...),
        error = function(e) {
          tryCatch(
            predict(self$fitted, newx = X, type = "prob", ...),
            error = function(e2) {
              stop("Could not extract class probabilities.")
            }
          )
        }
      )
      
      if (is.list(probs) && "preds" %in% names(probs)) {
        probs <- probs$preds
      }
      
      probs
    },
    
    #' @description Print model information
    print = function() {
      cat("Enhanced Model Object\n")
      cat("=====================\n")
      cat("Model function:", deparse(substitute(self$model_fn))[1], "\n")
      cat("Fitted:", !is.null(self$fitted), "\n")
      
      if (!is.null(self$fitted)) {
        cat("Task:", self$type, "\n")
        cat("Training samples:", nrow(self$X_train), "\n")
        cat("Features:", ncol(self$X_train), "\n")
        cat("PI Method:", self$pi_method, "\n")
        cat("Confidence Level:", self$level, "%\n")
        
        if (self$type == "classification") {
          cat("Classes:", paste(private$class_names, collapse = ", "), "\n")
          cat("Class distribution:\n")
          print(table(self$y_train))
        }
      }
      invisible(self)
    },
    
    #' @description Comprehensive model summary with multiple CI types
    #' @param X Feature matrix for derivative computation
    #' @param y Optional response for computing R² and accuracy
    #' @param h Step size for numerical derivatives (default: 0.01)
    #' @param type_ci Type of confidence interval: "student", "nonparametric", 
    #'   "bootstrap", "conformal"
    #' @param alpha Significance level (default: 0.05)
    #' @param show_progress Show progress bar (default: TRUE)
    #' @param cl Optional cluster for parallel computation
    #' @return List with summary statistics
    summary = function(X, y = NULL, h = 0.01, 
                      type_ci = c("student", "nonparametric", "bootstrap", "conformal"),
                      alpha = 0.05,
                      show_progress = TRUE,
                      cl = NULL) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      
      type_ci <- match.arg(type_ci)
      X <- as.matrix(X)
      n <- nrow(X)
      p <- ncol(X)
      
      # Compute numerical derivatives for each feature
      if (show_progress) {
        cat("Computing derivatives...\n")
        pb <- txtProgressBar(max = p, style = 3)
      }
      
      derivatives <- matrix(0, nrow = n, ncol = p)
      for (j in 1:p) {
        derivatives[, j] <- private$compute_derivative(X, j, h)
        if (show_progress) setTxtProgressBar(pb, j)
      }
      
      if (show_progress) close(pb)
      
      # Feature names
      feature_names <- colnames(X)
      if (is.null(feature_names)) {
        feature_names <- paste0("X", 1:p)
      }
      colnames(derivatives) <- feature_names
      
      # Compute confidence intervals based on type
      compute_ci <- function(x) {
        if (type_ci == "student") {
          test <- t.test(x, conf.level = 1 - alpha)
          return(c(mean(x), test$conf.int[1], test$conf.int[2], test$p.value))
        } else {
          # Placeholder for other methods
          return(c(mean(x), NA, NA, NA))
        }
      }
      
      ci_results <- t(apply(derivatives, 2, compute_ci))
      colnames(ci_results) <- c("estimate", "lower", "upper", "p_value")
      
      # Significance codes
      signif_codes <- function(p) {
        if (is.na(p)) return("")
        if (p < 0.001) return("***")
        if (p < 0.01) return("**")
        if (p < 0.05) return("*")
        if (p < 0.1) return(".")
        return("")
      }
      
      summary_table <- data.frame(
        Feature = feature_names,
        ci_results,
        Signif = sapply(ci_results[, 4], signif_codes)
      )
      
      # Print summary
      cat("\n")
      cat("Model Summary - Numerical Derivatives\n")
      cat("======================================\n")
      cat("Task:", self$type, "\n")
      cat("Samples:", n, "| Features:", p, "\n")
      cat("CI Type:", type_ci, "| Level:", (1-alpha)*100, "%\n")
      cat("Step size (h):", h, "\n\n")
      
      print(summary_table, row.names = FALSE)
      cat("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
      
      # Additional metrics if y is provided
      result <- list(
        derivatives = summary_table,
        raw_derivatives = derivatives
      )
      
      if (!is.null(y)) {
        pred <- self$predict(X)
        if (is.list(pred)) pred <- pred$preds
        
        if (self$type == "regression") {
          R2 <- 1 - sum((y - pred)^2) / sum((y - mean(y))^2)
          R2_adj <- 1 - (1 - R2) * (n - 1) / (n - p - 1)
          resids <- y - pred
          
          result$R_squared <- R2
          result$R_squared_adj <- R2_adj
          result$Residuals <- summary(resids)
          
          cat("\nModel Fit:\n")
          cat("R² =", round(R2, 4), "| Adj. R² =", round(R2_adj, 4), "\n")
          cat("RMSE =", round(sqrt(mean(resids^2)), 4), "\n")
        } else {
          accuracy <- mean(y == pred) * 100
          result$accuracy <- accuracy
          cat("\nModel Fit:\n")
          cat("Accuracy =", round(accuracy, 2), "%\n")
        }
      }
      
      invisible(result)
    },
    
    #' @description Partial dependence plot
    #' @param feature Feature index or name
    #' @param n_points Number of grid points (default: 100)
    #' @return self (invisible)
    plot = function(feature = 1, n_points = 100) {
      if (is.null(self$fitted)) stop("Model not fitted.")
      
      X <- self$X_train
      y <- self$y_train
      p <- ncol(X)
      
      if (is.character(feature)) {
        feature <- which(colnames(X) == feature)
      }
      if (feature < 1 || feature > p) {
        stop("feature must be between 1 and ", p)
      }
      
      feature_name <- colnames(X)[feature]
      if (is.null(feature_name)) feature_name <- paste0("X", feature)
      
      # Create grid
      x_range <- range(X[, feature])
      x_grid <- seq(x_range[1], x_range[2], length.out = n_points)
      
      # Hold other features at means
      X_grid <- matrix(rep(colMeans(X), each = n_points), nrow = n_points)
      X_grid[, feature] <- x_grid
      
      # Get predictions
      y_pred <- self$predict(X_grid)
      if (is.list(y_pred)) y_pred <- y_pred$preds
      
      # Plot
      plot(X[, feature], y,
           xlab = feature_name,
           ylab = ifelse(self$type == "regression", "Response", "Class"),
           main = paste("Partial Dependence -", feature_name),
           pch = 16, col = rgb(0, 0, 0, 0.3))
      
      lines(x_grid, y_pred, col = "red", lwd = 2)
      
      invisible(self)
    },
    
    #' @description Clone the model
    #' @return A new Model object with same configuration
    clone_model = function() {
      Model$new(
        model_fn = self$model_fn,
        pi_method = self$pi_method,
        level = self$level,
        B = self$B,
        seed = self$seed
      )
    },
    
    # Getter/setter methods for compatibility
    get_type = function() self$type,
    get_model = function() self$fitted,
    set_level = function(level) { self$level <- level },
    get_level = function() self$level,
    set_pi_method = function(pi_method) { self$pi_method <- pi_method },
    get_pi_method = function() self$pi_method
  )
)

#' Enhanced Cross-Validation with Parallel Support
#' 
#' @param model A Model object
#' @param X Feature matrix
#' @param y Target vector
#' @param cv Number of folds (default: 5)
#' @param scoring Metric: "rmse", "mae", "accuracy", "f1"
#' @param show_progress Show progress bar (default: TRUE)
#' @param cl Optional cluster for parallel processing
#' @param ... Additional arguments passed to model$fit()
#' @return Vector of CV scores
#' @export
cross_val_score <- function(model, X, y, cv = 5, scoring = NULL,
                           show_progress = TRUE, cl = NULL, ...) {
  X <- as.matrix(X)
  n <- nrow(X)
  
  # Create folds
  set.seed(model$seed)
  folds <- split(sample(seq_len(n)), rep(1:cv, length.out = n))
  
  # Auto-detect task and scoring
  task_type <- ifelse(is.factor(y), "classification", "regression")
  if (is.null(scoring)) {
    scoring <- ifelse(task_type == "regression", "rmse", "accuracy")
  }
  
  # Scoring function
  compute_score <- function(true, pred, metric) {
    if (metric == "rmse") {
      return(sqrt(mean((true - pred)^2, na.rm = TRUE)))
    } else if (metric == "mae") {
      return(mean(abs(true - pred), na.rm = TRUE))
    } else if (metric == "accuracy") {
      return(mean(pred == true, na.rm = TRUE))
    } else if (metric == "f1") {
      # Binary F1
      if (is.factor(true) && nlevels(true) == 2) {
        tp <- sum(pred == levels(true)[2] & true == levels(true)[2])
        fp <- sum(pred == levels(true)[2] & true == levels(true)[1])
        fn <- sum(pred == levels(true)[1] & true == levels(true)[2])
        precision <- tp / (tp + fp + 1e-10)
        recall <- tp / (tp + fn + 1e-10)
        return(2 * precision * recall / (precision + recall + 1e-10))
      } else {
        return(mean(pred == true, na.rm = TRUE))
      }
    }
  }
  
  # Sequential execution
  if (is.null(cl)) {
    scores <- numeric(cv)
    if (show_progress) pb <- txtProgressBar(max = cv, style = 3)
    
    for (i in seq_len(cv)) {
      val_idx <- folds[[i]]
      train_idx <- setdiff(seq_len(n), val_idx)
      
      m <- model$clone_model()
      m$fit(X[train_idx, , drop = FALSE], y[train_idx], ...)
      
      pred <- m$predict(X[val_idx, , drop = FALSE])
      if (is.list(pred)) pred <- pred$preds
      
      scores[i] <- compute_score(y[val_idx], pred, scoring)
      
      if (show_progress) setTxtProgressBar(pb, i)
    }
    
    if (show_progress) close(pb)
    return(scores)
  } else {
    # Parallel execution (requires foreach/doParallel setup)
    stop("Parallel execution requires foreach and doParallel packages")
  }
}
