#' Gradient Boosting Model using Model Interface
#' 
#' An R6 class that wraps the C++ boost_regression implementation to provide
#' a unified interface consistent with the Model class. Supports any base
#' learner that implements the Model interface.
#' 
#' @field base_learner_fn Function that returns a new Model object for base learning
#' @field B Number of boosting iterations
#' @field eta Learning rate (shrinkage parameter)
#' @field model_args Named list of arguments to pass to base learner's fit()
#' @field verbose Whether to print progress during fitting
#' @field boost_obj The fitted boosting object from C++
#' 
#' @export
BoostingModel <- R6::R6Class(
  "BoostingModel",
  inherit = Model,
  
  public = list(
    base_learner_fn = NULL,
    B = NULL,
    eta = NULL,
    model_args = NULL,
    verbose = NULL,
    boost_obj = NULL,
    
    #' @description Initialize a new BoostingModel
    #' @param base_learner_fn Function that returns a new Model object
    #' @param B Number of boosting iterations (default: 100)
    #' @param eta Learning rate / shrinkage parameter (default: 0.1)
    #' @param model_args Named list of arguments for base learner (default: list())
    #' @param verbose Whether to print progress (default: TRUE)
    #' @return A new BoostingModel object
    initialize = function(base_learner_fn, B = 100, eta = 0.1, 
                          model_args = list(), verbose = TRUE) {
      stopifnot(is.function(base_learner_fn))
      stopifnot(B > 0)
      stopifnot(eta > 0 && eta <= 1)
      
      self$base_learner_fn <- base_learner_fn
      self$B <- B
      self$eta <- eta
      self$model_args <- model_args
      self$verbose <- verbose
      
      # Set task to regression (boosting implementation is for regression)
      self$task <- "regression"
    },
    
    #' @description Fit the boosting model to training data
    #' 
    #' Uses gradient boosting with squared error loss. The base learner
    #' is fit to residuals at each iteration.
    #' 
    #' @param X Feature matrix or data.frame
    #' @param y Numeric target vector
    #' @param ... Additional arguments passed to base learner's fit()
    #' @return self (invisible) for method chaining
    fit = function(X, y, ...) {
      X <- as.matrix(X)
      y <- as.numeric(y)
      
      # Store training data
      self$X_train <- X
      self$y_train <- y
      
      # Merge additional arguments with stored model_args
      # Additional args from ... take precedence
      extra_args <- list(...)
      combined_args <- c(self$model_args, extra_args)
      # Remove duplicates, keeping later ones
      combined_args <- combined_args[!duplicated(names(combined_args), 
                                                 fromLast = TRUE)]
      
      # Call C++ boosting function
      self$boost_obj <- boost_regression(
        model_creator = self$base_learner_fn,
        X = X,
        y = y,
        B = self$B,
        model_args = combined_args,
        eta = self$eta,
        verbose = self$verbose
      )
      
      # Store the fitted boosting object as self$fitted for consistency
      self$fitted <- self$boost_obj
      
      invisible(self)
    },
    
    #' @description Generate predictions from fitted boosting model
    #' @param X Feature matrix for prediction
    #' @param ... Additional arguments (not used, for consistency)
    #' @return Vector of predictions
    predict = function(X, ...) {
      if (is.null(self$boost_obj)) {
        stop("Model not fitted. Call fit() first.")
      }
      
      X <- as.matrix(X)
      predict_boost(self$boost_obj, X)
    },
    
    #' @description Print boosting model information
    #' @return self (invisible) for method chaining
    print = function() {
      cat("Boosting Model (Gradient Boosting for Regression)\n")
      cat("==================================================\n")
      cat("Base learner: Custom function\n")
      cat("Iterations (B):", self$B, "\n")
      cat("Learning rate (eta):", self$eta, "\n")
      cat("Fitted:", !is.null(self$boost_obj), "\n")
      
      if (!is.null(self$boost_obj)) {
        cat("\nTraining Information:\n")
        cat("  Samples:", nrow(self$X_train), "\n")
        cat("  Features:", ncol(self$X_train), "\n")
        
        # Compute final training RMSE
        final_rmse <- sqrt(mean((self$y_train - self$boost_obj$f_hat)^2))
        cat("  Final Training RMSE:", round(final_rmse, 4), "\n")
      }
      
      invisible(self)
    },
    
    #' @description Compute and display variable importance
    #' 
    #' Uses correlation-based importance weighted by SSE improvement
    #' at each boosting iteration.
    #' 
    #' @param normalize Whether to normalize to sum to 100 (default: TRUE)
    #' @return Data frame with variable importance (invisible)
    variable_importance = function(normalize = TRUE) {
      if (is.null(self$boost_obj)) {
        stop("Model not fitted. Call fit() first.")
      }
      
      imp <- variable_importance_boost_with_X(
        self$boost_obj, 
        self$X_train, 
        normalize = normalize
      )
      
      # Create nice output
      feature_names <- colnames(self$X_train)
      if (is.null(feature_names)) {
        feature_names <- paste0("X", seq_len(length(imp)))
      }
      
      imp_df <- data.frame(
        Feature = feature_names,
        Importance = imp,
        stringsAsFactors = FALSE
      )
      imp_df <- imp_df[order(-imp_df$Importance), ]
      rownames(imp_df) <- NULL
      
      cat("\nVariable Importance\n")
      cat("===================\n")
      print(imp_df, row.names = FALSE)
      
      invisible(imp_df)
    },
    
    #' @description Plot training loss history
    #' 
    #' Shows how mean squared error decreases over boosting iterations.
    #' 
    #' @param log_scale Whether to use log scale for y-axis (default: FALSE)
    #' @return self (invisible) for method chaining
    plot_loss = function(log_scale = FALSE) {
      if (is.null(self$boost_obj)) {
        stop("Model not fitted. Call fit() first.")
      }
      
      loss_hist <- compute_loss_history(self$boost_obj)
      iterations <- 0:self$B
      
      y_lab <- if (log_scale) "MSE (log scale)" else "MSE"
      log_arg <- if (log_scale) "y" else ""
      
      plot(iterations, loss_hist, 
           type = "l", lwd = 2, col = "steelblue",
           xlab = "Boosting Iteration", 
           ylab = y_lab,
           main = "Training Loss History",
           log = log_arg)
      grid()
      
      # Add initial and final MSE annotations
      text(0, loss_hist[1], 
           labels = sprintf("Initial: %.4f", loss_hist[1]),
           pos = 4, cex = 0.8, col = "darkred")
      text(self$B, loss_hist[self$B + 1], 
           labels = sprintf("Final: %.4f", loss_hist[self$B + 1]),
           pos = 2, cex = 0.8, col = "darkgreen")
      
      invisible(self)
    },
    
    #' @description Enhanced summary with boosting-specific information
    #' 
    #' Displays variable importance and loss reduction information
    #' in addition to standard model summary.
    #' 
    #' @param show_importance Whether to show variable importance (default: TRUE)
    #' @return Summary data frame (invisible)
    summary = function(show_importance = TRUE) {
      if (is.null(self$boost_obj)) {
        stop("Model not fitted. Call fit() first.")
      }
      
      cat("\nBoosting Model Summary\n")
      cat("======================\n")
      cat("Task: Regression (Gradient Boosting)\n")
      cat("Base learner iterations:", self$B, "\n")
      cat("Learning rate (eta):", self$eta, "\n")
      cat("Training samples:", nrow(self$X_train), "\n")
      cat("Features:", ncol(self$X_train), "\n\n")
      
      # Loss reduction
      loss_hist <- compute_loss_history(self$boost_obj)
      initial_mse <- loss_hist[1]
      final_mse <- loss_hist[self$B + 1]
      reduction <- (initial_mse - final_mse) / initial_mse * 100
      
      cat("Loss Information:\n")
      cat("  Initial MSE:", round(initial_mse, 4), "\n")
      cat("  Final MSE:", round(final_mse, 4), "\n")
      cat("  Reduction:", round(reduction, 2), "%\n")
      cat("  Final RMSE:", round(sqrt(final_mse), 4), "\n\n")
      
      # Variable importance
      if (show_importance) {
        imp_df <- self$variable_importance(normalize = TRUE)
        return(invisible(imp_df))
      }
      
      invisible(NULL)
    },
    
    #' @description Create a deep copy of the boosting model
    #' 
    #' Creates a new instance with same configuration but no fitted data.
    #' Useful for cross-validation.
    #' 
    #' @return A new BoostingModel object with same configuration
    clone_model = function() {
      BoostingModel$new(
        base_learner_fn = self$base_learner_fn,
        B = self$B,
        eta = self$eta,
        model_args = self$model_args,
        verbose = FALSE  # Disable verbose for CV
      )
    },
    
    #' @description Get base learner models
    #' 
    #' Access individual base learner models from the ensemble.
    #' Useful for inspection and debugging.
    #' 
    #' @param indices Vector of iteration indices (default: all)
    #' @return List of base learner Model objects
    get_base_learners = function(indices = NULL) {
      if (is.null(self$boost_obj)) {
        stop("Model not fitted. Call fit() first.")
      }
      
      if (is.null(indices)) {
        return(self$boost_obj$models)
      } else {
        return(self$boost_obj$models[indices])
      }
    }
  )
)