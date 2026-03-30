#' Conformal Prediction Wrapper for Unified Machine Learning Models
#'
#' Wraps a Model object to provide conformal prediction intervals (regression)
#' or prediction sets (classification) with guaranteed marginal coverage.
#' Implements split conformal prediction with automatic data splitting.
#'
#' @name ConformalModel
#' @docType class
#' @author Your Name
#'
#' @import R6
#' @importFrom stats quantile
NULL

#' @title Conformal Prediction Model
#' @description
#' An R6 class that wraps a \code{Model} object to add conformal prediction
#' capabilities. Uses split conformal prediction to generate prediction
#' intervals (regression) or prediction sets (classification) with
#' guaranteed (1 - alpha) marginal coverage.
#'
#' @field base_model The wrapped Model object
#' @field alpha Significance level (coverage = 1 - alpha)
#' @field quantile Critical non-conformity score threshold
#' @field task Type of task: "regression" or "classification"
#' @field classes Class levels (for classification)
#' @field calib_scores Non-conformity scores from calibration set
#' @field is_calibrated Logical indicating if calibration is complete
#'
#' @examples
#' \donttest{
#' # Regression example
#' library(glmnet)
#' set.seed(123)
#' X <- matrix(rnorm(200), ncol = 4)
#' y <- 2*X[,1] - 1.5*X[,2] + rnorm(50)
#'
#' cmod <- ConformalModel$new(glmnet::glmnet, alpha = 0.1)
#' cmod$fit(X, y, alpha = 0, lambda = 0.1, cal_size = 0.3)
#' preds <- cmod$predict(X[1:5, ])
#' print(preds)
#'
#' # Classification example
#' data(iris)
#' iris_bin <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#' Xc <- as.matrix(iris_bin[, 1:4])
#' yc <- iris_bin$Species
#'
#' cmod2 <- ConformalModel$new(e1071::svm, alpha = 0.1)
#' cmod2$fit(Xc, yc, kernel = "radial", probability = TRUE, cal_size = 0.3)
#' sets <- cmod2$predict(Xc[1:5, ], type = "set")
#' print(sets)
#' }
#' @export
ConformalModel <- R6::R6Class(
  "ConformalModel",
  
  public = list(
    base_model    = NULL,
    alpha         = NULL,
    quantile      = NULL,
    task          = NULL,
    classes       = NULL,
    calib_scores  = NULL,
    is_calibrated = FALSE,
    
    #' @description Initialize the conformal prediction model
    #' @param model_fn Modeling function (e.g., glmnet::glmnet, randomForest::randomForest)
    #' @param alpha Significance level (default 0.1 gives 90\% coverage)
    #' @return A new ConformalModel object
    initialize = function(model_fn, alpha = 0.1) {
      stopifnot(is.function(model_fn))
      stopifnot(is.numeric(alpha), alpha > 0, alpha < 1)
      
      self$alpha      <- alpha
      self$base_model <- Model$new(model_fn)
      self$is_calibrated <- FALSE
    },
    
    #' @description Fit model and calibrate conformal scores
    #'
    #' Splits data into training and calibration sets. Fits the base model
    #' on the training set and computes non-conformity scores on the
    #' calibration set.
    #'
    #' @param X Feature matrix or data.frame
    #' @param y Target vector (numeric for regression, factor for classification)
    #' @param cal_size Proportion of data to use for calibration (default: 0.25)
    #' @param seed Optional integer seed for the train/calibration split.
    #'   Pass \code{NULL} (default) to leave the global RNG untouched.
    #' @param ... Additional arguments passed to the base model's fit method
    #' @return self (invisible) for method chaining
    fit = function(X, y, cal_size = 0.25, seed = NULL, ...) {
      X <- as.matrix(X)
      n <- nrow(X)
      n_cal <- max(1, floor(n * cal_size))
      
      # Optionally seed the split without corrupting the global RNG state
      if (!is.null(seed)) {
        old_seed <- .Random.seed
        on.exit(.Random.seed <<- old_seed, add = TRUE)
        set.seed(seed)
      }
      
      cal_idx   <- sample(seq_len(n), n_cal)
      train_idx <- setdiff(seq_len(n), cal_idx)
      
      X_train <- X[train_idx, , drop = FALSE]
      y_train <- y[train_idx]
      X_cal   <- X[cal_idx,   , drop = FALSE]
      y_cal   <- y[cal_idx]
      
      # Fit base model on training set
      self$base_model$fit(X_train, y_train, ...)
      self$task    <- self$base_model$task
      self$classes <- if (self$task == "classification") {
        levels(self$base_model$y_train)
      } else {
        NULL
      }
      
      # Compute calibration scores and quantile threshold
      self$calib_scores <- private$compute_scores(X_cal, y_cal)
      private$compute_and_store_quantile()
      self$is_calibrated <- TRUE
      
      invisible(self)
    },
    
    #' @description Fit with pre-specified calibration set
    #'
    #' Alternative to \code{fit()} when you want to explicitly control the
    #' training/calibration split.
    #'
    #' @param X_train Training feature matrix
    #' @param y_train Training target vector
    #' @param X_calib Calibration feature matrix
    #' @param y_calib Calibration target vector
    #' @param ... Additional arguments passed to the base model's fit method
    #' @return self (invisible) for method chaining
    fit_calibrate = function(X_train, y_train, X_calib, y_calib, ...) {
      X_train <- as.matrix(X_train)
      X_calib <- as.matrix(X_calib)
      
      # Fit base model on training set
      self$base_model$fit(X_train, y_train, ...)
      self$task    <- self$base_model$task
      self$classes <- if (self$task == "classification") {
        levels(self$base_model$y_train)
      } else {
        NULL
      }
      
      # Compute calibration scores and quantile threshold
      self$calib_scores <- private$compute_scores(X_calib, y_calib)
      private$compute_and_store_quantile()
      self$is_calibrated <- TRUE
      
      invisible(self)
    },
    
    #' @description Generate conformal predictions
    #'
    #' @param X New feature matrix for prediction
    #' @param type Prediction type:
    #'   \itemize{
    #'     \item For regression: \code{"interval"} (default) or \code{"point"}
    #'     \item For classification: \code{"set"} (default), \code{"point"},
    #'       or \code{"prob"}
    #'   }
    #' @param ... Additional arguments passed to the base model's predict method
    #' @return
    #'   \itemize{
    #'     \item Regression \code{"interval"}: data.frame with columns
    #'       \code{lower}, \code{fit}, \code{upper}
    #'     \item Regression \code{"point"}: numeric vector
    #'     \item Classification \code{"set"}: list of character vectors
    #'     \item Classification \code{"point"}: factor vector
    #'     \item Classification \code{"prob"}: numeric matrix
    #'   }
    predict = function(X, type = NULL, ...) {
      private$check_calibrated()
      X <- as.matrix(X)
      private$check_dimensions(X)
      
      if (self$task == "regression") {
        private$predict_regression(X, type, ...)
      } else {
        private$predict_classification(X, type, ...)
      }
    },
    
    #' @description Evaluate empirical coverage on a test set
    #'
    #' @param X_test Test feature matrix
    #' @param y_test Test target vector
    #' @return Named list with coverage statistics
    evaluate = function(X_test, y_test) {
      private$check_calibrated()
      X_test <- as.matrix(X_test)
      private$check_dimensions(X_test)
      
      if (self$task == "regression") {
        preds   <- self$predict(X_test, type = "interval")
        covered <- (y_test >= preds$lower) & (y_test <= preds$upper)
        widths  <- preds$upper - preds$lower
        
        list(
          coverage        = mean(covered),
          target_coverage = 1 - self$alpha,
          mean_width      = mean(widths),
          median_width    = median(widths),
          sd_width        = sd(widths),
          covered         = covered
        )
      } else {
        y_char  <- as.character(y_test)
        sets    <- self$predict(X_test, type = "set")
        covered <- mapply(
          function(true_label, pred_set) true_label %in% pred_set,
          y_char, sets
        )
        set_sizes <- lengths(sets)  # faster than sapply(sets, length)
        
        list(
          coverage         = mean(covered),
          target_coverage  = 1 - self$alpha,
          mean_set_size    = mean(set_sizes),
          median_set_size  = median(set_sizes),
          sd_set_size      = sd(set_sizes),
          covered          = covered,
          set_sizes        = set_sizes
        )
      }
    },
    
    #' @description Plot non-conformity score distribution
    #'
    #' Visualises the distribution of calibration scores and marks the
    #' critical quantile threshold.
    #'
    #' @param alpha_display Optional significance level to display (overrides
    #'   \code{self$alpha} for the plot only; does not change the model).
    #' @return self (invisible) for method chaining
    plot_scores = function(alpha_display = NULL) {
      private$check_calibrated()
      
      alpha_use <- if (is.null(alpha_display)) self$alpha else alpha_display
      n_scores  <- length(self$calib_scores)
      q_level   <- private$safe_q_level(n_scores, alpha_use)
      q_val     <- quantile(self$calib_scores, probs = q_level, names = FALSE)
      
      hist(self$calib_scores,
           breaks = 30,
           main   = "Non-conformity Score Distribution",
           xlab   = "Score",
           col    = "lightsteelblue",
           border = "white",
           freq   = FALSE)
      
      rug(self$calib_scores, col = rgb(0, 0, 0, 0.3))
      abline(v = q_val, col = "firebrick", lwd = 2, lty = 2)
      
      legend("topright",
             legend = sprintf("q (\u03b1 = %.2f) = %.3f", alpha_use, q_val),
             col    = "firebrick",
             lty    = 2,
             lwd    = 2,
             bty    = "n")
      
      invisible(self)
    },
    
    #' @description Plot conformal prediction intervals (regression only)
    #'
    #' Visualises conformal prediction intervals on test data, sorted by
    #' fitted value.
    #'
    #' @param X_test Test feature matrix
    #' @param y_test Test target vector
    #' @return self (invisible) for method chaining
    plot_intervals = function(X_test, y_test) {
      if (self$task != "regression") {
        stop("plot_intervals() is only available for regression tasks.")
      }
      private$check_calibrated()
      
      preds   <- self$predict(X_test, type = "interval")
      ord     <- order(preds$fit)
      y_range <- range(c(preds$lower, preds$upper, y_test))
      
      plot(seq_along(ord), y_test[ord],
           ylim = y_range,
           xlab = "Sample (sorted by predicted value)",
           ylab = "Target",
           main = sprintf("Conformal Prediction Intervals (%.0f%% coverage)",
                          100 * (1 - self$alpha)),
           pch  = 16,
           col  = rgb(0, 0, 0, 0.5),
           cex  = 0.7)
      
      polygon(c(seq_along(ord), rev(seq_along(ord))),
              c(preds$lower[ord], rev(preds$upper[ord])),
              col    = rgb(0.4, 0.6, 0.9, 0.25),
              border = NA)
      
      lines(seq_along(ord), preds$fit[ord], col = "steelblue", lwd = 2)
      
      empirical_cov <- mean(y_test >= preds$lower & y_test <= preds$upper)
      legend("topleft",
             legend = sprintf("Empirical coverage: %.1f%%", 100 * empirical_cov),
             bty    = "n")
      
      invisible(self)
    },
    
    #' @description Print a summary of the conformal model
    #' @return self (invisible) for method chaining
    print = function() {
      cat("Conformal Prediction Model\n")
      cat("==========================\n")
      cat("Base model task:", if (is.null(self$task)) "(not fitted)" else self$task, "\n")
      cat("Significance level (\u03b1):", self$alpha, "\n")
      cat("Target coverage:", 1 - self$alpha, "\n")
      
      if (self$is_calibrated) {
        cat("\nCalibration:\n")
        cat("  - Samples:", length(self$calib_scores), "\n")
        cat("  - Quantile (q):", round(self$quantile, 4), "\n")
        cat("  - Score range:",
            round(min(self$calib_scores), 4), "to",
            round(max(self$calib_scores), 4), "\n")
        
        if (self$task == "classification") {
          cat("  - Classes:", paste(self$classes, collapse = ", "), "\n")
        }
      } else {
        cat("\nStatus: Not calibrated\n")
        cat("Run $fit() or $fit_calibrate() to calibrate.\n")
      }
      
      invisible(self)
    }
  ),
  
  private = list(
    
    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    
    check_calibrated = function() {
      if (!self$is_calibrated) {
        stop("Model not calibrated. Run $fit() or $fit_calibrate() first.")
      }
    },
    
    check_dimensions = function(X) {
      expected <- ncol(self$base_model$X_train)
      if (!is.null(expected) && ncol(X) != expected) {
        stop(sprintf(
          "X has %d column(s) but the model was trained on %d.",
          ncol(X), expected
        ))
      }
    },
    
    # ------------------------------------------------------------------
    # Quantile helpers  (single source of truth — fixes duplication bug)
    # ------------------------------------------------------------------
    
    #' Clamp quantile level to [0, 1] and warn when calibration set is small
    safe_q_level = function(n_scores, alpha_use) {
      raw <- ceiling((n_scores + 1) * (1 - alpha_use)) / n_scores
      q   <- min(1, max(0, raw))
      
      # Warn when the calibration set is too small for the requested alpha
      min_n <- ceiling(1 / alpha_use) - 1
      if (n_scores < min_n) {
        warning(sprintf(
          paste0("Calibration set (%d samples) may be too small for ",
                 "alpha = %.3f (recommended >= %d). ",
                 "Coverage guarantee may not hold."),
          n_scores, alpha_use, min_n
        ))
      }
      
      q
    },
    
    compute_and_store_quantile = function() {
      n_scores      <- length(self$calib_scores)
      q_level       <- private$safe_q_level(n_scores, self$alpha)
      self$quantile <- quantile(self$calib_scores, probs = q_level, names = FALSE)
    },
    
    # ------------------------------------------------------------------
    # Non-conformity scores
    # ------------------------------------------------------------------
    
    compute_scores = function(X, y) {
      if (self$task == "regression") {
        preds <- self$base_model$predict(X)
        abs(as.numeric(y) - as.numeric(preds))
      } else {
        probs  <- private$get_probabilities(X)
        y_char <- as.character(y)
        cols   <- colnames(probs)
        
        sapply(seq_len(nrow(probs)), function(i) {
          idx <- match(y_char[i], cols)
          if (is.na(idx)) {
            warning("Class '", y_char[i], "' not found in probability columns.")
            return(1)   # worst-case score: guarantees conservative coverage
          }
          1 - probs[i, idx]
        })
      }
    },
    
    # ------------------------------------------------------------------
    # Probability extraction with transparent fallback
    # ------------------------------------------------------------------
    
    get_probabilities = function(X) {
      K <- length(self$classes)
      n <- nrow(X)
      
      # Attempt 1: type = "prob"
      probs <- tryCatch({
        p <- self$base_model$predict(X, type = "prob")
        private$validate_prob_matrix(p, n, K)
      }, error = function(e) NULL)
      
      # Attempt 2: type = "probabilities"
      if (is.null(probs)) {
        probs <- tryCatch({
          p <- self$base_model$predict(X, type = "probabilities")
          private$validate_prob_matrix(p, n, K)
        }, error = function(e) NULL)
      }
      
      # Fallback: hard predictions — warn because this breaks score semantics
      if (is.null(probs)) {
        warning(paste0(
          "Could not obtain class probabilities from the base model. ",
          "Falling back to hard predictions (all scores will be 0 or 1). ",
          "Coverage guarantee and set quality may be severely degraded. ",
          "Ensure the model supports type = 'prob' or type = 'probabilities'."
        ))
        hard  <- self$base_model$predict(X, type = "class")
        probs <- matrix(0, nrow = n, ncol = K,
                        dimnames = list(NULL, self$classes))
        for (i in seq_len(n)) {
          idx <- match(as.character(hard[i]), self$classes)
          if (!is.na(idx)) probs[i, idx] <- 1
        }
      }
      
      colnames(probs) <- self$classes
      probs
    },
    
    # Validate and normalise a candidate probability matrix/vector
    validate_prob_matrix = function(p, n, K) {
      if (is.matrix(p) && nrow(p) == n && ncol(p) == K) return(p)
      if (is.data.frame(p) && nrow(p) == n && ncol(p) == K) return(as.matrix(p))
      if (is.vector(p) && length(p) == n && K == 2) {
        mat        <- matrix(cbind(1 - p, p), nrow = n, ncol = 2)
        colnames(mat) <- self$classes
        return(mat)
      }
      NULL  # signal failure to the caller
    },
    
    # ------------------------------------------------------------------
    # Prediction dispatch
    # ------------------------------------------------------------------
    
    predict_regression = function(X, type, ...) {
      point_preds <- as.numeric(self$base_model$predict(X, ...))
      
      if (is.null(type) || type == "interval") {
        data.frame(
          lower = point_preds - self$quantile,
          fit   = point_preds,
          upper = point_preds + self$quantile
        )
      } else if (type == "point") {
        point_preds
      } else {
        stop("Unknown type for regression. Use 'interval' or 'point'.")
      }
    },
    
    predict_classification = function(X, type, ...) {
      if (is.null(type) || type == "set") {
        probs     <- private$get_probabilities(X)  # ... not needed here
        threshold <- 1 - self$quantile
        
        lapply(seq_len(nrow(probs)), function(i) {
          row_probs <- probs[i, ]
          included  <- self$classes[row_probs >= threshold]
          
          if (length(included) == 0L) {
            # Return empty set — preserving coverage semantics.
            # The caller can detect this with lengths(sets) == 0.
            character(0)
          } else {
            included
          }
        })
      } else if (type == "point") {
        self$base_model$predict(X, type = "class", ...)
      } else if (type == "prob") {
        private$get_probabilities(X)
      } else {
        stop("Unknown type for classification. Use 'set', 'point', or 'prob'.")
      }
    }
  )
)