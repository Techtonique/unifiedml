# ============================================================================
# FIXED PROBABILITY EXTRACTOR - Handles GLM and all model types properly
# ============================================================================

#' Extract probability predictions from any R model in a standardized format
#' 
#' @param model Fitted model object (any class)
#' @param X Feature matrix or data.frame for predictions
#' @param y_train Optional training labels (for class names)
#' @param verbose Print diagnostic information
#' @return Probability matrix [n_samples × n_classes] with column names as class labels
#' 
#' @export
extract_probabilities <- function(model, X, y_train = NULL, verbose = FALSE) {
  
  # Helper: Convert X to appropriate format for prediction
  prepare_X <- function(X, model) {
    # Check if model expects formula interface (needs data.frame)
    if (inherits(model, c("glm", "lm", "rpart", "nnet"))) {
      if (is.matrix(X)) {
        X <- as.data.frame(X)
      }
    }
    return(X)
  }
  
  # Helper: Get class labels from training data
  get_class_labels <- function(y) {
    if (is.factor(y)) {
      return(levels(y))
    } else if (is.character(y)) {
      return(sort(unique(y)))
    } else if (is.numeric(y)) {
      return(c("0", "1"))
    } else {
      return(NULL)
    }
  }
  
  # Helper: Standardize to probability matrix
  standardize_to_probs <- function(pred, y_train) {
    # Case 1: Perfect probability matrix
    if (is.matrix(pred) && all(pred >= 0) && all(pred <= 1)) {
      row_sums <- rowSums(pred)
      if (all(abs(row_sums - 1) < 1e-6)) {
        if (verbose) message("   Perfect probability matrix")
        if (is.null(colnames(pred)) && !is.null(y_train)) {
          colnames(pred) <- get_class_labels(y_train)
        }
        return(pred)
      }
    }
    
    # Case 2: Numeric vector (binary probability)
    if (is.numeric(pred) && !is.matrix(pred)) {
      if (all(pred >= 0 & pred <= 1)) {
        if (verbose) message("   Binary probability vector")
        n_classes <- if (!is.null(y_train)) nlevels(y_train) else 2
        prob_matrix <- matrix(0, nrow = length(pred), ncol = n_classes)
        prob_matrix[, n_classes] <- pred
        prob_matrix[, 1] <- 1 - pred
        if (!is.null(y_train)) {
          colnames(prob_matrix) <- get_class_labels(y_train)
        } else {
          colnames(prob_matrix) <- c("Class0", "Class1")
        }
        return(prob_matrix)
      } else {
        # Logits -> sigmoid
        if (verbose) message("   Logit vector -> sigmoid")
        probs <- 1 / (1 + exp(-pred))
        return(standardize_to_probs(probs, y_train))
      }
    }
    
    # Case 3: Factor (hard classes) -> one-hot
    if (is.factor(pred)) {
      if (verbose) message("   Hard classes -> one-hot (less informative)")
      classes <- levels(pred)
      prob_matrix <- matrix(0, nrow = length(pred), ncol = length(classes))
      for (i in seq_along(pred)) {
        prob_matrix[i, which(classes == pred[i])] <- 1
      }
      colnames(prob_matrix) <- classes
      return(prob_matrix)
    }
    
    # Case 4: Matrix that needs softmax
    if (is.matrix(pred)) {
      row_sums <- rowSums(pred)
      if (any(abs(row_sums - 1) > 1e-6)) {
        if (all(pred >= 0)) {
          if (verbose) message("   Raw scores -> softmax")
          prob_matrix <- pred / row_sums
        } else {
          if (verbose) message("   Logit matrix -> softmax")
          exp_pred <- exp(pred)
          prob_matrix <- exp_pred / rowSums(exp_pred)
        }
        if (!is.null(y_train)) {
          colnames(prob_matrix) <- get_class_labels(y_train)
        }
        return(prob_matrix)
      }
    }
    
    # Case 5: List with probabilities element
    if (is.list(pred) && !is.null(pred$probabilities)) {
      if (verbose) message("   List with 'probabilities' element")
      return(standardize_to_probs(pred$probabilities, y_train))
    }
    
    # Last resort: coerce and hope
    if (verbose) message("   Unknown format, attempting coercion")
    pred <- as.matrix(pred)
    return(standardize_to_probs(pred, y_train))
  }
  
  # Main extraction logic
  if (verbose) message("Model class: ", paste(class(model), collapse = ", "))
  
  # Prepare X in the right format
  X_pred <- prepare_X(X, model)
  
  prediction <- NULL
  method <- NULL
  
  # Model-specific optimizations
  model_class <- class(model)[1]
  
  # 1. randomForest
  if (inherits(model, "randomForest") && model$type == "classification") {
    prediction <- tryCatch({
      predict(model, X, type = "prob")
    }, error = function(e) NULL)
    if (!is.null(prediction)) method <- "randomForest::predict(type='prob')"
  }
  
  # 2. GLM - FIXED: Handle data frame requirement
  if (is.null(prediction) && inherits(model, "glm")) {
    if (family(model)$family == "binomial") {
      prediction <- tryCatch({
        # Ensure X is data.frame for GLM
        X_df <- as.data.frame(X)
        predict(model, newdata = X_df, type = "response")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "glm::predict(type='response')"
    } else if (family(model)$family == "multinomial") {
      prediction <- tryCatch({
        X_df <- as.data.frame(X)
        predict(model, newdata = X_df, type = "probs")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "glm::predict(type='probs')"
    }
  }
  
  # 3. SVM (e1071)
  if (is.null(prediction) && inherits(model, "svm")) {
    if (model$type == "C-classification") {
      prediction <- tryCatch({
        pred_obj <- predict(model, X, probability = TRUE)
        attr(pred_obj, "probabilities")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "e1071::svm with probability=TRUE"
    }
  }
  
  # 4. nnet
  if (is.null(prediction) && inherits(model, "nnet")) {
    prediction <- tryCatch({
      X_df <- as.data.frame(X)
      predict(model, newdata = X_df, type = "raw")
    }, error = function(e) NULL)
    if (!is.null(prediction)) method <- "nnet::predict(type='raw')"
  }
  
  # 5. rpart
  if (is.null(prediction) && inherits(model, "rpart")) {
    prediction <- tryCatch({
      X_df <- as.data.frame(X)
      predict(model, newdata = X_df, type = "prob")
    }, error = function(e) NULL)
    if (!is.null(prediction)) method <- "rpart::predict(type='prob')"
  }
  
  # 6. xgboost
  if (is.null(prediction) && inherits(model, "xgb.Booster")) {
    prediction <- tryCatch({
      # xgboost needs matrix
      X_mat <- as.matrix(X)
      pred <- predict(model, X_mat, type = "prob")
      if (is.null(dim(pred))) {
        pred <- cbind(1 - pred, pred)
      }
      pred
    }, error = function(e) NULL)
    if (!is.null(prediction)) method <- "xgboost::predict(type='prob')"
  }
  
  # 7. glmnet
  if (is.null(prediction) && inherits(model, "glmnet")) {
    prediction <- tryCatch({
      X_mat <- as.matrix(X)
      pred <- predict(model, X_mat, type = "response")
      if (is.matrix(pred) && ncol(pred) == 1) {
        pred <- cbind(1 - pred, pred)
      }
      pred
    }, error = function(e) NULL)
    if (!is.null(prediction)) method <- "glmnet::predict(type='response')"
  }
  
  # 8. ksvm (kernlab)
  if (is.null(prediction) && inherits(model, "ksvm")) {
    prediction <- tryCatch({
      X_mat <- as.matrix(X)
      predict(model, X_mat, type = "probabilities")
    }, error = function(e) NULL)
    if (!is.null(prediction)) method <- "kernlab::ksvm predict(type='probabilities')"
  }
  
  # 9. Generic fallback methods (IMPROVED)
  if (is.null(prediction)) {
    if (verbose) message("  Using generic fallback...")
    
    # Try different prediction interfaces
    # First, try with newdata (formula interface)
    if (is.null(prediction)) {
      prediction <- tryCatch({
        X_df <- as.data.frame(X)
        predict(model, newdata = X_df, type = "response")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "predict(newdata=..., type='response')"
    }
    
    # Try with newx (matrix interface)
    if (is.null(prediction)) {
      prediction <- tryCatch({
        X_mat <- as.matrix(X)
        predict(model, newx = X_mat, type = "response")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "predict(newx=..., type='response')"
    }
    
    # Try without newdata (assumes X is correct format)
    if (is.null(prediction)) {
      prediction <- tryCatch({
        predict(model, X, type = "response")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "predict(X, type='response')"
    }
    
    # Try type = "prob"
    if (is.null(prediction)) {
      prediction <- tryCatch({
        X_df <- as.data.frame(X)
        predict(model, newdata = X_df, type = "prob")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "predict(newdata=..., type='prob')"
    }
    
    # Try type = "raw"
    if (is.null(prediction)) {
      prediction <- tryCatch({
        X_df <- as.data.frame(X)
        predict(model, newdata = X_df, type = "raw")
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "predict(newdata=..., type='raw')"
    }
    
    # Try probability = TRUE
    if (is.null(prediction)) {
      prediction <- tryCatch({
        pred_obj <- predict(model, X, probability = TRUE)
        if (is.matrix(attr(pred_obj, "probabilities"))) {
          attr(pred_obj, "probabilities")
        } else {
          NULL
        }
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "predict(probability=TRUE)"
    }
    
    # Last resort: direct prediction
    if (is.null(prediction)) {
      prediction <- tryCatch({
        predict(model, X)
      }, error = function(e) NULL)
      if (!is.null(prediction)) method <- "direct prediction (fallback)"
    }
  }
  
  if (is.null(prediction)) {
    stop("Could not extract predictions from model of class: ", 
         paste(class(model), collapse = ", "))
  }
  
  if (verbose) message("Extraction method: ", method)
  
  # Standardize to probability matrix
  result <- standardize_to_probs(prediction, y_train)
  
  # Final validation
  if (!is.matrix(result)) {
    result <- as.matrix(result)
  }
  
  # Clip to [0,1]
  result[result < 0] <- 0
  result[result > 1] <- 1
  
  # Normalize row sums
  row_sums <- rowSums(result)
  if (any(abs(row_sums - 1) > 1e-4)) {
    if (verbose) message("  Normalizing row sums to 1")
    result <- result / row_sums
  }
  
  # Add metadata
  attr(result, "extraction_method") <- method
  attr(result, "model_class") <- model_class
  
  if (verbose) {
    message("Final output: ", paste(dim(result), collapse = "x"), " matrix")
    message("Classes: ", paste(colnames(result), collapse = ", "))
  }
  
  return(result)
}