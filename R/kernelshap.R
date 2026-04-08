#' Compute SHAP values for unifiedml Model objects
#' 
#' @param model A fitted unifiedml Model object
#' @param X Feature matrix (optional, uses training data if NULL)
#' @param bg_X Background data for SHAP (optional, uses first 20 rows if NULL)
#' @param class_index For classification, which class to explain (NULL = all classes)
#' @param ... Additional arguments passed to kernelshap
#' @return A kernelshap object containing SHAP values
#' 
#' @export
compute_shap_values <- function(model, X = NULL, bg_X = NULL, class_index = NULL, ...) {
  # Check if model is fitted
  if (is.null(model$fitted)) {
    stop("Model not fitted. Please run fit() first.")
  }
  
  # Check if kernelshap is available
  if (!requireNamespace("kernelshap", quietly = TRUE)) {
    stop("Package 'kernelshap' is required for SHAP values. ",
         "Please install it with: install.packages('kernelshap')")
  }
  
  # Use training data if X not provided
  if (is.null(X)) {
    X <- model$X_train
  }
  
  # Ensure X is a matrix
  X <- as.matrix(X)
  
  # Handle column names - use X1, X2, etc. if missing
  if (is.null(colnames(X))) {
    colnames(X) <- paste0("X", 1:ncol(X))
    message("Added column names: ", paste(colnames(X), collapse = ", "))
  }
  
  # Set background data if not provided
  if (is.null(bg_X)) {
    bg_n <- min(20, nrow(X))
    bg_X <- X[1:bg_n, , drop = FALSE]
    message("Using first ", bg_n, " rows as background data")
  } else {
    bg_X <- as.matrix(bg_X)
    # Ensure background data has column names
    if (is.null(colnames(bg_X))) {
      colnames(bg_X) <- colnames(X)
    }
  }
  
  # Create prediction function based on task type
  if (model$task == "regression") {
    pred_fun <- function(object, X_new) {
      # Ensure X_new has column names
      X_new <- as.matrix(X_new)
      if (is.null(colnames(X_new))) {
        colnames(X_new) <- colnames(X)
      }
      
      # Create temporary model for prediction
      temp_mod <- Model$new(model$model_fn)
      temp_mod$fitted <- object
      temp_mod$task <- "regression"
      temp_mod$X_train <- X_new
      
      # Return numeric predictions
      as.numeric(temp_mod$predict(X_new))
    }
    
    # Compute SHAP values for regression
    result <- kernelshap::kernelshap(
      object = model$fitted,
      X = X,
      bg_X = bg_X,
      pred_fun = pred_fun,
      ...
    )
    
  } else {
    # Classification task - return probabilities
    if (is.null(model$y_train)) {
      stop("Training labels (y_train) required for classification SHAP values")
    }
    
    classes <- levels(model$y_train)
    message("Classification with classes: ", paste(classes, collapse = ", "))
    
    pred_fun <- function(object, X_new) {
      # Ensure X_new has column names
      X_new <- as.matrix(X_new)
      if (is.null(colnames(X_new))) {
        colnames(X_new) <- colnames(X)
      }
      
      # Create temporary model for prediction
      temp_mod <- Model$new(model$model_fn)
      temp_mod$fitted <- object
      temp_mod$task <- "classification"
      temp_mod$X_train <- X_new
      temp_mod$y_train <- model$y_train
      
      # Get probability predictions
      pred <- tryCatch({
        # Try to get probabilities
        pred_result <- temp_mod$predict(X_new, type = "response")
        
        # Check what type of prediction we got
        if (is.matrix(pred_result) && ncol(pred_result) == length(classes)) {
          # Already probability matrix
          colnames(pred_result) <- classes
          return(pred_result)
        } else if (is.numeric(pred_result)) {
          # Binary classification with numeric output (probability for class 2)
          prob_matrix <- matrix(0, nrow = length(pred_result), ncol = length(classes))
          prob_matrix[, 2] <- pred_result
          prob_matrix[, 1] <- 1 - pred_result
          colnames(prob_matrix) <- classes
          return(prob_matrix)
        } else if (is.factor(pred_result)) {
          # Hard classes - convert to probabilities (not ideal)
          warning("Model returned hard classes instead of probabilities. ",
                  "SHAP values may be less informative.")
          prob_matrix <- matrix(0, nrow = length(pred_result), ncol = length(classes))
          for(i in seq_along(pred_result)) {
            prob_matrix[i, which(classes == pred_result[i])] <- 1
          }
          colnames(prob_matrix) <- classes
          return(prob_matrix)
        } else {
          stop("Unable to extract probability predictions from model")
        }
      }, error = function(e) {
        stop("Failed to get probability predictions: ", e$message,
             "\nFor classification SHAP, model must support probability predictions.")
      })
      
      return(pred)
    }
    
    # Compute SHAP values for classification
    result <- kernelshap::kernelshap(
      object = model$fitted,
      X = X,
      bg_X = bg_X,
      pred_fun = pred_fun,
      ...
    )
    
    # If class_index specified, extract only that class
    if (!is.null(class_index)) {
      if (is.character(class_index)) {
        if (!class_index %in% classes) {
          stop("class_index '", class_index, "' not found. Available classes: ",
               paste(classes, collapse = ", "))
        }
        class_pos <- which(classes == class_index)
      } else if (is.numeric(class_index)) {
        if (class_index < 1 || class_index > length(classes)) {
          stop("class_index must be between 1 and ", length(classes))
        }
        class_pos <- class_index
        class_index <- classes[class_pos]
      }
      
      message("Extracting SHAP values for class: ", class_index)
      result <- list(
        S = result$S[, , class_index, drop = FALSE],
        X = result$X,
        baseline = result$baseline[, class_index],
        class = class_index,
        classes = classes,
        feature_names = result$feature_names
      )
      class(result) <- "kernelshap"
    }
  }
  
  # Store additional metadata
  attr(result, "model_task") <- model$task
  attr(result, "feature_names") <- colnames(X)
  attr(result, "classes") <- if(model$task == "classification") classes else NULL
  
  message("SHAP computation complete!")
  return(result)
}


#' Enhanced SHAP analysis with automatic plotting
#' 
#' @param model Fitted unifiedml Model object
#' @param X Feature matrix (optional)
#' @param bg_X Background data (optional)
#' @param class_index For classification, which class to explain (default: NULL = all classes)
#' @param plot_type Type of plot: "importance", "summary", "dependence", or "none"
#' @param feature Feature name for dependence plot
#' @param ... Additional arguments passed to compute_shap_values or plotting functions
#' 
#' @return List containing SHAP results and plots
#' 
#' @export
analyze_shap <- function(model, X = NULL, bg_X = NULL, class_index = NULL,
                         plot_type = "importance", feature = NULL, ...) {
  
  # Check if shapviz is available for plotting
  if (plot_type != "none" && !requireNamespace("shapviz", quietly = TRUE)) {
    warning("shapviz package not installed for plotting. ",
            "Install with: install.packages('shapviz')")
    plot_type <- "none"
  }
  
  # Compute SHAP values
  shap_result <- compute_shap_values(model, X, bg_X, class_index = class_index, ...)
  
  # For classification without specific class, provide guidance
  if (model$task == "classification" && is.null(class_index) && 
      !is.null(attr(shap_result, "classes"))) {
    classes <- attr(shap_result, "classes")
    message("\nNote: SHAP values computed for all ", length(classes), " classes.")
    message("To explain a specific class, use class_index parameter, e.g.:")
    message("  analyze_shap(model, class_index = '", classes[2], "')")
    message("  or analyze_shap(model, class_index = 2)")
  }
  
  # Create plots if requested
  plots <- list()
  
  if (plot_type != "none") {
    # Get data for plotting
    X_plot <- if (is.null(X)) model$X_train else as.matrix(X)
    if (is.null(colnames(X_plot))) {
      colnames(X_plot) <- paste0("X", 1:ncol(X_plot))
    }
    
    # Create shapviz object
    sv <- shapviz::shapviz(shap_result, X = X_plot)
    
    # Generate requested plot
    switch(plot_type,
           importance = {
             plots$importance <- shapviz::sv_importance(sv, ...)
           },
           summary = {
             plots$summary <- shapviz::sv_importance(sv, kind = "both", ...)
           },
           dependence = {
             if (is.null(feature)) {
               feature <- colnames(X_plot)[1]
               message("No feature specified. Using: ", feature)
             }
             plots$dependence <- shapviz::sv_dependence(sv, v = feature, ...)
           }
    )
  }
  
  # Calculate feature importance
  if (model$task == "classification" && is.null(class_index)) {
    # For multi-class, return importance for each class
    feature_importance <- apply(abs(shap_result$S), 2:3, mean)
    colnames(feature_importance) <- attr(shap_result, "classes")
  } else {
    feature_importance <- colMeans(abs(shap_result$S))
  }
  
  # Return results
  result <- list(
    shap = shap_result,
    plots = plots,
    feature_importance = feature_importance
  )
  
  # Print feature importance summary
  cat("\n=== Feature Importance (Mean |SHAP|) ===\n")
  if (is.matrix(feature_importance)) {
    print(round(feature_importance, 4))
  } else {
    importance_df <- data.frame(
      Feature = names(feature_importance),
      Importance = round(feature_importance, 4)
    )
    importance_df <- importance_df[order(-importance_df$Importance), ]
    print(importance_df)
  }
  
  invisible(result)
}


#' Compare SHAP importance across multiple models
#' 
#' @param models List of fitted unifiedml Model objects
#' @param model_names Names for the models (optional)
#' @param X Feature matrix (optional, uses training data from first model)
#' 
#' @return Data frame with feature importance for all models
compare_shap_importance <- function(models, model_names = NULL, X = NULL) {
  if (is.null(model_names)) {
    model_names <- paste0("Model_", seq_along(models))
  }
  
  # Get feature names from first model
  if (is.null(X)) {
    X <- models[[1]]$X_train
  }
  features <- colnames(X)
  if (is.null(features)) {
    features <- paste0("X", 1:ncol(X))
  }
  
  # Compute importance for each model
  importance_list <- list()
  
  for (i in seq_along(models)) {
    message("\nProcessing ", model_names[i], "...")
    shap_result <- compute_shap_values(models[[i]], X = X)
    importance <- colMeans(abs(shap_result$S))
    importance_list[[model_names[i]]] <- importance
  }
  
  # Combine into data frame
  importance_df <- data.frame(
    Feature = features,
    do.call(cbind, importance_list)
  )
  
  return(importance_df)
}