#' @importFrom stats predict family model.matrix
NULL

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.get_class_labels <- function(y) {
  if (is.factor(y))          return(levels(y))
  if (is.character(y))       return(sort(unique(y)))
  if (is.numeric(y))         return(c("0", "1"))
  NULL
}

.to_matrix <- function(X) {
  if (!is.matrix(X)) as.matrix(X) else X
}

.to_df <- function(X) {
  if (!is.data.frame(X)) as.data.frame(X) else X
}

# Dispatcher table: each entry is function(model, X) -> raw prediction or NULL
.extractors <- list(
  
  randomForest = function(model, X) {
    if (model$type != "classification") return(NULL)
    tryCatch(predict(model, X, type = "prob"), error = function(e) NULL)
  },
  
  glm = function(model, X) {
    fam <- tryCatch(family(model)$family, error = function(e) NULL)
    if (is.null(fam)) return(NULL)
    X_df <- .to_df(X)
    if (fam == "binomial") {
      tryCatch(predict(model, newdata = X_df, type = "response"), error = function(e) NULL)
    } else {
      NULL  # multinomial is not a base-R glm family; handled via fallback
    }
  },
  
  multinom = function(model, X) {
    tryCatch(predict(model, newdata = .to_df(X), type = "probs"), error = function(e) NULL)
  },
  
  svm = function(model, X) {
    if (model$type != 0L) return(NULL)  # 0 = C-classification
    tryCatch({
      pred_obj <- predict(model, X, probability = TRUE)
      attr(pred_obj, "probabilities")
    }, error = function(e) NULL)
  },
  
  nnet = function(model, X) {
    tryCatch(predict(model, newdata = .to_df(X), type = "raw"), error = function(e) NULL)
  },
  
  rpart = function(model, X) {
    tryCatch(predict(model, newdata = .to_df(X), type = "prob"), error = function(e) NULL)
  },
  
  xgb.Booster = function(model, X) {
    tryCatch({
      pred <- predict(.to_matrix(X))
      if (is.null(dim(pred))) cbind(1 - pred, pred) else pred
    }, error = function(e) NULL)
  },
  
  glmnet = function(model, X) {
    tryCatch({
      pred <- predict(model, .to_matrix(X), type = "response")
      if (is.matrix(pred) && ncol(pred) == 1) cbind(1 - pred, pred) else pred
    }, error = function(e) NULL)
  },
  
  ksvm = function(model, X) {
    tryCatch(
      predict(model, .to_matrix(X), type = "probabilities"),
      error = function(e) NULL
    )
  }
)

# Generic fallback cascade - tried in order when no specific extractor matches
.fallback_extractors <- list(
  
  function(model, X)
    tryCatch(predict(model, newdata = .to_df(X),  type = "response"),   error = function(e) NULL),
  
  function(model, X)
    tryCatch(predict(model, newx = .to_matrix(X), type = "response"),   error = function(e) NULL),
  
  function(model, X)
    tryCatch(predict(model, X,                    type = "response"),   error = function(e) NULL),
  
  function(model, X)
    tryCatch(predict(model, newdata = .to_df(X),  type = "prob"),       error = function(e) NULL),
  
  function(model, X)
    tryCatch(predict(model, newdata = .to_df(X),  type = "raw"),        error = function(e) NULL),
  
  function(model, X)
    tryCatch({
      pred_obj <- predict(model, X, probability = TRUE)
      probs    <- attr(pred_obj, "probabilities")
      if (is.matrix(probs)) probs else NULL
    }, error = function(e) NULL),
  
  function(model, X)
    tryCatch(predict(model, X), error = function(e) NULL)
)

# ---------------------------------------------------------------------------
# Standardise raw prediction -> n x k probability matrix
# ---------------------------------------------------------------------------

.standardize_to_probs <- function(pred, y_train, verbose) {
  
  # Case 1: already a valid probability matrix
  if (is.matrix(pred) && all(pred >= 0) && all(pred <= 1)) {
    row_sums <- rowSums(pred)
    if (all(abs(row_sums - 1) < 1e-6)) {
      if (verbose) message("  Already a probability matrix")
      if (is.null(colnames(pred)) && !is.null(y_train))
        colnames(pred) <- .get_class_labels(y_train)
      return(pred)
    }
  }
  
  # Case 2: numeric vector
  if (is.numeric(pred) && !is.matrix(pred)) {
    if (all(pred >= 0 & pred <= 1)) {
      if (verbose) message("  Binary probability vector")
      n_classes <- if (!is.null(y_train) && is.factor(y_train)) nlevels(y_train) else 2
      if (n_classes != 2)
        stop("Probability vector implies binary classification but y_train has ", n_classes, " classes")
      prob_matrix <- cbind(1 - pred, pred)
      colnames(prob_matrix) <- if (!is.null(y_train))
        .get_class_labels(y_train) else c("Class0", "Class1")
      return(prob_matrix)
    } else {
      warning("Numeric predictions outside [0,1] interpreted as logits; verify this is correct")
      if (verbose) message("  Logit vector -> sigmoid")
      probs <- 1 / (1 + exp(-pred))
      return(.standardize_to_probs(probs, y_train, verbose))
    }
  }
  
  # Case 3: factor (hard classes) -> one-hot
  if (is.factor(pred)) {
    if (verbose) message("  Hard classes -> one-hot")
    classes      <- levels(pred)
    prob_matrix  <- model.matrix(~ pred - 1)
    colnames(prob_matrix) <- classes
    return(prob_matrix)
  }
  
  # Case 4: matrix needing normalisation
  if (is.matrix(pred)) {
    row_sums <- rowSums(pred)
    if (any(abs(row_sums - 1) > 1e-6)) {
      if (all(pred >= 0)) {
        if (verbose) message("  Raw score matrix -> row normalise")
        prob_matrix <- pred / row_sums
      } else {
        if (verbose) message("  Logit matrix -> softmax")
        exp_pred    <- exp(pred - apply(pred, 1, max))  # numerically stable
        prob_matrix <- exp_pred / rowSums(exp_pred)
      }
      if (!is.null(y_train))
        colnames(prob_matrix) <- .get_class_labels(y_train)
      return(prob_matrix)
    }
  }
  
  # Case 5: list with $probabilities
  if (is.list(pred) && !is.null(pred$probabilities)) {
    if (verbose) message("  List$probabilities")
    return(.standardize_to_probs(pred$probabilities, y_train, verbose))
  }
  
  # Last resort
  if (verbose) message("  Unknown format, coercing to matrix")
  .standardize_to_probs(as.matrix(pred), y_train, verbose)
}

# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

#' Extract probability predictions from any R model in a standardised format
#'
#' @param model Fitted model object (any class)
#' @param X Feature matrix or data.frame for predictions
#' @param y_train Optional training labels used to name output columns
#' @param verbose Print diagnostic information (default: FALSE)
#'
#' @return Numeric matrix of shape n_samples x n_classes with column names
#'   as class labels. Attributes \code{extraction_method} and
#'   \code{model_class} record how predictions were obtained.
#'
#' @export
extract_probabilities <- function(model, X, y_train = NULL, verbose = FALSE) {
  
  # Input validation
  if (is.null(model))           stop("'model' must not be NULL")
  if (NROW(X) == 0)             stop("'X' has no rows")
  if (!is.null(y_train) && !is.factor(y_train) &&
      !is.character(y_train) && !is.numeric(y_train))
    stop("'y_train' must be a factor, character, or numeric vector")
  
  model_class <- class(model)[1]
  if (verbose) message("Model class: ", paste(class(model), collapse = ", "))
  
  # 1. Try registered extractor
  extractor <- .extractors[[model_class]]
  prediction <- if (!is.null(extractor)) {
    if (verbose) message("  Trying registered extractor for '", model_class, "'")
    extractor(model, X)
  } else NULL
  
  method <- if (!is.null(prediction))
    paste0("registered::", model_class) else "unknown"
  
  # 2. Fall back to generic cascade
  if (is.null(prediction)) {
    if (verbose) message("  No registered extractor, trying fallback cascade...")
    for (i in seq_along(.fallback_extractors)) {
      prediction <- .fallback_extractors[[i]](model, X)
      if (!is.null(prediction)) {
        method <- paste0("fallback::", i)
        if (verbose) message("  Fallback ", i, " succeeded")
        break
      }
    }
  }
  
  if (is.null(prediction))
    stop("Could not extract predictions from model of class: ",
         paste(class(model), collapse = ", "))
  
  if (verbose) message("Extraction method: ", method)
  
  # Standardise to probability matrix
  result <- .standardize_to_probs(prediction, y_train, verbose)
  
  # Final checks
  if (!is.matrix(result))
    result <- as.matrix(result)
  
  result[result < 0] <- 0
  result[result > 1] <- 1
  
  row_sums <- rowSums(result)
  if (any(abs(row_sums - 1) > 0.01))
    warning("Row sums deviate from 1 by more than 0.01 before normalisation; check model output")
  if (any(abs(row_sums - 1) > 1e-4))
    result <- result / row_sums
  
  attr(result, "extraction_method") <- method
  attr(result, "model_class")       <- model_class
  
  if (verbose) {
    message("Output: ", paste(dim(result), collapse = "x"), " matrix")
    message("Classes: ", paste(colnames(result), collapse = ", "))
  }
  
  result
}