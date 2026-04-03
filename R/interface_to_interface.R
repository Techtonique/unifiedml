#' Convert a formula-based model to a matrix interface
#'
#' Wraps a model function that expects \code{formula + data} so it can be called
#' with a plain \code{X} (data.frame or matrix) and \code{y} (vector).
#' Factors in \code{X} are preserved; special column names are safely backtick-quoted
#' in the generated formula so they survive the formula parser.
#'
#' @param fit_func   A model-fitting function whose first two arguments are
#'   \code{formula} and \code{data} (e.g. \code{lm}, \code{glm}).
#' @param predict_func A prediction function with signature
#'   \code{function(model, newdata, ...)}. Defaults to \code{stats::predict}.
#'
#' @return A named list with two elements:
#' \describe{
#'   \item{\code{fit(X, y, weights, ...)}}{Fits the model. \code{X} is a
#'     data.frame (or coercible matrix), \code{y} is the response vector.
#'     Extra arguments are forwarded to \code{fit_func}.}
#'   \item{\code{predict(model, newdata, ...)}}{Generates predictions.
#'     \code{newdata} must have the same columns as the \code{X} used in
#'     \code{fit}. Extra arguments are forwarded to \code{predict_func}.}
#' }
#'
#' @examples
#' lm_matrix <- formula_to_matrix(lm)
#' X <- data.frame(wt = mtcars$wt, hp = mtcars$hp, cyl = factor(mtcars$cyl))
#' y <- mtcars$mpg
#' model <- lm_matrix$fit(X, y)
#' lm_matrix$predict(model, X[1:5, ])
#'
#' @export
formula_to_matrix <- function(fit_func, predict_func = stats::predict) {
  
  ## ---- helpers ---------------------------------------------------------------
  
  # Safely quote a single column name for use inside a formula string.
  # Names that are valid R symbols are left as-is; everything else is
  # wrapped in backticks (with any embedded backticks escaped).
  safe_quote <- function(nm) {
    if (make.names(nm) == nm) nm
    else paste0("`", gsub("`", "\\\\`", nm), "`")
  }
  
  # Build a formula of the form:  .response ~ col1 + col2 + ...
  # using a dedicated response column injected into the data.
  build_formula <- function(col_names) {
    rhs <- if (length(col_names) == 0L) "1"
    else paste(vapply(col_names, safe_quote, character(1L)), collapse = " + ")
    stats::as.formula(paste(".response ~", rhs))
  }
  
  ## ---- fit -------------------------------------------------------------------
  
  fit <- function(X, y, weights = NULL, ...) {
    if (!is.data.frame(X)) X <- as.data.frame(X)
    
    # Inject response into a *copy* of X so the original is unmodified.
    df        <- X
    df[[".response"]] <- y
    
    fml <- build_formula(names(X))
    
    # lm() (and many formula-based models) capture the `weights` argument as
    # a language object via match.call(). Passing it through `...` or as a
    # named argument from inside a wrapper causes "invalid type (closure)"
    # errors.  The safe pattern is to inject weights as a column in `df` and
    # reference it by name.
    extra <- list(...)
    if (!is.null(weights)) {
      df[[".weights"]] <- weights
      do.call(fit_func, c(list(fml, data = df, weights = df[[".weights"]]), extra))
    } else {
      do.call(fit_func, c(list(fml, data = df), extra))
    }
  }
  
  ## ---- predict ---------------------------------------------------------------
  
  predict <- function(model, newdata, ...) {
    if (!is.data.frame(newdata)) newdata <- as.data.frame(newdata)
    predict_func(model, newdata = newdata, ...)
  }
  
  list(fit = fit, predict = predict)
}


#' Convert a matrix-based model to a formula interface
#'
#' Wraps a model function that expects a numeric matrix \code{X} and a response
#' vector \code{y} (like \code{glmnet::glmnet}) so it can be called with the
#' familiar \code{formula + data} interface.  The formula is expanded via
#' \code{\link[stats]{model.matrix}}, which handles factor dummy-coding,
#' interactions, and inline transformations automatically.
#'
#' @param fit_func     A model-fitting function whose first two positional
#'   arguments are \code{x} (numeric matrix) and \code{y} (response vector),
#'   e.g. \code{glmnet::glmnet}.
#' @param predict_func A prediction function with signature
#'   \code{function(model, newX, ...)} where \code{newX} is a numeric matrix.
#'   Defaults to a thin wrapper around \code{stats::predict} that passes
#'   \code{newdata} as \code{newx}.
#'
#' @return A named list with two elements:
#' \describe{
#'   \item{\code{fit(formula, data, ...)}}{Fits the model. The formula is
#'     expanded with \code{model.matrix}; the intercept column is dropped
#'     before passing to \code{fit_func} (add it back via \code{...} if your
#'     model needs it). Extra arguments are forwarded to \code{fit_func}.}
#'   \item{\code{predict(model, newdata, ...)}}{Generates predictions.
#'     \code{newdata} is expanded with the same \code{model.matrix} terms
#'     captured at fit time. Extra arguments are forwarded to
#'     \code{predict_func}.}
#' }
#'
#' @examples
#' \dontrun{
#' glmnet_formula <- matrix_to_formula(
#'   fit_func = glmnet::glmnet,
#'   predict_func = function(model, newX, ...) {
#'     glmnet::predict.glmnet(model, newx = newX, s = 0.01, ...)
#'   }
#' )
#' model <- glmnet_formula$fit(mpg ~ wt + hp + factor(cyl), data = mtcars)
#' glmnet_formula$predict(model, newdata = mtcars[1:5, ])
#' }
#'
#' @export
matrix_to_formula <- function(
    fit_func,
    predict_func = function(model, newX, ...) stats::predict(model, newdata = newX, ...)
) {
  
  # We store the formula terms object after fitting so that newdata is expanded
  # identically (same dummy levels, same column order) at predict time.
  terms_ref <- NULL   # will be set inside fit(), read inside predict()
  
  ## ---- fit -------------------------------------------------------------------
  
  fit <- function(formula, data, ...) {
    mf   <- stats::model.frame(formula, data = data)
    mt   <- attr(mf, "terms")
    y    <- stats::model.response(mf)
    
    # model.matrix drops the response but keeps the intercept column "(Intercept)".
    # Most matrix-interface models (glmnet, xgboost, …) want no intercept column.
    X    <- stats::model.matrix(mt, data = mf)
    X    <- X[, colnames(X) != "(Intercept)", drop = FALSE]
    
    # Persist terms so predict() can re-expand newdata identically.
    terms_ref <<- mt
    
    fit_func(X, y, ...)
  }
  
  ## ---- predict ---------------------------------------------------------------
  
  predict <- function(model, newdata, ...) {
    if (is.null(terms_ref)) {
      stop("Call `fit()` before `predict()`.")
    }
    
    newX <- stats::model.matrix(terms_ref, data = newdata)
    newX <- newX[, colnames(newX) != "(Intercept)", drop = FALSE]
    
    predict_func(model, newX, ...)
  }
  
  list(fit = fit, predict = predict)
}