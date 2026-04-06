#' Split data into training and test sets
#'
#' Randomly splits a feature matrix or data.frame and its corresponding
#' response vector into training and test subsets.
#'
#' @param X A matrix or data.frame of features.
#' @param y A vector of responses (numeric or factor). Must have the same
#'   number of rows as \code{X}.
#' @param test_size Proportion of observations to use as the test set.
#'   A number in (0, 1). Default is \code{0.2} (80/20 split).
#' @param seed An optional integer random seed for reproducibility. If
#'   \code{NULL} (default) the current RNG state is used.
#'
#' @return A named list with four elements:
#'   \item{X_train}{Training features (same type as \code{X}).}
#'   \item{X_test}{Test features (same type as \code{X}).}
#'   \item{y_train}{Training response.}
#'   \item{y_test}{Test response.}
#'
#' @examples
#' # matrix input
#' X <- iris[, 1:4]
#' y <- iris$Species
#' d <- unifiedml::train_test_split(X, y, test_size = 0.3, seed = 42)
#' dim(d$X_train)  # 105 x 4
#' dim(d$X_test)   #  45 x 4
#'
#' # data.frame input
#' d2 <- unifiedml::train_test_split(iris[, 1:4], iris$Species, test_size = 0.2)
#' is.data.frame(d2$X_train)  # TRUE
#'
#' @export
train_test_split <- function(X, y, test_size = 0.2, seed = NULL) {
  stopifnot(
    is.matrix(X) || is.data.frame(X),
    is.vector(y) || is.factor(y),
    nrow(X) == length(y),
    test_size > 0 && test_size < 1
  )
  
  if (!is.null(seed)) set.seed(seed)
  
  n        <- nrow(X)
  n_test   <- max(1L, floor(n * test_size))
  test_idx <- sample(n, size = n_test)
  
  list(
    X_train = X[-test_idx, , drop = FALSE],
    X_test  = X[ test_idx, , drop = FALSE],
    y_train = y[-test_idx],
    y_test  = y[ test_idx]
  )
}