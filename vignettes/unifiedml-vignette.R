## ----fig.width=7--------------------------------------------------------------
library(unifiedml) # this package
library(glmnet)
library(randomForest)
library(e1071)

# ------------------------------------------------------------
# REGRESSION EXAMPLES
# ------------------------------------------------------------

cat("\n=== REGRESSION EXAMPLES ===\n\n")

# Example 1: Synthetic data (numeric y → automatic regression)
set.seed(123)
X <- MASS::Boston[, -ncol(MASS::Boston)]
y <- MASS::Boston$medv

# glmnet regression
cat("1. Ridge Regression (glmnet) - Auto-detected: Regression\n")
mod1 <- Model$new(glmnet::glmnet)  # No task parameter needed!
mod1$fit(X, y, alpha = 0, lambda = 0.1)
mod1$print()
cat("\nPredictions:\n")
print(head(mod1$predict(X)))
cat("\n")
mod1$summary()
mod1$plot(feature = 1)
(cv1 <- cross_val_score(mod1, X, y, cv = 5L))  # Auto-uses RMSE
cat("\nMean RMSE:", mean(cv1), "\n\n")

# randomForest regression
cat("\n2. Random Forest Regression - Auto-detected: Regression\n")
mod2 <- Model$new(randomForest::randomForest)  # No task parameter!
mod2$fit(X, y, ntree = 50)
mod2$print()
cat("\n")
mod2$summary(h = 0.01)

# ------------------------------------------------------------
# CLASSIFICATION EXAMPLES
# ------------------------------------------------------------

cat("\n\n=== CLASSIFICATION EXAMPLES ===\n\n")

# Example: Iris dataset (factor y → automatic classification)
data(iris)

# Binary classification with factor
cat("3. Binary Classification with Factor Response\n")
iris_binary <- iris[iris$Species %in% c("setosa", "versicolor"), ]
X_binary <- as.matrix(iris_binary[, 1:4])
y_binary <- iris_binary$Species  # factor → classification

# Multi-class classification
cat("4. Multi-class Classification\n")
X_multi <- as.matrix(iris[, 1:4])
y_multi <- iris$Species  # factor with 3 levels → multi-class classification

mod4 <- Model$new(randomForest::randomForest)  # No task parameter!
mod4$fit(X_multi, y_multi, ntree = 50)
mod4$print()
(cv4 <- cross_val_score(mod4, X_multi, y_multi, cv = 5L))  # Auto-uses accuracy
cat("\nMean Accuracy:", mean(cv4), "\n")


y_multi_numeric <- as.numeric(y_multi)
mod4 <- Model$new(glmnet::glmnet)  # No task parameter!
mod4$fit(X_multi, y_multi_numeric, family="multinomial")
mod4$print()
(cv4 <- cross_val_score(mod4, X_multi, y_multi_numeric, cv = 5L))  # Auto-uses accuracy
cat("\nMean Accuracy:", mean(cv4), "\n")

