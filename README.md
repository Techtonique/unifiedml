# unifiedml

> A Unified Machine Learning Interface for R

[![](https://cranlogs.r-pkg.org/badges/unifiedml)](https://cran.r-project.org/package=unifiedml)
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](http://docs.techtonique.net/unifiedml/index.html)

## Overview

`unifiedml` provides a consistent, sklearn-like interface for various (any) machine learning models in R. 

It eliminates the need to remember different function signatures across packages by **automatically detecting the appropriate interface** (formula vs matrix) and **task type** (regression vs classification).

### Key Features

For now: 

- **Automatic Task Detection**: Automatically detects regression vs classification based on response variable type (numeric → regression, factor → classification)
- **Universal Interface**: Works seamlessly with `glmnet`, `randomForest`, `e1071::svm`, and other popular ML packages with formula or matrix interface
- **Built-in Cross-Validation**: Consistent `cross_val_score()` function with automatic metric selection
- **Model Interpretability**: Numerical derivatives and statistical significance testing via `summary()`
- **Partial Dependence Plots**: Visualize feature effects with `plot()`
- **Method Chaining**: Clean, pipeable syntax with R6 classes

## Installation

- From CRAN: 

  ```R
  install.packages("unifiedml")
  ```
  
- From Github (development version):
  
  ```R
  # Install from GitHub (development version, for now)
  devtools::install_github("Techtonique/unifiedml")
  ```

## Quick Start

### Regression Example

```R
library(unifiedml)
library(glmnet)

# Prepare data
data(mtcars)
X <- as.matrix(mtcars[, -1])
y <- mtcars$mpg  # numeric → automatic regression

# Fit model
mod <- Model$new(glmnet::glmnet)
mod$fit(X, y, alpha = 0, lambda = 0.1)

# Make predictions
predictions <- mod$predict(X)

# Get model summary with feature importance
mod$summary()

# Visualize partial dependence
mod$plot(feature = 1)

# Cross-validation (automatically uses RMSE for regression)
cv_scores <- cross_val_score(mod, X, y, cv = 5)
cat("Mean RMSE:", mean(cv_scores), "\n")
```

### Classification Example

```R
library(randomForest)

# Prepare data
data(iris)
X <- as.matrix(iris[, 1:4])
y <- iris$Species  # factor → automatic classification

# Fit model
mod <- Model$new(randomForest::randomForest)
mod$fit(X, y, ntree = 100)

# Make predictions
predictions <- mod$predict(X)

# Get model summary
mod$summary()

# Cross-validation (automatically uses accuracy for classification)
cv_scores <- cross_val_score(mod, X, y, cv = 5)
cat("Mean Accuracy:", mean(cv_scores), "\n")
```

## Core Functionality

### The Model Class

The `Model` R6 class provides a unified interface for any machine learning function:

```R
# Create a model wrapper
mod <- Model$new(model_function)

# Fit the model (task type auto-detected from y)
mod$fit(X, y, ...)

# Make predictions
predictions <- mod$predict(X_new)

# Get interpretable summary
mod$summary(h = 0.01, alpha = 0.05)

# Visualize feature effects
mod$plot(feature = 1)

# Print model info
mod$print()
```

### Cross-Validation

The `cross_val_score()` function provides consistent k-fold cross-validation:

```R
# Automatic metric selection based on task
scores <- cross_val_score(mod, X, y, cv = 5)

# Specify custom metric
scores <- cross_val_score(mod, X, y, cv = 10, scoring = "mae")

# Available metrics:
# - Regression: "rmse" (default), "mae"
# - Classification: "accuracy" (default), "f1"
```

### Model Interpretability

The `summary()` method uses numerical derivatives to assess feature importance:

```R
mod$summary()
# Output:
# Model Summary - Numerical Derivatives
# ======================================
# Task: regression
# Samples: 150 | Features: 4
# 
# Feature         Mean_Derivative  Std_Error  t_value  p_value  Significance
# Sepal.Length    0.523           0.042       12.45    < 0.001  ***
# Sepal.Width    -0.234           0.038       -6.16    < 0.001  ***
# ...
```

## Supported Models

`unifiedml` automatically detects the appropriate interface for:

- **glmnet**: Ridge, Lasso, Elastic Net regression and classification
- **randomForest**: Random forest for regression and classification
- **e1071::svm**: Support Vector Machines
- **Any model** with either formula (`y ~ .`) or matrix (`x, y`) interface

## Automatic Task Detection

The package automatically determines the task type:

```R
# Regression (numeric y)
y_reg <- c(1.2, 3.4, 5.6, ...)
mod$fit(X, y_reg)  # → task = "regression"

# Classification (factor y)
y_class <- factor(c("A", "B", "A", ...))
mod$fit(X, y_class)  # → task = "classification"
```

## Advanced Features

### Partial Dependence Plots

```R
# Visualize how feature j affects predictions
mod$plot(feature = 3, n_points = 100)
```

### Model Cloning

```R
# Create independent copy for parallel processing
mod_copy <- mod$clone_model()
```

## Examples

See the package vignette for comprehensive examples:

```R
vignette("introduction", package = "unifiedml")
```

## Why unifiedml?

Traditional R modeling requires remembering different interfaces:

```R
# Different interfaces = cognitive overhead
glmnet(x = X, y = y, ...)               # matrix interface
randomForest(y ~ ., data = df, ...)     # formula interface
svm(x = X, y = y, ...)                  # matrix interface
```

With `unifiedml`, it's always the same:

```R
# One interface to rule them all
Model$new(glmnet)$fit(X, y, ...)
Model$new(randomForest)$fit(X, y, ...)
Model$new(svm)$fit(X, y, ...)
```

## Contributing

Contributions are welcome, feel free to submit a Pull Request.

## License

The Clear BSD License - see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
@Manual{unifiedml,
  title = {unifiedml: Unified Machine Learning Interface for R},
  author = {T. Moudiki},
  year = {2025},
  note = {R package version 0.1.0}
}
```
