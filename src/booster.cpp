#include <Rcpp.h>
using namespace Rcpp;

//' Boosting for Regression using Any R Model
//' 
//' Implements Algorithm 8.2 for gradient boosting with regression trees
//' or any other base learner compatible with the Model R6 class interface.
//' This implementation uses the negative gradient of squared error loss.
//' 
//' @param model_creator An R function that returns a new Model object with
//'   fit() and predict() methods
//' @param X Feature matrix (n x p)
//' @param y Response vector (length n)
//' @param B Number of boosting iterations
//' @param model_args Named list of additional arguments to pass to model$fit()
//' @param eta Shrinkage parameter (learning rate), default 0.1
//' @param verbose Whether to print progress (default: TRUE)
//' 
//' @return A list containing:
//' \itemize{
//'   \item models: List of B fitted model objects
//'   \item f_hat: Final predictions on training data
//'   \item y: Original response vector (needed for variable importance)
//'   \item residuals_history: Matrix of residuals at each iteration (n x B+1)
//'     Column 0 contains initial residuals (y), columns 1:B contain updated residuals
//'   \item predictions_history: Matrix of base learner predictions (n x B)
//'   \item eta: The shrinkage parameter used
//'   \item B: Number of iterations
//' }
//' 
//' @details
//' For squared error loss L(y,f) = 0.5*(y-f)^2, the negative gradient is
//' -dL/df = y - f, which equals the residuals. Each iteration fits a base
//' learner to these residuals (pseudo-residuals) and updates predictions.
//' 
//' @examples
//' \dontrun{
//' 
//' # Model creator function
//' model_creator <- function() {
//'   Model$new(lm)
//' }
//' 
//' # Generate data
//' set.seed(123)
//' X <- matrix(rnorm(200), ncol = 4)
//' y <- 2*X[,1] - 1.5*X[,2] + rnorm(50)
//' 
//' # Run boosting
//' result <- boost_regression(
//'   model_creator = model_creator,
//'   X = X, y = y, 
//'   B = 100, eta = 0.1,
//'   model_args = list(maxdepth = 2)
//' )
//' 
//' # Make predictions
//' preds <- predict_boost(result, X)
//' sqrt(mean((y - preds)^2))  # RMSE
//' }
//' 
// [[Rcpp::export]]
List boost_regression(Function model_creator,
                     NumericMatrix X,
                     NumericVector y,
                     int B,
                     List model_args,
                     double eta = 0.1,
                     bool verbose = true) {
  
  RNGScope scope;
 
 if (model_args.size() == 0)
   model_args = List::create();
 
 // Input validation
 int n = X.nrow();
 int p = X.ncol();
 
 if (n != y.size()) {
   stop("X and y must have compatible dimensions");
 }
 if (B <= 0) {
   stop("B must be positive");
 }
 if (eta <= 0 || eta > 1) {
   stop("eta must be in (0, 1]");
 }
 if (n == 0 || p == 0) {
   stop("X cannot be empty");
 }
 
 // Step 1: Initialize f(x) = 0 and r_i = y_i
 NumericVector f_hat(n, 0.0);  // Current predictions
 NumericVector residuals = clone(y);  // Current residuals (negative gradient)
 
 // Store all models, residual history, and prediction history
 List models(B);
 NumericMatrix residuals_history(n, B + 1);
 NumericMatrix predictions_history(n, B);
 
 // Store initial residuals (column 0)
 residuals_history(_, 0) = clone(residuals);
 
 // Prepare model_args with proper names
 List fit_args = clone(model_args);
 
 // Step 2: For b = 1, 2, ..., B
 for (int b = 0; b < B; b++) {
   
   if (verbose && (b + 1) % 10 == 0) {
     Rcout << "Iteration " << (b + 1) << " / " << B << std::endl;
   }
   
   // (a) Fit a tree to training data (X, r)
   // Create a new model instance
   Environment model = model_creator();
   
   // Verify model has required methods
   if (!model.exists("fit") || !model.exists("predict")) {
     stop("Model object must have 'fit' and 'predict' methods");
   }
   
   // Get fit method and call it
   Function fit = model["fit"];
   
   // Set X and y in fit_args (overwrites if already present)
   fit_args["X"] = X;
   fit_args["y"] = residuals;
   
   // Fit the model using do.call
   Function do_call("do.call");
   try {
     do_call(fit, fit_args);
   } catch(std::exception &ex) {
     stop("Model fit failed at iteration " + std::to_string(b + 1) + 
       ": " + ex.what());
   }
   
   // (b) Update f_hat by adding shrunk version of new tree
   // Get predictions: f_b(x_i)
   Function predict = model["predict"];
   NumericVector f_b;
   
   try {
     f_b = predict(X);
   } catch(std::exception &ex) {
     stop("Model predict failed at iteration " + std::to_string(b + 1) + 
       ": " + ex.what());
   }
   
   // Store base learner predictions (for later use)
   predictions_history(_, b) = f_b;
   
   // f_hat(x) <- f_hat(x) + eta * f_b(x)
   f_hat = f_hat + eta * f_b;
   
   // (c) Update the residuals: r_i <- r_i - eta * f_b(x_i)
   // This is the negative gradient for squared error loss
   residuals = residuals - eta * f_b;
   
   // Store updated residuals (column b+1)
   residuals_history(_, b + 1) = clone(residuals);
   
   // Store the fitted model
   models[b] = model;
 }
 
 // Step 3: Output the boosted model
 return List::create(
   Named("models") = models,
   Named("f_hat") = f_hat,
   Named("y") = y,  // Store original y for variable importance
   Named("residuals_history") = residuals_history,
   Named("predictions_history") = predictions_history,  // Store for efficiency
   Named("eta") = eta,
   Named("B") = B
 );
}

//' Predict using a Boosted Model
//' 
//' Generate predictions from a boosted model on new data.
//' 
//' @param boost_obj A boosted model object from boost_regression()
//' @param X_new Feature matrix for prediction (m x p)
//' 
//' @return Vector of predictions (length m)
//' 
// [[Rcpp::export]]
NumericVector predict_boost(List boost_obj, NumericMatrix X_new) {
  
  RNGScope scope;
 
 List models = boost_obj["models"];
 double eta = boost_obj["eta"];
 int B = models.size();
 int m = X_new.nrow();
 
 // Initialize predictions to 0
 NumericVector predictions(m, 0.0);
 
 // Sum over all B models: f_hat(x) = sum_{b=1}^B eta * f_b(x)
 for (int b = 0; b < B; b++) {
   Environment model = models[b];
   
   if (!model.exists("predict")) {
     stop("Model at iteration " + std::to_string(b + 1) + 
       " does not have predict method");
   }
   
   Function predict = model["predict"];
   NumericVector f_b;
   
   try {
     f_b = predict(X_new);
   } catch(std::exception &ex) {
     stop("Prediction failed at iteration " + std::to_string(b + 1) + 
       ": " + ex.what());
   }
   
   predictions = predictions + eta * f_b;
 }
 
 return predictions;
}

//' Compute Variable Importance for Boosted Model
//' 
//' Calculate relative importance of each feature based on correlation
//' with base learner predictions, weighted by SSE improvement.
//' 
//' @param boost_obj A boosted model object from boost_regression()
//' @param normalize Whether to normalize importances to sum to 100
//' 
//' @return Vector of variable importances (length p)
//' 
//' @details
//' This is a generic, correlation-based importance metric suitable for
//' any base learner. For tree-based models, consider using model-specific
//' importance measures if available. The importance for feature j is
//' computed as sum over iterations of:
//' SSE_improvement * |correlation(X_j, f_b)|
//' 
// [[Rcpp::export]]
NumericVector variable_importance_boost(List boost_obj,
                                       bool normalize = true) {
  
  RNGScope scope;
 
 // Extract components
 NumericMatrix resid_hist = boost_obj["residuals_history"];
 NumericMatrix pred_hist = boost_obj["predictions_history"];
 NumericVector y = boost_obj["y"];
 double eta = boost_obj["eta"];
 int B = boost_obj["B"];
 
 // Get training X from first model prediction attempt
 // NOTE: X is not stored, so we need to accept it as parameter
 // This is a design limitation - fixing below
 stop("variable_importance_boost requires X parameter - use variable_importance_boost_with_X instead");
 
 return NumericVector(0);
}

//' Compute Variable Importance for Boosted Model (with X)
//' 
//' Calculate relative importance of each feature based on correlation
//' with base learner predictions, weighted by SSE improvement.
//' 
//' @param boost_obj A boosted model object from boost_regression()
//' @param X Training feature matrix (n x p)
//' @param normalize Whether to normalize importances to sum to 100
//' 
//' @return Vector of variable importances (length p)
//' 
//' @details
//' This computes a generic importance metric by:
//' 1. Computing SSE improvement from each iteration
//' 2. Computing correlation between each feature and base learner predictions
//' 3. Allocating improvement proportional to squared correlation
//' 
//' Note: This is a heuristic proxy. For tree-based models, model-specific
//' importance measures (e.g., split improvement) may be more accurate.
//' 
// [[Rcpp::export]]
NumericVector variable_importance_boost_with_X(List boost_obj, 
                                              NumericMatrix X,
                                              bool normalize = true) {
  
  RNGScope scope;
 
 // Extract components
 NumericMatrix resid_hist = boost_obj["residuals_history"];
 NumericMatrix pred_hist = boost_obj["predictions_history"];
 NumericVector y = boost_obj["y"];
 double eta = boost_obj["eta"];
 int B = boost_obj["B"];
 
 int n = X.nrow();
 int p = X.ncol();
 
 if (n != resid_hist.nrow()) {
   stop("X must have same number of rows as training data");
 }
 
 NumericVector importance(p, 0.0);
 
 // For each boosting iteration
 for (int b = 0; b < B; b++) {
   
   // Get residuals before and after this iteration
   NumericVector r_before = resid_hist(_, b);      // Initial or previous residuals
   NumericVector r_after = resid_hist(_, b + 1);   // After update
   
   // Compute improvement in SSE
   double sse_before = 0.0, sse_after = 0.0;
   for (int i = 0; i < n; i++) {
     sse_before += r_before[i] * r_before[i];
     sse_after += r_after[i] * r_after[i];
   }
   double improvement = sse_before - sse_after;
   
   // Get base learner predictions for this iteration
   NumericVector f_b = pred_hist(_, b);
   
   // Compute mean of f_b
   double mean_f = 0.0;
   for (int i = 0; i < n; i++) {
     mean_f += f_b[i];
   }
   mean_f /= n;
   
   // Compute variance of f_b
   double var_f = 0.0;
   for (int i = 0; i < n; i++) {
     double df = f_b[i] - mean_f;
     var_f += df * df;
   }
   
   // Skip if f_b is constant
   if (var_f < 1e-10) continue;
   
   // Distribute improvement based on feature correlation with predictions
   double total_cor = 0.0;
   NumericVector cor_squared(p);
   
   for (int j = 0; j < p; j++) {
     // Compute mean of X[,j]
     double mean_x = 0.0;
     for (int i = 0; i < n; i++) {
       mean_x += X(i, j);
     }
     mean_x /= n;
     
     // Compute covariance and variance
     double cov = 0.0, var_x = 0.0;
     for (int i = 0; i < n; i++) {
       double dx = X(i, j) - mean_x;
       double df = f_b[i] - mean_f;
       cov += dx * df;
       var_x += dx * dx;
     }
     
     // Compute squared correlation
     if (var_x > 1e-10) {
       double cor = cov / std::sqrt(var_x * var_f);
       cor_squared[j] = cor * cor;  // Use squared correlation
       total_cor += cor_squared[j];
     } else {
       cor_squared[j] = 0.0;
     }
   }
   
   // Distribute improvement proportional to squared correlation
   if (total_cor > 0) {
     for (int j = 0; j < p; j++) {
       importance[j] += improvement * (cor_squared[j] / total_cor);
     }
   }
 }
 
 // Normalize if requested
 if (normalize) {
   double total = 0.0;
   for (int j = 0; j < p; j++) {
     total += importance[j];
   }
   if (total > 0) {
     for (int j = 0; j < p; j++) {
       importance[j] = (importance[j] / total) * 100.0;
     }
   }
 }
 
 return importance;
}

//' Compute Training Loss History
//' 
//' Calculate MSE on training data at each boosting iteration.
//' 
//' @param boost_obj A boosted model object from boost_regression()
//' 
//' @return Vector of MSE values (length B+1), starting with initial MSE
//' 
// [[Rcpp::export]]
NumericVector compute_loss_history(List boost_obj) {
  
  RNGScope scope;
 
 NumericMatrix resid_hist = boost_obj["residuals_history"];
 int n = resid_hist.nrow();
 int B = boost_obj["B"];
 
 NumericVector loss_history(B + 1);
 
 for (int b = 0; b <= B; b++) {
   NumericVector residuals = resid_hist(_, b);
   double mse = 0.0;
   for (int i = 0; i < n; i++) {
     mse += residuals[i] * residuals[i];
   }
   loss_history[b] = mse / n;
 }
 
 return loss_history;
}