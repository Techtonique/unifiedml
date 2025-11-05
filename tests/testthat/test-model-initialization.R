test_that("Model initialization works", {
  # Valid initialization
  mod <- Model$new(glmnet::glmnet)
  expect_s3_class(mod, "Model")
  expect_s3_class(mod, "R6")
  expect_null(mod$fitted)
  expect_null(mod$task)
  
  # Invalid initialization
  expect_error(Model$new("not_a_function"))
  expect_error(Model$new(123))
})
