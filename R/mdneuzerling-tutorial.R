# learning mlflow following this:

# https://mdneuzerling.com/post/tracking-tidymodels-with-mlflow/
# initial steps are her: https://mdneuzerling.com/post/machine-learning-pipelines-with-tidymodels-and-targets/

library(mlflow)
library(tidyverse)
library(tidymodels)
library(tidytuesdayR)
library(visdat)

# data in

coffee <- invisible(tidytuesdayR::tt_load(2020, week = 28)$coffee)

# visualise

visdat::vis_dat(coffee)

coffee %>% select(is_numeric) %>% visdat::vis_cor()

# split data
coffee_split <- initial_split(coffee, prop = 0.8)
coffee_train <- training(coffee_split)
coffee_test <- testing(coffee_split)


# define recipe
coffee_recipe <- recipe(coffee_train) %>%
  update_role(everything(), new_role = "support") %>%
  update_role(cupper_points, new_role = "outcome") %>%
  update_role(
    variety, processing_method, country_of_origin,
    aroma, flavor, aftertaste, acidity, sweetness, altitude_mean_meters,
    new_role = "predictor"
  ) %>%
  step_string2factor(all_nominal(), -all_outcomes()) %>%
  step_impute_knn(country_of_origin,
                 impute_with = imp_vars(
                   in_country_partner, company, region, farm_name, certification_body
                 )
  ) %>%
  step_impute_knn(altitude_mean_meters,
                 impute_with = imp_vars(
                   in_country_partner, company, region, farm_name, certification_body,
                   country_of_origin
                 )
  ) %>%
  step_unknown(variety, processing_method, new_level = "Unknown") %>%
  step_other(country_of_origin, threshold = 0.01) %>%
  step_other(processing_method, threshold = 0.10) %>%
  step_other(variety, threshold = 0.10) %>%
  step_normalize(all_numeric(), -all_outcomes())

# define model and engine
coffee_model <- rand_forest(trees = tune(), mtry = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# define workflow
coffee_workflow <- workflows::workflow() %>%
  add_recipe(coffee_recipe) %>%
  add_model(coffee_model)


# define parameters
coffee_grid <- expand_grid(mtry = 3:5, trees = seq(500, 1500, by = 200))

# run model
coffee_grid_results <- coffee_workflow %>%
  tune_grid(resamples <- vfold_cv(coffee_train, v = 5), grid = coffee_grid)

# extract parameters
hyperparameters <- coffee_grid_results %>%
  select_by_pct_loss(metric = "rmse", limit = 5, trees)


# have a look at workflow
coffee_workflow


### mlflow
#
# function for logging model hyperparameters as MLflow parameters. This function
# will only log hyperparameters set by the user, since the default values have a
# NULL expression, but I think that this approach makes sense. It also passes on
# the input workflow unmodified

log_workflow_parameters <- function(workflow) {
  # Would help to have a check here: has this workflow been finalised?
  # It may be sufficient to check that the arg quosures carry no environments.
  spec <- extract_spec_parsnip(workflow)
  parameter_names <- names(spec$args)
  parameter_values <- lapply(spec$args, rlang::get_expr)
  for (i in seq_along(spec$args)) {
    parameter_name <- parameter_names[[i]]
    parameter_value <- parameter_values[[i]]
    if (!is.null(parameter_value)) {
      mlflow_log_param(parameter_name, parameter_value)
    }
  }
  workflow
}


# log metrics. The input to this function will be a metrics tibble produced
# by the yardstick package, which is a component of tidymodels:

log_metrics <- function(metrics, estimator = "standard") {
  metrics %>% filter(.estimator == estimator) %>% pmap(
    function(.metric, .estimator, .estimate) {
      mlflow_log_metric(.metric, .estimate)
    }
  )
  metrics
}


# store artifacts with each run. These are usually models, but could be
# anything. MLflow supports exporting models with the carrier::crate function.
# This is a tricky function to use, since the user must comprehensively list
# their dependencies. For a workflow with a recipe, it’s a lot easier. All of
# the preprocessing is contained within the recipe, and the fitted workflow
# object contains this.


# I haven't yet defined fitted_coffee_model, so I won't run this
crated_model <- carrier::crate(
  function(x) workflows:::predict.workflow(fitted_coffee_model, x),
  fitted_coffee_model = fitted_coffee_model
)


## set experiment ====
# set my experiment as coffee. I only need to do this once per session:

  mlflow_set_experiment(experiment_name = "coffee")




with(mlflow_start_run(nested = TRUE), {
  fitted_coffee_model <- coffee_workflow %>%
    finalize_workflow(hyperparameters) %>%
    log_workflow_parameters() %>%
    fit(coffee_train)
  metrics <- fitted_coffee_model %>%
    predict(coffee_test) %>%
    metric_set(rmse, mae, rsq)(coffee_test$cupper_points, .pred) %>%
    log_metrics()
  crated_model <- carrier::crate(
    function(x) workflows:::predict.workflow(fitted_coffee_model, x),
    fitted_coffee_model = fitted_coffee_model
  )
  mlflow_save_model(crated_model, here::here("models"))
  mlflow_log_artifact(here::here("models", "crate.bin"))
})

fs::dir_tree("mlruns/1/")
