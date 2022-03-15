#a simple tidymodels workflow with mlflow

# based on https://www.tidymodels.org/


# load packages ----
library(tidymodels)
library(nycflights13)
library(skimr)


# load data ----
flight_data <-
  flights %>%
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = lubridate::as_date(time_hour)
  ) %>%
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>%
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance,
         carrier, date, arr_delay, time_hour) %>%
  # Exclude missing data
  na.omit() %>%
  mutate_if(is.character, as.factor)


# split data ----

set.seed(222)
# Put 3/4 of the data into the training set
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

# recipe ----
flights_rec <-
  recipe(arr_delay ~ ., data = train_data) %>%
  update_role(flight, time_hour, new_role = "ID") %>%
  step_date(date, features = c("dow", "month")) %>%
  step_holiday(date,
               holidays = timeDate::listHolidays("US"),
               keep_original_cols = FALSE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

# model spec ----
lr_mod <-
  logistic_reg() %>%
  set_engine("glm")

rf_mod <-
  rand_forest(trees = 10, mtry = 20) %>%
  set_engine("ranger") %>%
  set_mode("classification")


# workflow ----
flights_wflow <-
  workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(flights_rec)

# fit ----
flights_fit <-
  flights_wflow %>%
  fit(data = train_data)


# predict ----
flights_aug <- augment(flights_fit, test_data)
