# 2015 Predicted Woody Cover and Biophysical Variable Decision Tree-rpart engine
# Austin Rutherford
# arutherford@email.arizona.edu
# 2021-06-024

# Load packages
library(tidyverse)
library(tidymodels)
library(tidypredict)
library(vip)
library(doParallel)

# Read in data
pwc_data <- read_csv('Data/2015_allpts_NDVI_samples_full.csv')

# Check out the data
summary(pwc_data)
glimpse(pwc_data)

# Replace -9999 (missing data) value with NA, 
# Set negative 2015 NDVI Landsat predicted woody cover (pwc) values to 0,
# Divide pwc by 100 to convert percent to decimal,
pwc_clean <- pwc_data %>%
  mutate(clay = na_if(clay, -9999),
         depclay = na_if(depclay, -9999),
         fallmean = na_if(fallmean, -9999),
         fallsum = na_if(fallsum, -9999),
         ppt = na_if(ppt, -9999),
         slope = na_if(slope, -9999),
         springmean = na_if(springmean, -9999),
         springsum = na_if(springsum, -9999),
         summermean = na_if(summermean, -9999),
         summersum = na_if(summersum, -9999),
         tmax = na_if(tmax, -9999),
         washdist = na_if(washdist, -9999),
         wintermean = na_if(wintermean, -9999),
         wintersum = na_if(wintersum, -9999),
         summertmax = na_if(summertmax, -9999),
         summertmin = na_if(summertmin, -9999),
         springtmax = na_if(springtmax, -9999),
         springtmin = na_if(springtmin, -9999),
         wintertmax = na_if(wintertmax, -9999),
         wintertmin = na_if(wintertmin, -9999),
         falltmax = na_if(falltmax, -9999),
         falltmin = na_if(falltmin, -9999),
         pwc = replace(pwc, which(pwc<0),NA),
         pwc = pwc/100,
         pwcclass = as.factor(pwcclass),
         ecosite = as.factor(ecosite))

# Look at data again
glimpse(pwc_clean)

# Drop NAs and create data subset (half of the data)
pwc_samps <- pwc_clean %>% drop_na()

# Set up training and testing split
set.seed(4242)

# Train and test split
data_split <- initial_split(pwc_samps, prop = 0.80, strata = 'pwcclass')
pwc_train <- training(data_split)
pwc_test <- testing(data_split)

# Preprocess data
pwc_rec <- recipe(pwc ~ ., data = pwc_train) %>%
  step_rm(pwcclass, point, easting, northing) %>% # remove from model
  update_role(point, new_role = "id variable") %>% # make point an id for back tracking
  step_dummy(ecosite) %>% # use dummy variables for factors in model
  step_center(all_predictors()) %>% #mean center data, m = 0
  step_scale(all_predictors()) %>% #scale so sd = 1, finish normalizing
  step_nzv(all_predictors()) #remove zero variance (no info) cols

# Apply recipe preprocessing to training data
pwc_prepped <- prep(pwc_rec, training = pwc_train) # preps data, applies recipe

# Run (bake) prepped preprocessng to training data to see the number of final dummy variables
pwc_train_bake <- bake(pwc_prepped, new_data = pwc_train)

# Setup our model (using decision trees, C5 package/engine)
rpart_mod <- decision_tree(min_n = tune(), tree_depth = tune()) %>% 
  set_engine("rpart", parms = list(split ='gini')) %>% 
  set_mode("regression")

# Build workflow to pair model and cross validation and tuning with data preprocessing
pwc_wflow <- workflow() %>% 
  add_model(rpart_mod) %>% 
  add_recipe(pwc_rec)

# Set up cross validation
folds <- vfold_cv(pwc_train, v = 5, repeats = 5) #5-fold (v) cross validation, only do it once (no repeats)
folds

# Set up the initial tuning grid for finding best min_n (hyperparameters)
# Match min_n to sqrt (rounded up) of number of predictor variables (31)
rpart_param <-
  pwc_wflow %>%
  parameters() %>%
  update(min_n = min_n(range = c(2L, 31L)),
         tree_depth = tree_depth(range = c(1L, 31L)))

rpart_tune_grid <- grid_regular(rpart_param, levels = 31)

rpart_tune_grid

# Training rpart decision tree model/tuning
# First, initiate parallel processing
num_cores <- 100
cl <- makeCluster(num_cores, outfile = "", type = "FORK")
clusterEvalQ(cl, library(tidymodels))
registerDoParallel(cl)

# Run rpart decision tree models to tune hyperparameters
# rpart_fit_results <- tune_grid(pwc_wflow, 
#                             resamples = folds,
#                             grid = rpart_tune_grid,
#                             metrics = metric_set(rsq, mae, rmse))

# Save an RDS file for future use if needed, save that memory
#saveRDS(rpart_fit_results, "./rpart_rds/first_model_dectree_full.rds")

# Load RDS
rpart_fit_results <- readRDS("./rpart_rds/first_model_dectree_full.rds")

# Graphs rmse for all min_n
rpart_param_plot <- rpart_fit_results %>%
  collect_metrics() %>%
  dplyr::filter(.metric == "rmse") %>%
  dplyr::select(mean, min_n, tree_depth) %>%
  tidyr::pivot_longer(min_n:tree_depth,
                      values_to = "value",
                      names_to = "parameter") %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "RMSE")

rpart_param_plot

#ggsave('Graphs/rpart_param_plot.png', plot = rpart_param_plot)

# Look at best decision tree models based on the Mean Absolute Error (and/or RMSE)
show_best(rpart_fit_results, metric = "mae")
show_best(rpart_fit_results, metric = "rmse")

# Update tuning grid based on best +1, -1 range of min_n and mtry
rpart_grid_update <- grid_regular(
  tree_depth(range = c(3, 5)),
  min_n(range = c(2, 10)),
  levels = 16)

# Tune decision tree with targets min_n and tree depth ranges
# rpart_fit_update <- tune_grid(pwc_wflow, 
#                            resamples = folds,
#                            grid = rpart_grid_update,
#                            metrics = metric_set(rmse, mae, rsq))

# Save an RDS file for future use if needed, save that memory
#saveRDS(rpart_fit_update, "./rpart_rds/final_model_dectree_full.rds")

# Load RDS
rpart_fit_update <- readRDS("./rpart_rds/final_model_dectree_full.rds")

# Graphs rmse for targeted mtry and min_n
rpart_param_plot_tune <- rpart_fit_update  %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(tree_depth, 100*mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "RMSE")

rpart_param_plot_tune

#ggsave('Graphs/rpart_param_plot_tune.png', plot = rpart_param_plot_tune)

# Pick the best set of hyperparameters
show_best(rpart_fit_update, metric = "rmse")

# Pull out parameters of the best model based on RMSE (for prediction), min_n = 2
pwc_rpart_best <- rpart_fit_update %>% select_best(min_n, metric = "rmse")

# Finalize the model with the parameters of the best selected model
final_rpart <- finalize_model(rpart_mod, pwc_rpart_best)

# Rerun final model and output the variable importance based on the GINI index, point is ID only
pwc_vip <- final_rpart %>%
  fit(pwc ~ .,
      data = pwc_train_bake) %>%
  vip(geom = "point", num_features = 31) # variable importance graph

pwc_vip

#ggsave('Graphs/pwc_vip_31.png', plot = pwc_vip)

# Finalize Workflow
final_wf <- workflow() %>%
  add_recipe(pwc_rec) %>%
  add_model(final_rpart)

# Function for unregistrering cluster for parallel processing, problems with 'ranger'?  
#(https://stackoverflow.com/questions/25097729/un-register-a-doparallel-cluster)
# unregister <- function() {
#   env <- foreach:::.foreachGlobals
#   rm(list=ls(name=env), pos=env)
# }

# Apply last workflow to training and testing data
final_res <- final_wf %>%
  last_fit(data_split)

# Look at final model performance metrics (mae, rmse, rsq)
final_res %>%
  collect_metrics()

# Save an RDS file for future use if needed, save that memory
#saveRDS(final_res, "./rpart_rds/final_model_dectree_full_tuned.rds")

# Load RDS
final_res <- readRDS("./rpart_rds/final_model_dectree_full_tuned.rds")

# Compare predicted pwc values to test pwc
final_res %>% 
  collect_predictions(summarize = FALSE) %>% arrange(desc(pwc))

# Graph of predicted values to testing pwc
rpart_plot <- final_res %>%
  collect_predictions() %>%
  ggplot(aes(pwc, .pred)) +
  geom_point(size = 0.5, alpha = 0.5) +
  labs(y = 'Decision Tree Predicted Woody Cover (%)', x = 'Landsat NDVI Predicted Woody Cover (%)') +
  ylim(c(0, 0.60))+
  xlim(c(0, 0.80))+
  scale_color_manual(values = c("gray80", "darkred"))+
  geom_abline(intercept = 0, slope = 1, color = "blue", size = 1)+
  stat_smooth(method = "lm", formula = y ~ x, color = "red")+
  ggpmisc::stat_poly_eq(formula = y ~ x, 
                        aes(label =  paste(stat(eq.label),
                                           stat(rr.label), stat(p.value.label), sep = "*\", \"*")),
                        parse = TRUE)+
  theme_bw()

rpart_plot

# Save comparison plot of full decision tree model
#ggsave('Graphs/DecTree_Model_Full.png', plot = rpart_plot)

### Variable Selection based on VIP/Impurity, create new/small model ###
# Limit the number of predictors to the top 9 based on full model impurity, excluding spatial data for mapping later
pwc_rec_small <- recipe(pwc ~ point + easting + northing + elevation + clay + aspect + fallsum + fallmean + wintersum + wintermean + slope + washdist, data = pwc_train) %>%
  step_rm(point, easting, northing) %>% # remove from model
  update_role(point, new_role = "id variable") %>% # make point an id for back tracking
  step_center(all_predictors()) %>% #mean center data, m = 0
  step_scale(all_predictors()) %>% #scale so sd = 1, finish normalizing
  step_nzv(all_predictors()) #remove zero variance (no info) cols

# Apply recipe preprocessing to training data
pwc_prepped_small <- prep(pwc_rec_small, training = pwc_train) # preps data, applies recipe

# Run (bake) prepped preprocessng to training data to see the number of final dummy variables
pwc_train_bake_small <- bake(pwc_prepped_small, new_data = pwc_train)

# Create a small decision tree model to tune
rpart_mod_small <- decision_tree(min_n = tune(), tree_depth = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

# Create small workflow
final_wf_small <- workflow() %>%
  add_recipe(pwc_rec_small) %>%
  add_model(rpart_mod_small)

# Create tuning grid to find optimal tree depth and min_n with 9 predictors
# Match tree depth with number of predictor variables (9)
# Match min_n to sqrt (rounded up) of number of predictor variables (3)
rpart_grid_small <- grid_regular(
  tree_depth(range = c(1, 9)),
  min_n(range = c(2, 4)),
  levels = 9)

# Tune small decision tree model with targets min_n and mtry
# rpart_fit_small <- tune_grid(final_wf_small, 
#                           resamples = folds,
#                           grid = rpart_grid_small,
#                           metrics = metric_set(rmse, mae, rsq))

# Save an RDS file for future use if needed, save that memory
#saveRDS(rpart_fit_small, "./rpart_rds/first_model_dectree_small.rds")

# Load RDS
rpart_fit_small <- readRDS("./rpart_rds/first_model_dectree_small.rds")

# Graphs rmse for targeted mtry and min_n using small decision tree model
rpart_param_plot_small <- rpart_fit_small  %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(tree_depth, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "rmse")

rpart_param_plot_small

#ggsave('Graphs/rpart_param_plot_small.png', plot = rpart_param_plot_small)

# Pick the best set of hyperparameters
show_best(rpart_fit_small, metric = "mae")
show_best(rpart_fit_small, metric = "rsq")
show_best(rpart_fit_small, metric = "rmse")

# Pull out parameters of the best model based on RMSE (for prediction), mtry = 4, min_n = 2
pwc_rpart_small <- rpart_fit_small %>% 
  select_best(metric = "rmse")

# Finalize the model with the parameters of the best selected model
final_rpart_small <- finalize_model(rpart_mod_small, pwc_rpart_small)

# Rerun final model and output the variable importance based on the GINI index
pwc_vip_small <- final_rpart_small %>%
  set_engine("rpart") %>%
  fit(pwc ~ elevation + clay + aspect + fallsum + fallmean + wintersum + wintermean + slope + washdist,
      data = pwc_train_bake_small) %>%
  vip(geom = "point", num_features = 9) # variable importance graph

pwc_vip_small

# Save VIP plot of the small/9 variable decision tree model
#ggsave('Graphs/pwc_vip_9.png', plot = pwc_vip_small)

# Finalize Workflow with 9 predictors
final_wf_small <- workflow() %>%
  add_recipe(pwc_rec_small) %>%
  add_model(final_rpart_small)

# Final decision tree model run with testing data
final_res_small <- final_wf_small %>%
  last_fit(data_split)

# Look at final model performance metrics (mae, rmse, rsq)
final_res_small %>%
  collect_metrics()

# Save the RDS file for further use, save that memory
#saveRDS(final_res_small, "./rpart_rds/final_model_dectree_small.rds")

# Load RDS
final_res_small<- readRDS("./rpart_rds/final_model_dectree_small.rds")

# Compare predicted pwc values to test pwc
final_res_small %>% 
  collect_predictions(summarize = FALSE) %>% arrange(desc(pwc))

# graph of predicted values to testing pwc
rpart_plot_small <- final_res_small %>%
  collect_predictions() %>%
  ggplot(aes(100*pwc, 100*.pred)) +
  geom_point(size = 0.5, alpha = 0.5) +
  labs(y = 'Decision Tree Predicted Woody Cover (%)', x = 'Landsat NDVI Predicted Woody Cover (%)') +
  ylim(c(0, 60))+
  xlim(c(0, 80))+
  scale_color_manual(values = c("gray80", "darkred"))+
  geom_abline(intercept = 0, slope = 1, color = "blue", size = 1)+
  stat_smooth(method = "lm", formula = y ~ x, color = "red")+
  ggpmisc::stat_poly_eq(formula = y ~ x, 
                        aes(label =  paste(stat(eq.label),
                                           stat(rr.label), stat(p.value.label), sep = "*\", \"*")),
                        parse = TRUE)+
  theme_bw()

rpart_plot_small

# Save comparison plot of small/9 variable decision tree model
#ggsave('Graphs/DecTree_Model_Small.png', plot = rpart_plot_small)


# Apply decision tree model to create predictions on full raster
### Below only needed if using prediction, commenting out because pred aren't great but keeping code just in case ###
# # Load RDS file if need to shut down R
#rpart_fit_small_last <- readRDS("./rpart_rds/final_model_dectree_small.rds")

# rpart_fit_small_last_pred <- as.data.frame(rpart_fit_small_last$.predictions)
# 
# # Apply recipe preprocessing to training data
# pwc_full_prep <- prep(pwc_rec, training = pwc_samps) # preps data, applies recipe
# 
# # Run (bake) prepped preprocessng to training data to see the number of final dummy variables
# pwc_full_baked <- bake(pwc_prepped, new_data = pwc_samps)
# 
# # take final (best) decision tree model, run on VIPs with baked (recipe applied) full dataset
# rpart_mod_pred <- final_rpart %>% 
#   fit(pwc ~ elevation + clay + aspect + fallsum + fallmean + wintersum + wintermean + slope + washdist,
#       data = pwc_full_baked)
# 
# # Save the RDS file for further use, save that memory
# saveRDS(rpart_mod_pred, "./rpart_rds/pred_model_dectree_small.rds")
# 
# # look at fit
# rpart_mod_pred$fit
# 
# # look at the predictions
# rpart_mod_pred$fit$predictions
# 
# # look at variable importance based on GINI impurity
# rpart_mod_pred$fit$variable.importance
# 
# # create new table with only point, easting, northing, original pwc, and predicted pwc
# pwc_small_rpart_comb <- pwc_samps %>% 
#   mutate(pred = rpart_mod_pred$fit$predictions) %>% 
#   dplyr::select(point, easting, northing, pwc, pred) %>% 
#   rename(predicted_pwc = pred)
# 
# # look at good of fit between predicted and original pwc values
# pwc_small_rpart_comb %>% 
#   ggplot(aes(100*pwc, 100*predicted_pwc)) +
#   geom_point(size = 0.5, alpha = 0.5) +
#   labs(y = 'Decision Trees Predicted Woody Cover (%)', x = 'Landsat NDVI Predicted Woody Cover (%)') +
#   ylim(c(0, 60))+
#   xlim(c(0, 80))+
#   scale_color_manual(values = c("gray80", "darkred"))+
#   geom_abline(intercept = 0, slope = 1, color = "blue", size = 1)+
#   stat_smooth(method = "lm", formula = y ~ x, color = "red")+
#   ggpmisc::stat_poly_eq(formula = y ~ x, 
#                         aes(label =  paste(stat(eq.label),
#                                            stat(rr.label), stat(p.value.label), sep = "*\", \"*")),
#                         parse = TRUE)+
#   theme_bw()
# 
# 
# # write the original and predicted pwc values to new .csv
# write_csv(pwc_small_rpart_comb, path = "Data/rpart_model_pred.csv")

# Stop the cluster/parallel processing
stopCluster(cl)
