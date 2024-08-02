library(caret)
library(tidyverse)
library(skimr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(doParallel)
library(xgboost)
library(ROCR)

insurance <- read.csv("insurance_data.csv")

# Step 1 Partition and preprocess data.

remove_cols <- nearZeroVar(insurance[,-86], names = TRUE)

insurance <-insurance %>% dplyr::select(-one_of(remove_cols))

insurance <- insurance %>% mutate_at(c(1, 4:43), as.factor)

dummies_model <- dummyVars(response~., data = insurance)
predictor_dummy <- data.frame(predict(dummies_model, newdata = insurance))
insurance <- cbind(response = insurance$response, predictor_dummy)

insurance$response <- as.factor(insurance$response)

#Rename the response from 0, 1 
insurance$response <- fct_recode(insurance$response,
                                 buy="1",
                                 notbuy= "0")

insurance$response <- relevel(insurance$response,
                              ref="buy")

typeof(insurance$response)

set.seed(99) #set random seed
index <- createDataPartition(insurance$response, p = .8,list = FALSE)
insurance_train <- insurance[index,]
insurance_test <- insurance[-index,]

################################################################################
# Step 2 Fit a Logistic regression model to the data.
set.seed(10)

lasso_model <- train(response ~.,
                     data = insurance_train,
                     method = "glmnet",
                     standarize = TRUE,
                     tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 20)),
                     trControl = trainControl(method = "cv", number = 5))


lasso_model

coef(lasso_model$finalModel, lasso_model$bestTune$lambda)

# STEP 3: predicted probability on Test set
insurance_pred_lasso <- predict(lasso_model, insurance_test, type = "prob")

insurance_pred_lasso

# STEP 4: AUC ROC Model Perfomance
pred_lasso <- prediction(insurance_pred_lasso$buy,
                         insurance_test$response,
                         label.ordering = c("notbuy", "buy")) # -ve class first, then +ve

perf_lasso <- performance(pred_lasso, "tpr", "fpr")

plot(perf_lasso, colorize = TRUE, main = "Logistic Regression AUC")

auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso 

################################################################################
# Step 2 Train a classification model.

set.seed(12)
insurance_tree_model <- train(response ~.,
                     data = insurance_train,
                     method = "rpart",
                     tuneGrid = expand.grid(cp = seq(0.01, 0.2, length = 5)),
                     trControl = trainControl(method = "cv",
                                              number = 5,
                                              classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                     metric = "ROC")

insurance_tree_model
plot(insurance_tree_model)
rpart.plot(insurance_tree_model$finalModel, type = 5)

# Step 3 get predicted probabilities.
predprob_insurance <- predict(insurance_tree_model, insurance_test, type = "prob")

pred_lasso <- prediction(predprob_insurance$buy,#Predicted probability of category name of positive class
                         insurance_test$response,#test set response variable
                         label.ordering = c("notbuy","buy"))
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE, main = "Classification AUC")

#Step 4 Get the AUC.
auc_lasso<-unlist(slot(performance(pred_lasso, "auc"), "y.values"))

auc_lasso

# A Classification tree generated an output similar to that of a logit function. 
# The AUC is split half-half, and the AUC value reflects exactly that at 0.5.
# Shows this is not a good algorithm. Moving to a Random Forest ensemble.

################################################################################
set.seed(8)
random_model <- train(response ~.,
                      method = "rf",
                      data = insurance_train,
                      tuneGrid = expand.grid(mtry = c(1, 3, 6, 9)),
                      trControl = trainControl(method = "cv",
                                               number = 5,
                                               classProbs = TRUE,
                                               summaryFunction = twoClassSummary),
                      metric = "ROC")

random_model

plot(random_model)

rpart.plot(random_model$finalModel, type = 5)

predprob_insurance <- predict(random_model, insurance_test, type = "prob")

pred_random <- prediction(predprob_insurance$buy,
                          insurance_test$response,
                          label.ordering = c("notbuy", "buy"))

perf_random <- performance(pred_random, "tpr", "fpr")

plot(perf_random, colorize = T)

auc_random <- unlist(slot(performance(pred_random, "auc"), "y.values"))

auc_random
# mtry = sqrt(ncol(insurance_train)) # 454 21.3

################################################################################
num_cores <- detectCores(logical = FALSE)
num_cores

cl <- makePSOCKcluster(num_cores-2)
registerDoParallel(cl)
library(randomForest)
set.seed(12)
start_time <- Sys.time()
model_gbm <- train(response ~.,
                   data = insurance_train,
                   method="xgbTree",
                   tuneGrid = expand.grid(
                     nrounds = c(50,200),
                     eta = c(0.025, 0.05),
                     max_depth = c(2, 3),
                     gamma = 0,
                     colsample_bytree = 1,
                     min_child_weight = 1,
                     subsample = 1),
                   trControl = trainControl(method = "cv",
                                            number=5,
                                            classProbs=TRUE,
                                            summaryFunction = twoClassSummary),
                   metric = "ROC")
end_time <- Sys.time()
print(end_time - start_time)

stopCluster(cl)
registerDoSEQ()

model_gbm

plot(varImp(model_gbm))



install.packages("SHAPforxgboost")
library(SHAPforxgboost)

Xdata <- as.matrix(select(insurance_train, -response))

#calculate SHAP values 
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)

shap.plot.summary.wrap1(model_gbm$finalModel, X= Xdata, top_n = 10)

p <- shap.plot.dependence(shap,
                          x = "total_car",
                          color_feature = "perc_lowereducation",
                          smooth = F,
                          jitter_width = 0.01,
                          alpha = 0.4) +
  ggtitle("total_car")
print(p)


shap.plot.summary(shap)

# Use 4 most important predictor variables
top4<-shap.importance(shap, names_only = TRUE)[1:4]
for (x in top4) {
  p <- shap.plot.dependence(
    shap, 
    x = x, 
    color_feature = "auto", 
    smooth = FALSE, 
    jitter_width = 0.01, 
    alpha = 0.4
  ) +
    ggtitle(x)
  print(p)
}


library(gridExtra)

top4 <- shap.importance(shap, names_only = TRUE)[1:4]

# Create a list to store individual plots
plots_list <- list()

# Generate individual plots
for (x in top4) {
  p <- shap.plot.dependence(
    shap, 
    x = x, 
    color_feature = "auto", 
    smooth = FALSE, 
    jitter_width = 0.01, 
    alpha = 0.4
  ) +
    ggtitle(x)
  
  plots_list[[length(plots_list) + 1]] <- p
}

# Combine and display the individual plots in one grid
grid.arrange(grobs = plots_list, ncol = 2)

#step 3: Step 3: Get Predictions using Testing Set Data

ins_prob<- predict(model_gbm, insurance_test, type ="prob")

#Step 4: Evaluate Model Performance
library(ROCR)
pred = prediction(ins_prob$buy, insurance_test$response,label.ordering =c("notbuy","buy")) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE, main = "XGBOOST AUC")

unlist(slot(performance(pred, "auc"), "y.values"))

