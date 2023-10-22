# *****************************************************************************
# Lab 6: Evaluation Metrics ----
#
# Course Code: BBT4206
# Course Name: Business Intelligence II
# Semester Duration: 21st August 2023 to 28th November 2023
#
# Lecturer: Allan Omondi
# Contact: aomondi [at] strathmore.edu
#
# Note: The lecture contains both theory and practice. This file forms part of
#       the practice. It has required lab work submissions that are graded for
#       coursework marks.
#
# License: GNU GPL-3.0-or-later
# See LICENSE file for licensing information.
# *****************************************************************************

# **[OPTIONAL] Initialization: Install and use renv ----
# The R Environment ("renv") package helps you create reproducible environments
# for your R projects. This is helpful when working in teams because it makes
# your R projects more isolated, portable and reproducible.

# Further reading:
#   Summary: https://rstudio.github.io/renv/
#   More detailed article: https://rstudio.github.io/renv/articles/renv.html

# "renv" It can be installed as follows:
# if (!is.element("renv", installed.packages()[, 1])) {
# install.packages("renv", dependencies = TRUE,
# repos = "https://cloud.r-project.org") # nolint
# }
# require("renv") # nolint

# Once installed, you can then use renv::init() to initialize renv in a new
# project.

# The prompt received after executing renv::init() is as shown below:
# This project already has a lockfile. What would you like to do?

# 1: Restore the project from the lockfile.
# 2: Discard the lockfile and re-initialize the project.
# 3: Activate the project without snapshotting or installing any packages.
# 4: Abort project initialization.

# Select option 1 to restore the project from the lockfile
# renv::init() # nolint

# This will set up a project library, containing all the packages you are
# currently using. The packages (and all the metadata needed to reinstall
# them) are recorded into a lockfile, renv.lock, and a .Rprofile ensures that
# the library is used every time you open the project.

# Consider a library as the location where packages are stored.
# Execute the following command to list all the libraries available in your
# computer:
.libPaths()

# One of the libraries should be a folder inside the project if you are using
# renv

# Then execute the following command to see which packages are available in
# each library:
lapply(.libPaths(), list.files)

# This can also be configured using the RStudio GUI when you click the project
# file, e.g., "BBT4206-R.Rproj" in the case of this project. Then
# navigate to the "Environments" tab and select "Use renv with this project".

# As you continue to work on your project, you can install and upgrade
# packages, using either:
# install.packages() and update.packages or
# renv::install() and renv::update()

# You can also clean up a project by removing unused packages using the
# following command: renv::clean()

# After you have confirmed that your code works as expected, use
# renv::snapshot(), AT THE END, to record the packages and their
# sources in the lockfile.

# Later, if you need to share your code with someone else or run your code on
# a new machine, your collaborator (or you) can call renv::restore() to
# reinstall the specific package versions recorded in the lockfile.

# [OPTIONAL]
# Execute the following code to reinstall the specific package versions
# recorded in the lockfile (restart R after executing the command):
# renv::restore() # nolint

# [OPTIONAL]
# If you get several errors setting up renv and you prefer not to use it, then
# you can deactivate it using the following command (restart R after executing
# the command):
# renv::deactivate() # nolint

# If renv::restore() did not install the "languageserver" package (required to
# use R for VS Code), then it can be installed manually as follows (restart R
# after executing the command):

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Introduction ----
# The choice of evaluation metric depends on the specific problem,
# the characteristics of the data, and the goals of the modeling task.
# It's often a good practice to use multiple evaluation metrics to gain a more
# comprehensive understanding of a model's performance.

# There are several evaluation metrics that can be used to evaluate algorithms.
# The default metrics used are:
## (1) "Accuracy" for classification problems and
## (2) "RMSE" for regression problems

# Accuracy is the percentage of correctly classified instances out of all
# instances. Accuracy is more useful in binary classification problems than
# in multi-class classification problems.

# On the other hand, Cohen's Kappa is similar to Accuracy however, it is more
# useful on classification problems that do not have an equal distribution of
# instances amongst the classes in the dataset.

# For example, instead of Red are 50 instances and Blue are 50 instances,
# the distribution can be that Red are 70 instances and Blue are 30 instances.

# STEP 1. Install and Load the Required Packages ----
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# 1. Accuracy and Cohen's Kappa ----
## 1.a. Load the dataset ----
library(readr)
census <- read_csv("data/census.csv")
View(census)

## 1.b. Determine the Baseline Accuracy ----
# Identify the number of instances that belong to each class (distribution or
# class breakdown).

# The result should show that 75% earn below or equal to 50K and 24% earn above 50K
# for quality.

# This means that an algorithm can achieve a 75% accuracy by
# predicting that all instances belong to the class "below or equal to 50k".

# This in turn implies that the baseline accuracy is 75%.

census_freq <- census$quality
cbind(frequency =
        table(census_freq),
      percentage = prop.table(table(census_freq)) * 100)

## 1.c. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(census$quality,
                                   p = 0.75,
                                   list = FALSE)
census_train <- census[train_index, ]
census_test <- census[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

# We then train a Generalized Linear Model to predict the value of quality
# (whether the patient will earn below or equal to 50k/above 50k).

# `set.seed()` is a function that is used to specify a starting point for the
# random number generator to a specific value. This ensures that every time you
# run the same code, you will get the same "random" numbers.
set.seed(7)
quality_model_glm <-
  train(quality ~ ., data = census_train, method = "glm",
        metric = "Accuracy", trControl = train_control)

## 1.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an accuracy of approximately 84% and a Kappa of approximately 56%.
print(quality_model_glm)

### Option 2: Compute the metric yourself using the test dataset ----
# A confusion matrix is useful for multi-class classification problems.
# Please watch the following video first: https://youtu.be/Kdsp6soqA7o

# The Confusion Matrix is a type of matrix which is used to visualize the
# predicted values against the actual Values. The row headers in the
# confusion matrix represent predicted values and column headers are used to
# represent actual values.

predictions <- predict(quality_model_glm, census_test[, 1:13])
confusion_matrix <-
  caret::confusionMatrix(predictions,as.factor(
                         census_test[, 1:14]$quality))
print(confusion_matrix)

### Option 3: Display a graphical confusion matrix ----

# Visualizing Confusion Matrix
fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

# 2. RMSE, R Squared, and MAE ----

# RMSE stands for "Root Mean Squared Error" and it is defined as the average
# deviation of the predictions from the observations.

# R Squared (R^2) is also known as the "coefficient of determination".
# It provides a goodness of fit measure for the predictions to the
# observations.

# NOTE: R Squared (R^2) is a value between 0 and 1 such that
# 0 refers to "no fit" and 1 refers to a "perfect fit".

## 2.a. Load the dataset ----
library(readr)
Crop_recommendation <- read_csv("data/Crop_recommendation.csv")
View(Crop_recommendation)

#census$workclass<-as.numeric(factor(census$workclass,levels=unique(census$workclass)))
#census$education_level<-as.numeric(factor(census$education_level,levels=unique(census$education_level)))
#census$marital_status<-as.numeric(factor(census$marital_status,levels=unique(census$marital_status)))
#census$occupation<-as.numeric(factor(census$occupation,levels=unique(census$occupation)))
#census$race<-as.numeric(factor(census$race,levels=unique(census$race)))
#census$sex<-as.numeric(factor(census$sex,levels=unique(census$sex)))
#census$native_country<-as.numeric(factor(census$native_country,levels=unique(census$native_country)))
#census$relationship<-as.numeric(factor(census$relationship,levels=unique(census$relationship)))

print(Crop_recommendation)
Crop_recommendation_no_na <- na.omit(Crop_recommendation)

## 2.b. Split the dataset ----
# Define a train:test data split of the dataset. Such that 10/14 are in the
# train set and the remaining 8/14 observations are in the test set.

# In this case, we split randomly without using a predictor variable in the
# caret::createDataPartition function.

# For reproducibility; by ensuring that you end up with the same
# "random" samples
set.seed(7)

# We apply simple random sampling using the base::sample function to get
# 13 samples
train_index <- sample(1:dim(Crop_recommendation)[1], 7) # nolint: seq_linter.
Crop_recommendation_train <- Crop_recommendation[train_index, ]
Crop_recommendation_test <- Crop_recommendation[-train_index, ]

## 2.c. Train the Model ----
# We apply bootstrapping with 1,000 repetitions
sapply(Crop_recommendation, class)
train_control <- trainControl(method = "boot", number = 1000)

# We then train a linear regression model to predict the value of Employed
# (the number of people that will be employed given the independent variables).
Crop_recommendation_model_lm <-
  train(quality ~ ., data = Crop_recommendation_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)

## 2.d. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an RMSE value of approximately 4.3898 and
# an R Squared value of approximately 0.8594
# (the closer the R squared value is to 1, the better the model).
print(Crop_recommendation_model_lm)

### Option 2: Compute the metric yourself using the test dataset ----
predictions <- predict(Crop_recommendation_model_lm, Crop_recommendation_test[, 1:2])

# These are the 6 values for employment that the model has predicted:
print(predictions)

#### RMSE ----
rmse <- sqrt(mean((Crop_recommendation_test$quality - predictions)^2))
print(paste("RMSE =", rmse))

#### SSR ----
# SSR is the sum of squared residuals (the sum of squared differences
# between observed and predicted values)
ssr <- sum((Crop_recommendation$quality - predictions)^2)
print(paste("SSR =", ssr))

#### SST ----
# SST is the total sum of squares (the sum of squared differences
# between observed values and their mean)
sst <- sum((Crop_recommendation_test$quality - mean(Crop_recommendation_test$quality))^2)
print(paste("SST =", sst))

#### R Squared ----
# We then use SSR and SST to compute the value of R squared
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))

#### MAE ----
# MAE measures the average absolute differences between the predicted and
# actual values in a dataset. MAE is useful for assessing how close the model's
# predictions are to the actual values.

# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.
absolute_errors <- abs(predictions - Crop_recommendation_test$quality)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))

# 3. Area Under ROC Curve ----
# Area Under Receiver Operating Characteristic Curve (AUROC) or simply
# "Area Under Curve (AUC)" or "ROC" represents a model's ability to
# discriminate between two classes.

# ROC is a value between 0.5 and 1 such that 0.5 refers to a model with a
# very poor prediction (essentially a random prediction; 50-50 accuracy)
# and an AUC of 1 refers to a model that predicts perfectly.

# ROC can be broken down into:
## Sensitivity ----
#         The number of instances from the first class (positive class)
#         that were actually predicted correctly. This is the true positive
#         rate, also known as the recall.
## Specificity ----
#         The number of instances from the second class (negative class)
#         that were actually predicted correctly. This is the true negative
#         rate.

## 3.a. Load the dataset ----
library(readr)
WineQT <- read_csv("data/WineQT.csv")
View(WineQT)

## 3.b. Determine the Baseline Accuracy ----
# The baseline accuracy is 65%.

WineQT_freq <- WineQT$quality
cbind(frequency =
        table(WineQT_freq),
      percentage = prop.table(table(WineQT_freq)) * 100)

## 3.c. Split the dataset ----
# Define an 80:20 train:test data split of the dataset.
train_index <- createDataPartition(WineQT$quality,
                                   p = 0.8,
                                   list = FALSE)
WineQT_train <- WineQT[train_index, ]
WineQT_test <- WineQT[-train_index, ]

## 3.d. Train the Model ----
# We apply the 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

# We then train a k Nearest Neighbours Model to predict the value of quality
# (whether the patient will test positive/negative for quality).

set.seed(7)
quality_model_knn <-
  train(quality ~ ., data = WineQT_train, method = "knn",
        metric = "ROC", trControl = train_control)

## 3.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show a ROC value of approximately 0.76 (the closer to 1,
# the higher the prediction accuracy) when the parameter k = 9
# (9 nearest neighbours).

print(quality_model_knn)

### Option 2: Compute the metric yourself using the test dataset ----
#### Sensitivity and Specificity ----
predictions <- predict(quality_model_knn, WineQT_test[, 1:12])
# These are the values for quality that the
# model has predicted:
print(predictions)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         WineQT_test[, 1:13]$quality)

# We can see the sensitivity (≈ 0.86) and the specificity (≈ 0.60) below:
print(confusion_matrix)

#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class qualitys.
predictions <- predict(quality_model_knn, WineQT_test[, 1:13],
                       type = "prob")

# These are the class probability values for quality that the
# model has predicted:
print(predictions)

# "Controls" and "Cases": In a binary classification problem, you typically
# have two classes, often referred to as "controls" and "cases."
# These classes represent the different outcomes you are trying to predict.
# For example, in a medical context, "controls" might represent patients without
# a disease, and "cases" might represent patients with the disease.

# Setting the Direction: The phrase "Setting direction: controls < cases"
# specifies how you define which class is considered the positive class (cases)
# and which is considered the negative class (controls) when calculating
# sensitivity and specificity.
roc_curve <- roc(WineQT_test$quality, predictions$neg)

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)

