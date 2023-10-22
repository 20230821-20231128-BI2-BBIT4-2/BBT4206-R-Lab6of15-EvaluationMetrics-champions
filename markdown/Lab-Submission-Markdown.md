Business Intelligence Project
================
<Champions>
\<22/10/2023\>

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [STEP 1 : Install and load all the
  packages](#step-1--install-and-load-all-the-packages)
- [Accuracy and Cohen’s Kappa](#accuracy-and-cohens-kappa)
  - [Load the Dataset](#load-the-dataset)
  - [Determine the Baseline Accuracy](#determine-the-baseline-accuracy)
  - [Split the dataset](#split-the-dataset)
  - [Train the Model](#train-the-model)
    - [Apply 5 fold cross validation resampling
      method](#apply-5-fold-cross-validation-resampling-method)
    - [Train a Generalized Linear Model to predict the value of
      Diabetes](#train-a-generalized-linear-model-to-predict-the-value-of-diabetes)
  - [Display the Model’s Performance](#display-the-models-performance)
    - [Use the metric calculated by caret when training the
      model](#use-the-metric-calculated-by-caret-when-training-the-model)
    - [Compute the metric yourself using the test
      dataset](#compute-the-metric-yourself-using-the-test-dataset)
    - [Display a graphical confusion
      matrix](#display-a-graphical-confusion-matrix)
- [STEP 2 : RMSE, R Squared, and MAE](#step-2--rmse-r-squared-and-mae)
  - [2.a. Load the dataset](#2a-load-the-dataset)
  - [2.b. Split the dataset](#2b-split-the-dataset)
    - [Perform reproducibility](#perform-reproducibility)
    - [Apply simple random sampling using the
      base](#apply-simple-random-sampling-using-the-base)
  - [2.c. Train the Model](#2c-train-the-model)
    - [Apply bootstrapping with 1,000
      repetitions](#apply-bootstrapping-with-1000-repetitions)
    - [Train a linear regression model to predict the value of
      Employed](#train-a-linear-regression-model-to-predict-the-value-of-employed)
  - [2.d. Display the Model’s
    Performance](#2d-display-the-models-performance)
    - [1.Use the metric calculated by caret when training the
      model](#1use-the-metric-calculated-by-caret-when-training-the-model)
    - [2.Compute the metric yourself using the test
      dataset](#2compute-the-metric-yourself-using-the-test-dataset)
      - [Print the predictions for
        employment](#print-the-predictions-for-employment)
      - [RMSE](#rmse)
      - [SSR](#ssr)
      - [SST](#sst)
      - [R Squared](#r-squared)
      - [MAE](#mae)
- [STEP 3. Area Under ROC Curve](#step-3-area-under-roc-curve)
  - [3.a. Load the dataset](#3a-load-the-dataset)
  - [3.b. Determine the Baseline
    Accuracy](#3b-determine-the-baseline-accuracy)
  - [3.c. Split the dataset](#3c-split-the-dataset)
  - [3.d. Train the Model](#3d-train-the-model)
    - [Apply the 10-fold cross validation resampling
      method](#apply-the-10-fold-cross-validation-resampling-method)
    - [Train a k Nearest Neighbours Model to predict the value of
      Diabetes](#train-a-k-nearest-neighbours-model-to-predict-the-value-of-diabetes)
  - [3.e. Display the Model’s
    Performance](#3e-display-the-models-performance)
    - [Use the metric calculated by caret when training the
      model](#use-the-metric-calculated-by-caret-when-training-the-model-1)
    - [Compute the metric yourself using the test
      dataset](#compute-the-metric-yourself-using-the-test-dataset-1)
    - [The values that have been
      predicted](#the-values-that-have-been-predicted)
    - [Sensitivity and specificity](#sensitivity-and-specificity)
    - [AUC](#auc)
    - [Class predictions](#class-predictions)
    - [Setting the Direction](#setting-the-direction)
    - [Plot the ROC curve](#plot-the-roc-curve)
- [STEP 4. Logarithmic Loss (LogLoss)](#step-4-logarithmic-loss-logloss)
  - [4.a. Load the dataset](#4a-load-the-dataset)
  - [4.b. Train the Model](#4b-train-the-model)
    - [Create a CART](#create-a-cart)
  - [4.c. Display the Model’s
    Performance](#4c-display-the-models-performance)
    - [Use the metric calculated by caret when training the
      model](#use-the-metric-calculated-by-caret-when-training-the-model-2)

# Student Details

<table style="width:99%;">
<colgroup>
<col style="width: 65%" />
<col style="width: 33%" />
</colgroup>
<tbody>
<tr class="odd">
<td><strong>Student ID Number</strong></td>
<td><p>134111</p>
<p>133996</p>
<p>126761</p>
<p>135859</p>
<p>127707</p></td>
</tr>
<tr class="even">
<td><strong>Student Name</strong></td>
<td><p>Juma Immaculate Haayo</p>
<p>Trevor Ngugi</p>
<p>Virginia Wanjiru</p>
<p>Pauline Wang’ombe</p>
<p>Clarice Gitonga</p></td>
</tr>
<tr class="odd">
<td><strong>BBIT 4.2 Group</strong></td>
<td>B</td>
</tr>
<tr class="even">
<td><strong>BI Project Group Name/ID (if applicable)</strong></td>
<td>Champions</td>
</tr>
</tbody>
</table>

# Setup Chunk

**Note:** the following KnitR options have been set as the global
defaults: <BR>
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

# STEP 1 : Install and load all the packages

We installed all the packages that will enable us execute this lab.

``` r
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: ggplot2

``` r
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: caret

    ## Loading required package: lattice

``` r
## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: mlbench

``` r
## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: pROC

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: dplyr

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(readr)
```

# Accuracy and Cohen’s Kappa

It is used in cases where classification problems that do not have an
equal distribution of instances amongst the classes in the dataset.

## Load the Dataset

We then proceeded to load the census dataset

``` r
library(readr)
census <- read_csv("../data/census.csv")
```

    ## Rows: 45222 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (9): workclass, education_level, marital_status, occupation, relationshi...
    ## dbl (5): age, education-num, capital-gain, capital-loss, hours-per-week
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(census)
```

## Determine the Baseline Accuracy

Identify the number of instances that belong to each class (distribution
or class breakdown).The result should show that 75% earn below or equal
to 50K and 24% earn above 50K for income.This means that an algorithm
can achieve a 75% accuracy by predicting that all instances belong to
the class “below or equal to 50k”.This in turn implies that the baseline
accuracy is 75%.

``` r
census_freq <- census$income
cbind(frequency =
        table(census_freq),
      percentage = prop.table(table(census_freq)) * 100)
```

    ##       frequency percentage
    ## <=50K     34014    75.2156
    ## >50K      11208    24.7844

## Split the dataset

Here, we split the dataset so that some portion of it is used for
training and another for training. In our case 75% is used for training
the dataset and the remaining 25% is used for testing the dataset. list=
false means we retain the structure train_index means the 75% that will
be used for training the dataset and -train_index means the remaining
portion that was not used will now be used for testing the dataset.

``` r
train_index <- createDataPartition(census$income,
                                   p = 0.75,
                                   list = FALSE)
census_train <- census[train_index, ]
census_test <- census[-train_index, ]
```

## Train the Model

The model can be trained by first applying a 5 fold cross validation
method then train a generalized linear model on the dataset.

### Apply 5 fold cross validation resampling method

``` r
train_control <- trainControl(method = "cv", number = 5)
```

### Train a Generalized Linear Model to predict the value of Diabetes

seed(7)will ensure we get the same “random” numbers. The method used is
generalized linear model and the metric used is Accuracy. In the code
below we are predicting census data which is the dependent variable in
line with the other independent variables.

``` r
set.seed(7)
income_model_glm <-
  train(income ~ ., data = census_train, method = "glm",
        metric = "Accuracy", trControl = train_control)
```

## Display the Model’s Performance

There are various ways to display the model’s performance.

### Use the metric calculated by caret when training the model

The result acquired after running the code is an Accuracy of 78% and
Kappa of 50% .We notice the Accuracy is higher than the baseline
accuracy.

``` r
print(income_model_glm)
```

    ## Generalized Linear Model 
    ## 
    ## 33917 samples
    ##    13 predictor
    ##     2 classes: '<=50K', '>50K' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 27134, 27134, 27133, 27134, 27133 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.8474217  0.5647964

### Compute the metric yourself using the test dataset

We use a confusion matrix for multi-class classification problems. The
Confusion Matrix visualize the predicted values against the actual
values.

``` r
predictions <- predict(income_model_glm, census_test[, 1:13])
confusion_matrix <-
  caret::confusionMatrix(predictions,as.factor(
                         census_test[, 1:14]$income))
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction <=50K >50K
    ##      <=50K  7903 1130
    ##      >50K    600 1672
    ##                                           
    ##                Accuracy : 0.847           
    ##                  95% CI : (0.8402, 0.8536)
    ##     No Information Rate : 0.7521          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5618          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ##                                           
    ##             Sensitivity : 0.9294          
    ##             Specificity : 0.5967          
    ##          Pos Pred Value : 0.8749          
    ##          Neg Pred Value : 0.7359          
    ##              Prevalence : 0.7521          
    ##          Detection Rate : 0.6991          
    ##    Detection Prevalence : 0.7990          
    ##       Balanced Accuracy : 0.7631          
    ##                                           
    ##        'Positive' Class : <=50K           
    ## 

### Display a graphical confusion matrix

Here is where the confusion matrix is shown in light blue and grey.

``` r
fourfoldplot(as.table(confusion_matrix), color = c("grey", "navyblue"),
             main = "Confusion Matrix")
```

![](Lab-Submission-Markdown_files/figure-gfm/Your%20ninth%20Code%20Chunk-1.png)<!-- -->

# STEP 2 : RMSE, R Squared, and MAE

## 2.a. Load the dataset

``` r
Student_Marks <- read_csv("../data/Student_Marks.csv")
```

    ## Rows: 100 Columns: 3
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (3): number_courses, time_study, Marks
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(Student_Marks)
student_no_na <- na.omit(Student_Marks)
```

## 2.b. Split the dataset

A portion of the dataset is used to train and another to test.

### Perform reproducibility

The code below ensures we get the same “random” samples

``` r
set.seed(7)
```

### Apply simple random sampling using the base

We have applied a function to get 10 samples

``` r
train_index <- sample(1:dim(Student_Marks)[1], 10) # nolint: seq_linter.
Student_Marks_train <- Student_Marks[train_index, ]
Student_Marks_test <- Student_Marks[-train_index, ]
```

## 2.c. Train the Model

### Apply bootstrapping with 1,000 repetitions

``` r
train_control <- trainControl(method = "boot", number = 1000)
```

### Train a linear regression model to predict the value of Employed

We are calculating the number of people that will be employed given the
independent variables

``` r
Student_Marks_model_lm <-
  train(Marks ~ ., data = Student_Marks_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)
```

## 2.d. Display the Model’s Performance

### 1.Use the metric calculated by caret when training the model

It results in the RMSE value of approximately 4.3898 and an R Squared
value of approximately 0.8594 (the closer the R squared value is to 1,
the better the model).

``` r
print(Student_Marks_model_lm)
```

    ## Linear Regression 
    ## 
    ## 10 samples
    ##  2 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (1000 reps) 
    ## Summary of sample sizes: 10, 10, 10, 10, 10, 10, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   3.324156  0.9619118  2.816464
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

### 2.Compute the metric yourself using the test dataset

In the code below,we are predicting the student marks by focusing on the
first and second variables.

``` r
predictions <- predict(Student_Marks_model_lm, Student_Marks_test[, 1:2])
```

#### Print the predictions for employment

It shows the 6 values for employment that the model has predicted

``` r
print(predictions)
```

    ##           1           2           3           4           5           6 
    ## 23.25538185 -1.58797995 16.31622566 46.80638593 48.56256590 19.10998884 
    ##           7           8           9          10          11          12 
    ## 32.42266526 23.84459924 33.07115477 40.02767851  3.84069022 26.21357804 
    ##          13          14          15          16          17          18 
    ## 21.87586782 22.96029537 32.72858690 38.35769929 45.63910483 25.95975908 
    ##          19          20          21          22          23          24 
    ## 25.74880023 36.29496064 11.10440969 30.08682898 19.27569593  4.97387325 
    ##          25          26          27          28          29          30 
    ##  4.34179575 28.32379808 -0.06729582  5.50413705  8.01587855  8.00839059 
    ##          31          32          33          34          35          36 
    ## 39.05398870 20.74985423 41.14779674 15.33170069 43.11589113 39.86133438 
    ##          37          38          39          40          41          42 
    ## 46.01640781 -1.32858415 21.35022530  8.19210229  9.36559727  8.82943811 
    ##          43          44          45          46          47          48 
    ## 20.23074411 26.01998675 29.89945157 22.08156835 39.69658284 24.52782386 
    ##          49          50          51          52          53          54 
    ##  2.39138744 37.00798057 47.26558690 18.52013441 47.56035485 46.44055514 
    ##          55          56          57          58          59          60 
    ## 33.21853874 46.54045773  2.39664576 12.21895108 21.43865568 25.38360649 
    ##          61          62          63          64          65          66 
    ##  2.11335001 -0.07844950  6.07009153 23.45486850 19.87112720 13.85754239 
    ##          67          68          69          70          71          72 
    ## 23.99230174  6.28264298 38.84239281  1.41307635 36.61952391 37.79764017 
    ##          73          74          75          76          77          78 
    ## 45.27327404  1.42486707 48.25632575 15.84921821 17.86049123 14.19851766 
    ##          79          80          81          82          83          84 
    ## 39.07103774 25.32337882  9.64236063 26.00134511 27.48203581 40.66278470 
    ##          85          86          87          88          89          90 
    ## 25.61782825 21.17336452 -1.54639392 40.07452287  3.16861929 34.02620294

#### RMSE

``` r
rmse <- sqrt(mean((Student_Marks_test$Marks - predictions)^2))
print(paste("RMSE =", rmse))
```

    ## [1] "RMSE = 4.10052465373179"

#### SSR

``` r
ssr <- sum((Student_Marks_test$Marks - predictions)^2)
print(paste("SSR =", ssr))
```

    ## [1] "SSR = 1513.2872192276"

#### SST

``` r
sst <- sum((Student_Marks_test$Marks - mean(Student_Marks_test$Marks))^2)
print(paste("SST =", sst))
```

    ## [1] "SST = 18896.106836"

#### R Squared

We then use SSR and SST to compute the value of R squared.

``` r
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))
```

    ## [1] "R Squared = 0.919915396734286"

#### MAE

``` r
absolute_errors <- abs(predictions - Student_Marks_test$Marks)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))
```

    ## [1] "MAE = 3.25681795550345"

# STEP 3. Area Under ROC Curve

## 3.a. Load the dataset

``` r
data(PimaIndiansDiabetes)
library(readr)
Customer_Churn <- read_csv("../data/Customer Churn.csv")
```

    ## Rows: 3150 Columns: 14
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (14): Call  Failure, Complains, Subscription  Length, Charge  Amount, Se...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
Customer_Churn$Churn <- ifelse(Customer_Churn$Churn == 0, "No", "Yes")

View(Customer_Churn)
```

## 3.b. Determine the Baseline Accuracy

The percentages of No and yes is 84 and 16 respectively.

``` r
Customer_Churn_freq <- Customer_Churn$Churn
cbind(frequency =
        table(Customer_Churn_freq),
      percentage = prop.table(table(Customer_Churn_freq)) * 100)
```

    ##     frequency percentage
    ## No       2655   84.28571
    ## Yes       495   15.71429

## 3.c. Split the dataset

80% of the dataset is used for testing and 20% is used for training.

``` r
train_index <- createDataPartition(Customer_Churn$Churn,
                                   p = 0.8,
                                   list = FALSE)
Customer_Churn_train <- Customer_Churn[train_index, ]
Customer_Churns_test <- Customer_Churn[-train_index, ]
```

## 3.d. Train the Model

### Apply the 10-fold cross validation resampling method

``` r
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)
```

### Train a k Nearest Neighbours Model to predict the value of Diabetes

We want to see whether the patient will test positive/negative for
diabetes

``` r
set.seed(7)
churn_model_knn <-
  train(Churn ~ ., data = Customer_Churn_train, method = "knn",
        metric = "ROC", trControl = train_control)
```

## 3.e. Display the Model’s Performance

### Use the metric calculated by caret when training the model

The results show a ROC value of approximately 0.76 (the closer to 1, the
higher the prediction accuracy) when the parameter k = 9 (9 nearest
neighbours).

``` r
print(churn_model_knn)
```

    ## k-Nearest Neighbors 
    ## 
    ## 2520 samples
    ##   13 predictor
    ##    2 classes: 'No', 'Yes' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 2268, 2268, 2267, 2268, 2267, 2269, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  ROC        Sens       Spec     
    ##   5  0.8163272  0.9354925  0.3989103
    ##   7  0.8240459  0.9383227  0.3636538
    ##   9  0.8310128  0.9500974  0.3308333
    ## 
    ## ROC was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 9.

### Compute the metric yourself using the test dataset

``` r
predictions <- predict(churn_model_knn, Customer_Churns_test[, 1:13])
```

### The values that have been predicted

``` r
print(predictions)
```

    ##   [1] No  Yes No  No  No  Yes No  No  Yes No  No  No  No  No  No  No  No  No 
    ##  [19] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ##  [37] No  No  No  No  No  No  No  No  No  Yes No  No  No  No  No  No  No  No 
    ##  [55] No  No  No  No  No  Yes No  No  No  No  No  No  No  No  Yes Yes No  No 
    ##  [73] No  No  No  No  No  No  No  No  No  No  No  No  Yes No  No  Yes No  No 
    ##  [91] No  No  No  No  No  No  No  No  No  No  No  No  Yes No  No  No  No  No 
    ## [109] No  No  No  No  No  No  No  No  No  Yes No  No  No  No  No  No  No  No 
    ## [127] No  No  No  Yes No  Yes No  No  No  No  No  No  No  No  No  No  No  No 
    ## [145] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [163] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [181] No  No  No  No  No  No  No  Yes No  No  No  No  No  No  No  No  Yes No 
    ## [199] No  No  No  No  No  Yes No  No  No  No  No  Yes No  No  No  No  No  No 
    ## [217] No  No  No  No  No  No  No  Yes No  No  Yes No  No  No  No  No  No  No 
    ## [235] No  No  No  No  No  No  No  No  Yes No  Yes No  No  No  Yes No  No  No 
    ## [253] No  No  No  Yes No  No  No  No  No  No  Yes No  No  No  No  No  No  No 
    ## [271] No  No  Yes Yes No  No  No  No  No  No  No  No  No  Yes No  No  No  Yes
    ## [289] No  No  No  No  No  No  Yes No  No  No  No  No  No  No  No  No  Yes No 
    ## [307] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [325] Yes No  No  No  No  No  Yes No  No  No  Yes No  No  No  No  No  No  No 
    ## [343] No  No  No  No  No  No  No  No  No  No  No  Yes No  No  No  Yes No  No 
    ## [361] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [379] No  No  No  No  No  No  No  No  Yes Yes Yes No  Yes No  No  Yes No  No 
    ## [397] Yes No  No  No  No  No  No  No  No  No  Yes Yes Yes No  No  No  No  No 
    ## [415] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [433] No  No  No  No  No  No  No  No  No  No  No  No  Yes No  No  No  No  No 
    ## [451] No  No  No  No  No  No  No  Yes No  No  No  No  No  Yes No  Yes Yes No 
    ## [469] Yes No  No  No  No  No  No  No  No  No  No  No  Yes No  No  No  No  Yes
    ## [487] No  No  No  Yes No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [505] No  No  No  Yes No  No  No  No  Yes Yes Yes No  No  No  No  No  No  No 
    ## [523] No  No  No  Yes No  No  No  No  No  No  No  No  No  Yes No  No  No  No 
    ## [541] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [559] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [577] No  No  No  Yes No  No  No  No  No  No  No  No  No  No  Yes No  No  No 
    ## [595] No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No  No 
    ## [613] No  No  No  No  No  No  No  No  No  No  Yes No  No  No  No  No  No  Yes
    ## Levels: No Yes

``` r
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         as.factor(Customer_Churns_test[, 1:14]$Churn))
```

### Sensitivity and specificity

We can see the sensitivity (≈ 0.86) and the specificity (≈ 0.60) below

``` r
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  No Yes
    ##        No  502  65
    ##        Yes  29  34
    ##                                           
    ##                Accuracy : 0.8508          
    ##                  95% CI : (0.8205, 0.8777)
    ##     No Information Rate : 0.8429          
    ##     P-Value [Acc > NIR] : 0.3145292       
    ##                                           
    ##                   Kappa : 0.339           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0003062       
    ##                                           
    ##             Sensitivity : 0.9454          
    ##             Specificity : 0.3434          
    ##          Pos Pred Value : 0.8854          
    ##          Neg Pred Value : 0.5397          
    ##              Prevalence : 0.8429          
    ##          Detection Rate : 0.7968          
    ##    Detection Prevalence : 0.9000          
    ##       Balanced Accuracy : 0.6444          
    ##                                           
    ##        'Positive' Class : No              
    ## 

### AUC

The type = “prob” argument specifies that you want to obtain class
probabilities as the output of the prediction instead of class labels.

``` r
predictions <- predict(churn_model_knn, Customer_Churns_test[, 1:13],
                       type = "prob")
```

### Class predictions

These are the class probability values for diabetes that the model has
predicted

``` r
print(predictions)
```

    ##             No        Yes
    ## 1   1.00000000 0.00000000
    ## 2   0.33333333 0.66666667
    ## 3   1.00000000 0.00000000
    ## 4   1.00000000 0.00000000
    ## 5   1.00000000 0.00000000
    ## 6   0.50000000 0.50000000
    ## 7   0.66666667 0.33333333
    ## 8   0.66666667 0.33333333
    ## 9   0.33333333 0.66666667
    ## 10  1.00000000 0.00000000
    ## 11  0.88888889 0.11111111
    ## 12  0.88888889 0.11111111
    ## 13  0.88888889 0.11111111
    ## 14  1.00000000 0.00000000
    ## 15  1.00000000 0.00000000
    ## 16  1.00000000 0.00000000
    ## 17  0.88888889 0.11111111
    ## 18  1.00000000 0.00000000
    ## 19  0.66666667 0.33333333
    ## 20  1.00000000 0.00000000
    ## 21  1.00000000 0.00000000
    ## 22  0.55555556 0.44444444
    ## 23  0.77777778 0.22222222
    ## 24  0.77777778 0.22222222
    ## 25  1.00000000 0.00000000
    ## 26  0.77777778 0.22222222
    ## 27  0.66666667 0.33333333
    ## 28  1.00000000 0.00000000
    ## 29  1.00000000 0.00000000
    ## 30  0.88888889 0.11111111
    ## 31  1.00000000 0.00000000
    ## 32  1.00000000 0.00000000
    ## 33  0.88888889 0.11111111
    ## 34  0.88888889 0.11111111
    ## 35  1.00000000 0.00000000
    ## 36  0.77777778 0.22222222
    ## 37  1.00000000 0.00000000
    ## 38  1.00000000 0.00000000
    ## 39  0.80000000 0.20000000
    ## 40  0.55555556 0.44444444
    ## 41  1.00000000 0.00000000
    ## 42  0.55555556 0.44444444
    ## 43  0.66666667 0.33333333
    ## 44  1.00000000 0.00000000
    ## 45  1.00000000 0.00000000
    ## 46  0.45454545 0.54545455
    ## 47  1.00000000 0.00000000
    ## 48  1.00000000 0.00000000
    ## 49  1.00000000 0.00000000
    ## 50  0.55555556 0.44444444
    ## 51  1.00000000 0.00000000
    ## 52  1.00000000 0.00000000
    ## 53  0.55555556 0.44444444
    ## 54  0.88888889 0.11111111
    ## 55  0.90000000 0.10000000
    ## 56  0.66666667 0.33333333
    ## 57  1.00000000 0.00000000
    ## 58  1.00000000 0.00000000
    ## 59  0.77777778 0.22222222
    ## 60  0.22222222 0.77777778
    ## 61  1.00000000 0.00000000
    ## 62  0.66666667 0.33333333
    ## 63  1.00000000 0.00000000
    ## 64  0.88888889 0.11111111
    ## 65  1.00000000 0.00000000
    ## 66  1.00000000 0.00000000
    ## 67  1.00000000 0.00000000
    ## 68  0.55555556 0.44444444
    ## 69  0.33333333 0.66666667
    ## 70  0.44444444 0.55555556
    ## 71  1.00000000 0.00000000
    ## 72  0.66666667 0.33333333
    ## 73  0.88888889 0.11111111
    ## 74  1.00000000 0.00000000
    ## 75  1.00000000 0.00000000
    ## 76  1.00000000 0.00000000
    ## 77  0.77777778 0.22222222
    ## 78  0.66666667 0.33333333
    ## 79  0.88888889 0.11111111
    ## 80  1.00000000 0.00000000
    ## 81  1.00000000 0.00000000
    ## 82  0.88888889 0.11111111
    ## 83  0.77777778 0.22222222
    ## 84  1.00000000 0.00000000
    ## 85  0.25000000 0.75000000
    ## 86  1.00000000 0.00000000
    ## 87  0.88888889 0.11111111
    ## 88  0.44444444 0.55555556
    ## 89  1.00000000 0.00000000
    ## 90  1.00000000 0.00000000
    ## 91  1.00000000 0.00000000
    ## 92  0.66666667 0.33333333
    ## 93  1.00000000 0.00000000
    ## 94  0.88888889 0.11111111
    ## 95  0.77777778 0.22222222
    ## 96  0.66666667 0.33333333
    ## 97  0.77777778 0.22222222
    ## 98  1.00000000 0.00000000
    ## 99  1.00000000 0.00000000
    ## 100 1.00000000 0.00000000
    ## 101 0.88888889 0.11111111
    ## 102 0.90000000 0.10000000
    ## 103 0.33333333 0.66666667
    ## 104 1.00000000 0.00000000
    ## 105 1.00000000 0.00000000
    ## 106 1.00000000 0.00000000
    ## 107 1.00000000 0.00000000
    ## 108 1.00000000 0.00000000
    ## 109 0.88888889 0.11111111
    ## 110 0.55555556 0.44444444
    ## 111 0.55555556 0.44444444
    ## 112 1.00000000 0.00000000
    ## 113 1.00000000 0.00000000
    ## 114 1.00000000 0.00000000
    ## 115 0.55555556 0.44444444
    ## 116 1.00000000 0.00000000
    ## 117 0.55555556 0.44444444
    ## 118 0.33333333 0.66666667
    ## 119 1.00000000 0.00000000
    ## 120 0.55555556 0.44444444
    ## 121 1.00000000 0.00000000
    ## 122 1.00000000 0.00000000
    ## 123 0.88888889 0.11111111
    ## 124 1.00000000 0.00000000
    ## 125 1.00000000 0.00000000
    ## 126 0.77777778 0.22222222
    ## 127 1.00000000 0.00000000
    ## 128 0.55555556 0.44444444
    ## 129 1.00000000 0.00000000
    ## 130 0.45454545 0.54545455
    ## 131 1.00000000 0.00000000
    ## 132 0.33333333 0.66666667
    ## 133 0.55555556 0.44444444
    ## 134 1.00000000 0.00000000
    ## 135 0.77777778 0.22222222
    ## 136 0.55555556 0.44444444
    ## 137 1.00000000 0.00000000
    ## 138 0.88888889 0.11111111
    ## 139 1.00000000 0.00000000
    ## 140 0.88888889 0.11111111
    ## 141 1.00000000 0.00000000
    ## 142 0.55555556 0.44444444
    ## 143 0.66666667 0.33333333
    ## 144 1.00000000 0.00000000
    ## 145 1.00000000 0.00000000
    ## 146 0.88888889 0.11111111
    ## 147 1.00000000 0.00000000
    ## 148 0.55555556 0.44444444
    ## 149 0.77777778 0.22222222
    ## 150 1.00000000 0.00000000
    ## 151 1.00000000 0.00000000
    ## 152 0.66666667 0.33333333
    ## 153 1.00000000 0.00000000
    ## 154 0.77777778 0.22222222
    ## 155 1.00000000 0.00000000
    ## 156 1.00000000 0.00000000
    ## 157 1.00000000 0.00000000
    ## 158 1.00000000 0.00000000
    ## 159 1.00000000 0.00000000
    ## 160 0.66666667 0.33333333
    ## 161 1.00000000 0.00000000
    ## 162 0.77777778 0.22222222
    ## 163 1.00000000 0.00000000
    ## 164 1.00000000 0.00000000
    ## 165 0.77777778 0.22222222
    ## 166 1.00000000 0.00000000
    ## 167 0.66666667 0.33333333
    ## 168 1.00000000 0.00000000
    ## 169 0.88888889 0.11111111
    ## 170 0.88888889 0.11111111
    ## 171 1.00000000 0.00000000
    ## 172 0.66666667 0.33333333
    ## 173 0.55555556 0.44444444
    ## 174 0.66666667 0.33333333
    ## 175 1.00000000 0.00000000
    ## 176 1.00000000 0.00000000
    ## 177 1.00000000 0.00000000
    ## 178 0.55555556 0.44444444
    ## 179 1.00000000 0.00000000
    ## 180 1.00000000 0.00000000
    ## 181 0.66666667 0.33333333
    ## 182 1.00000000 0.00000000
    ## 183 1.00000000 0.00000000
    ## 184 0.55555556 0.44444444
    ## 185 0.77777778 0.22222222
    ## 186 1.00000000 0.00000000
    ## 187 0.77777778 0.22222222
    ## 188 0.44444444 0.55555556
    ## 189 0.66666667 0.33333333
    ## 190 1.00000000 0.00000000
    ## 191 1.00000000 0.00000000
    ## 192 1.00000000 0.00000000
    ## 193 0.88888889 0.11111111
    ## 194 1.00000000 0.00000000
    ## 195 1.00000000 0.00000000
    ## 196 1.00000000 0.00000000
    ## 197 0.50000000 0.50000000
    ## 198 1.00000000 0.00000000
    ## 199 1.00000000 0.00000000
    ## 200 1.00000000 0.00000000
    ## 201 1.00000000 0.00000000
    ## 202 0.60000000 0.40000000
    ## 203 1.00000000 0.00000000
    ## 204 0.33333333 0.66666667
    ## 205 0.66666667 0.33333333
    ## 206 1.00000000 0.00000000
    ## 207 0.66666667 0.33333333
    ## 208 1.00000000 0.00000000
    ## 209 0.55555556 0.44444444
    ## 210 0.11111111 0.88888889
    ## 211 1.00000000 0.00000000
    ## 212 0.88888889 0.11111111
    ## 213 1.00000000 0.00000000
    ## 214 1.00000000 0.00000000
    ## 215 0.88888889 0.11111111
    ## 216 0.55555556 0.44444444
    ## 217 0.88888889 0.11111111
    ## 218 1.00000000 0.00000000
    ## 219 0.66666667 0.33333333
    ## 220 1.00000000 0.00000000
    ## 221 1.00000000 0.00000000
    ## 222 1.00000000 0.00000000
    ## 223 0.88888889 0.11111111
    ## 224 0.09090909 0.90909091
    ## 225 1.00000000 0.00000000
    ## 226 0.77777778 0.22222222
    ## 227 0.45454545 0.54545455
    ## 228 0.88888889 0.11111111
    ## 229 1.00000000 0.00000000
    ## 230 0.77777778 0.22222222
    ## 231 0.88888889 0.11111111
    ## 232 1.00000000 0.00000000
    ## 233 0.66666667 0.33333333
    ## 234 1.00000000 0.00000000
    ## 235 0.66666667 0.33333333
    ## 236 0.77777778 0.22222222
    ## 237 0.88888889 0.11111111
    ## 238 1.00000000 0.00000000
    ## 239 1.00000000 0.00000000
    ## 240 0.77777778 0.22222222
    ## 241 0.88888889 0.11111111
    ## 242 1.00000000 0.00000000
    ## 243 0.41666667 0.58333333
    ## 244 1.00000000 0.00000000
    ## 245 0.44444444 0.55555556
    ## 246 0.55555556 0.44444444
    ## 247 1.00000000 0.00000000
    ## 248 0.88888889 0.11111111
    ## 249 0.50000000 0.50000000
    ## 250 1.00000000 0.00000000
    ## 251 1.00000000 0.00000000
    ## 252 0.88888889 0.11111111
    ## 253 1.00000000 0.00000000
    ## 254 1.00000000 0.00000000
    ## 255 0.66666667 0.33333333
    ## 256 0.44444444 0.55555556
    ## 257 1.00000000 0.00000000
    ## 258 0.90000000 0.10000000
    ## 259 0.66666667 0.33333333
    ## 260 1.00000000 0.00000000
    ## 261 0.88888889 0.11111111
    ## 262 0.66666667 0.33333333
    ## 263 0.22222222 0.77777778
    ## 264 0.77777778 0.22222222
    ## 265 0.90909091 0.09090909
    ## 266 0.55555556 0.44444444
    ## 267 0.88888889 0.11111111
    ## 268 1.00000000 0.00000000
    ## 269 0.66666667 0.33333333
    ## 270 1.00000000 0.00000000
    ## 271 1.00000000 0.00000000
    ## 272 1.00000000 0.00000000
    ## 273 0.33333333 0.66666667
    ## 274 0.33333333 0.66666667
    ## 275 0.66666667 0.33333333
    ## 276 1.00000000 0.00000000
    ## 277 1.00000000 0.00000000
    ## 278 1.00000000 0.00000000
    ## 279 0.77777778 0.22222222
    ## 280 0.88888889 0.11111111
    ## 281 1.00000000 0.00000000
    ## 282 1.00000000 0.00000000
    ## 283 1.00000000 0.00000000
    ## 284 0.44444444 0.55555556
    ## 285 1.00000000 0.00000000
    ## 286 1.00000000 0.00000000
    ## 287 0.77777778 0.22222222
    ## 288 0.11111111 0.88888889
    ## 289 0.77777778 0.22222222
    ## 290 0.88888889 0.11111111
    ## 291 1.00000000 0.00000000
    ## 292 1.00000000 0.00000000
    ## 293 1.00000000 0.00000000
    ## 294 1.00000000 0.00000000
    ## 295 0.44444444 0.55555556
    ## 296 1.00000000 0.00000000
    ## 297 1.00000000 0.00000000
    ## 298 1.00000000 0.00000000
    ## 299 0.77777778 0.22222222
    ## 300 1.00000000 0.00000000
    ## 301 1.00000000 0.00000000
    ## 302 1.00000000 0.00000000
    ## 303 1.00000000 0.00000000
    ## 304 0.66666667 0.33333333
    ## 305 0.45454545 0.54545455
    ## 306 1.00000000 0.00000000
    ## 307 1.00000000 0.00000000
    ## 308 1.00000000 0.00000000
    ## 309 1.00000000 0.00000000
    ## 310 0.88888889 0.11111111
    ## 311 0.54545455 0.45454545
    ## 312 1.00000000 0.00000000
    ## 313 1.00000000 0.00000000
    ## 314 1.00000000 0.00000000
    ## 315 0.77777778 0.22222222
    ## 316 0.66666667 0.33333333
    ## 317 0.88888889 0.11111111
    ## 318 0.55555556 0.44444444
    ## 319 0.70000000 0.30000000
    ## 320 1.00000000 0.00000000
    ## 321 1.00000000 0.00000000
    ## 322 0.88888889 0.11111111
    ## 323 1.00000000 0.00000000
    ## 324 1.00000000 0.00000000
    ## 325 0.25000000 0.75000000
    ## 326 1.00000000 0.00000000
    ## 327 1.00000000 0.00000000
    ## 328 0.77777778 0.22222222
    ## 329 0.55555556 0.44444444
    ## 330 1.00000000 0.00000000
    ## 331 0.33333333 0.66666667
    ## 332 0.88888889 0.11111111
    ## 333 0.66666667 0.33333333
    ## 334 1.00000000 0.00000000
    ## 335 0.08333333 0.91666667
    ## 336 0.66666667 0.33333333
    ## 337 1.00000000 0.00000000
    ## 338 0.77777778 0.22222222
    ## 339 0.88888889 0.11111111
    ## 340 1.00000000 0.00000000
    ## 341 1.00000000 0.00000000
    ## 342 1.00000000 0.00000000
    ## 343 1.00000000 0.00000000
    ## 344 1.00000000 0.00000000
    ## 345 1.00000000 0.00000000
    ## 346 1.00000000 0.00000000
    ## 347 0.88888889 0.11111111
    ## 348 1.00000000 0.00000000
    ## 349 1.00000000 0.00000000
    ## 350 0.90000000 0.10000000
    ## 351 0.66666667 0.33333333
    ## 352 1.00000000 0.00000000
    ## 353 0.55555556 0.44444444
    ## 354 0.20000000 0.80000000
    ## 355 0.88888889 0.11111111
    ## 356 0.66666667 0.33333333
    ## 357 0.88888889 0.11111111
    ## 358 0.44444444 0.55555556
    ## 359 1.00000000 0.00000000
    ## 360 1.00000000 0.00000000
    ## 361 1.00000000 0.00000000
    ## 362 0.88888889 0.11111111
    ## 363 0.88888889 0.11111111
    ## 364 0.66666667 0.33333333
    ## 365 1.00000000 0.00000000
    ## 366 1.00000000 0.00000000
    ## 367 1.00000000 0.00000000
    ## 368 1.00000000 0.00000000
    ## 369 0.60000000 0.40000000
    ## 370 0.77777778 0.22222222
    ## 371 1.00000000 0.00000000
    ## 372 0.77777778 0.22222222
    ## 373 0.77777778 0.22222222
    ## 374 1.00000000 0.00000000
    ## 375 1.00000000 0.00000000
    ## 376 1.00000000 0.00000000
    ## 377 0.66666667 0.33333333
    ## 378 0.77777778 0.22222222
    ## 379 1.00000000 0.00000000
    ## 380 1.00000000 0.00000000
    ## 381 1.00000000 0.00000000
    ## 382 1.00000000 0.00000000
    ## 383 1.00000000 0.00000000
    ## 384 0.77777778 0.22222222
    ## 385 1.00000000 0.00000000
    ## 386 0.88888889 0.11111111
    ## 387 0.25000000 0.75000000
    ## 388 0.45454545 0.54545455
    ## 389 0.25000000 0.75000000
    ## 390 1.00000000 0.00000000
    ## 391 0.22222222 0.77777778
    ## 392 0.66666667 0.33333333
    ## 393 1.00000000 0.00000000
    ## 394 0.38461538 0.61538462
    ## 395 0.88888889 0.11111111
    ## 396 1.00000000 0.00000000
    ## 397 0.45454545 0.54545455
    ## 398 0.88888889 0.11111111
    ## 399 0.66666667 0.33333333
    ## 400 1.00000000 0.00000000
    ## 401 0.88888889 0.11111111
    ## 402 1.00000000 0.00000000
    ## 403 1.00000000 0.00000000
    ## 404 0.66666667 0.33333333
    ## 405 0.77777778 0.22222222
    ## 406 1.00000000 0.00000000
    ## 407 0.45454545 0.54545455
    ## 408 0.11111111 0.88888889
    ## 409 0.11111111 0.88888889
    ## 410 0.55555556 0.44444444
    ## 411 0.66666667 0.33333333
    ## 412 1.00000000 0.00000000
    ## 413 0.88888889 0.11111111
    ## 414 0.88888889 0.11111111
    ## 415 0.88888889 0.11111111
    ## 416 0.66666667 0.33333333
    ## 417 1.00000000 0.00000000
    ## 418 1.00000000 0.00000000
    ## 419 1.00000000 0.00000000
    ## 420 1.00000000 0.00000000
    ## 421 1.00000000 0.00000000
    ## 422 0.66666667 0.33333333
    ## 423 0.55555556 0.44444444
    ## 424 0.77777778 0.22222222
    ## 425 0.77777778 0.22222222
    ## 426 1.00000000 0.00000000
    ## 427 1.00000000 0.00000000
    ## 428 1.00000000 0.00000000
    ## 429 1.00000000 0.00000000
    ## 430 0.77777778 0.22222222
    ## 431 1.00000000 0.00000000
    ## 432 1.00000000 0.00000000
    ## 433 0.66666667 0.33333333
    ## 434 1.00000000 0.00000000
    ## 435 1.00000000 0.00000000
    ## 436 0.77777778 0.22222222
    ## 437 1.00000000 0.00000000
    ## 438 1.00000000 0.00000000
    ## 439 0.88888889 0.11111111
    ## 440 1.00000000 0.00000000
    ## 441 1.00000000 0.00000000
    ## 442 0.77777778 0.22222222
    ## 443 1.00000000 0.00000000
    ## 444 0.55555556 0.44444444
    ## 445 0.33333333 0.66666667
    ## 446 1.00000000 0.00000000
    ## 447 1.00000000 0.00000000
    ## 448 1.00000000 0.00000000
    ## 449 0.77777778 0.22222222
    ## 450 1.00000000 0.00000000
    ## 451 1.00000000 0.00000000
    ## 452 1.00000000 0.00000000
    ## 453 0.77777778 0.22222222
    ## 454 0.88888889 0.11111111
    ## 455 0.77777778 0.22222222
    ## 456 1.00000000 0.00000000
    ## 457 0.55555556 0.44444444
    ## 458 0.09090909 0.90909091
    ## 459 1.00000000 0.00000000
    ## 460 1.00000000 0.00000000
    ## 461 0.88888889 0.11111111
    ## 462 1.00000000 0.00000000
    ## 463 0.77777778 0.22222222
    ## 464 0.11111111 0.88888889
    ## 465 0.88888889 0.11111111
    ## 466 0.44444444 0.55555556
    ## 467 0.33333333 0.66666667
    ## 468 0.66666667 0.33333333
    ## 469 0.44444444 0.55555556
    ## 470 1.00000000 0.00000000
    ## 471 1.00000000 0.00000000
    ## 472 1.00000000 0.00000000
    ## 473 1.00000000 0.00000000
    ## 474 1.00000000 0.00000000
    ## 475 1.00000000 0.00000000
    ## 476 1.00000000 0.00000000
    ## 477 1.00000000 0.00000000
    ## 478 1.00000000 0.00000000
    ## 479 1.00000000 0.00000000
    ## 480 0.77777778 0.22222222
    ## 481 0.11111111 0.88888889
    ## 482 1.00000000 0.00000000
    ## 483 1.00000000 0.00000000
    ## 484 0.55555556 0.44444444
    ## 485 0.66666667 0.33333333
    ## 486 0.11111111 0.88888889
    ## 487 1.00000000 0.00000000
    ## 488 0.80000000 0.20000000
    ## 489 0.70000000 0.30000000
    ## 490 0.44444444 0.55555556
    ## 491 0.55555556 0.44444444
    ## 492 1.00000000 0.00000000
    ## 493 1.00000000 0.00000000
    ## 494 1.00000000 0.00000000
    ## 495 1.00000000 0.00000000
    ## 496 1.00000000 0.00000000
    ## 497 1.00000000 0.00000000
    ## 498 0.88888889 0.11111111
    ## 499 1.00000000 0.00000000
    ## 500 1.00000000 0.00000000
    ## 501 0.88888889 0.11111111
    ## 502 0.77777778 0.22222222
    ## 503 1.00000000 0.00000000
    ## 504 1.00000000 0.00000000
    ## 505 0.88888889 0.11111111
    ## 506 0.66666667 0.33333333
    ## 507 1.00000000 0.00000000
    ## 508 0.11111111 0.88888889
    ## 509 1.00000000 0.00000000
    ## 510 1.00000000 0.00000000
    ## 511 1.00000000 0.00000000
    ## 512 0.88888889 0.11111111
    ## 513 0.11111111 0.88888889
    ## 514 0.44444444 0.55555556
    ## 515 0.33333333 0.66666667
    ## 516 0.88888889 0.11111111
    ## 517 1.00000000 0.00000000
    ## 518 0.88888889 0.11111111
    ## 519 0.77777778 0.22222222
    ## 520 1.00000000 0.00000000
    ## 521 1.00000000 0.00000000
    ## 522 0.77777778 0.22222222
    ## 523 1.00000000 0.00000000
    ## 524 0.77777778 0.22222222
    ## 525 1.00000000 0.00000000
    ## 526 0.11111111 0.88888889
    ## 527 1.00000000 0.00000000
    ## 528 0.88888889 0.11111111
    ## 529 1.00000000 0.00000000
    ## 530 1.00000000 0.00000000
    ## 531 0.77777778 0.22222222
    ## 532 0.55555556 0.44444444
    ## 533 0.77777778 0.22222222
    ## 534 0.55555556 0.44444444
    ## 535 1.00000000 0.00000000
    ## 536 0.44444444 0.55555556
    ## 537 1.00000000 0.00000000
    ## 538 1.00000000 0.00000000
    ## 539 0.77777778 0.22222222
    ## 540 1.00000000 0.00000000
    ## 541 1.00000000 0.00000000
    ## 542 0.66666667 0.33333333
    ## 543 0.88888889 0.11111111
    ## 544 1.00000000 0.00000000
    ## 545 1.00000000 0.00000000
    ## 546 1.00000000 0.00000000
    ## 547 0.88888889 0.11111111
    ## 548 0.88888889 0.11111111
    ## 549 1.00000000 0.00000000
    ## 550 0.77777778 0.22222222
    ## 551 1.00000000 0.00000000
    ## 552 0.90000000 0.10000000
    ## 553 0.88888889 0.11111111
    ## 554 0.88888889 0.11111111
    ## 555 1.00000000 0.00000000
    ## 556 0.66666667 0.33333333
    ## 557 0.66666667 0.33333333
    ## 558 0.66666667 0.33333333
    ## 559 1.00000000 0.00000000
    ## 560 1.00000000 0.00000000
    ## 561 0.88888889 0.11111111
    ## 562 0.77777778 0.22222222
    ## 563 0.66666667 0.33333333
    ## 564 1.00000000 0.00000000
    ## 565 1.00000000 0.00000000
    ## 566 0.88888889 0.11111111
    ## 567 1.00000000 0.00000000
    ## 568 1.00000000 0.00000000
    ## 569 0.81818182 0.18181818
    ## 570 1.00000000 0.00000000
    ## 571 1.00000000 0.00000000
    ## 572 0.66666667 0.33333333
    ## 573 0.77777778 0.22222222
    ## 574 1.00000000 0.00000000
    ## 575 0.88888889 0.11111111
    ## 576 1.00000000 0.00000000
    ## 577 1.00000000 0.00000000
    ## 578 1.00000000 0.00000000
    ## 579 1.00000000 0.00000000
    ## 580 0.44444444 0.55555556
    ## 581 1.00000000 0.00000000
    ## 582 1.00000000 0.00000000
    ## 583 1.00000000 0.00000000
    ## 584 0.77777778 0.22222222
    ## 585 1.00000000 0.00000000
    ## 586 0.88888889 0.11111111
    ## 587 0.77777778 0.22222222
    ## 588 1.00000000 0.00000000
    ## 589 1.00000000 0.00000000
    ## 590 1.00000000 0.00000000
    ## 591 0.33333333 0.66666667
    ## 592 0.88888889 0.11111111
    ## 593 1.00000000 0.00000000
    ## 594 1.00000000 0.00000000
    ## 595 0.88888889 0.11111111
    ## 596 0.66666667 0.33333333
    ## 597 0.55555556 0.44444444
    ## 598 1.00000000 0.00000000
    ## 599 1.00000000 0.00000000
    ## 600 0.88888889 0.11111111
    ## 601 1.00000000 0.00000000
    ## 602 0.77777778 0.22222222
    ## 603 1.00000000 0.00000000
    ## 604 0.55555556 0.44444444
    ## 605 1.00000000 0.00000000
    ## 606 1.00000000 0.00000000
    ## 607 1.00000000 0.00000000
    ## 608 1.00000000 0.00000000
    ## 609 0.88888889 0.11111111
    ## 610 1.00000000 0.00000000
    ## 611 0.88888889 0.11111111
    ## 612 0.66666667 0.33333333
    ## 613 0.55555556 0.44444444
    ## 614 0.55555556 0.44444444
    ## 615 0.77777778 0.22222222
    ## 616 1.00000000 0.00000000
    ## 617 1.00000000 0.00000000
    ## 618 1.00000000 0.00000000
    ## 619 0.88888889 0.11111111
    ## 620 1.00000000 0.00000000
    ## 621 1.00000000 0.00000000
    ## 622 0.88888889 0.11111111
    ## 623 0.11111111 0.88888889
    ## 624 1.00000000 0.00000000
    ## 625 1.00000000 0.00000000
    ## 626 1.00000000 0.00000000
    ## 627 1.00000000 0.00000000
    ## 628 0.77777778 0.22222222
    ## 629 1.00000000 0.00000000
    ## 630 0.44444444 0.55555556

### Setting the Direction

``` r
roc_curve <- roc(Customer_Churns_test$Churn, predictions$No)
```

    ## Setting levels: control = No, case = Yes

    ## Setting direction: controls > cases

### Plot the ROC curve

``` r
plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)
```

![](Lab-Submission-Markdown_files/figure-gfm/Yourthirtififth%20Code%20Chunk-1.png)<!-- -->

# STEP 4. Logarithmic Loss (LogLoss)

## 4.a. Load the dataset

``` r
library(readr)
Crop_recommendation <- read_csv("../data/Crop_recommendation.csv")
```

    ## Rows: 2200 Columns: 8
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (1): label
    ## dbl (7): N, P, K, temperature, humidity, ph, rainfall
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(Crop_recommendation)
```

## 4.b. Train the Model

We apply the 5-fold repeated cross validation resampling method with 3
repeats

``` r
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)
```

### Create a CART

One of the parameters used by a CART model is “cp”. “cp” refers to the
“complexity parameter”. It is used to impose a penalty to the tree for
having too many splits. The default value is 0.01.

``` r
crop_model_cart <- train(label ~ ., data = Crop_recommendation, method = "rpart",
                         metric = "logLoss", trControl = train_control)
```

## 4.c. Display the Model’s Performance

### Use the metric calculated by caret when training the model

The results show that a cp value of ≈ 0 resulted in the lowest LogLoss
value. The lowest logLoss value is ≈ 0.46.

``` r
print(crop_model_cart)
```

    ## CART 
    ## 
    ## 2200 samples
    ##    7 predictor
    ##   22 classes: 'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 1760, 1760, 1760, 1760, 1760, 1760, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          logLoss  
    ##   0.03761905  0.2458116
    ##   0.04214286  0.2722337
    ##   0.04761905  0.5315899
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.03761905.
