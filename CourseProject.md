# Practical Machine Learning - Course Project
Jerry Harber  
August 30, 2017  



#### Executive Summary  
  
Using a random forest prediction model, the following predictors were used to predict "classe";  

* avg_roll_belt
* avg_pitch_belt
* avg_yaw_belt
* avg_roll_arm
* avg_pitch_arm
* avg_yaw_arm
* avg_roll_dumbbell
* avg_pitch_dumbbell
* avg_yaw_dumbbell
* avg_roll_forearm
* avg_pitch_forearm
* avg_yaw_forearm

These predictors were based on the the following paper;  

* Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


The data used in this analysis was based on this paper. The authors also used summary statistics in order to create their model.


#### Analysis Results  
  
Analysis results indicated the random forest classification model used to predict "classe" was somewhat moderately significant. Using all the predictors in the data set (i.e., 160 variables) was problematic for a random forest model due to computer memory constraints. Reducing the number of predictors to 12 and using only observations that had a value for the predictor, significantly reduced the amount of computer memory. See the final results in each section below.  
  
#### My setup

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```
  
#### Download and load data
  
Download the training and the testing data. Read the data sets into a data.frame

```r
url <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url,"training.csv")
training_raw <- read.csv("training.csv")
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url,"testing.csv")
testCases_raw <- read.csv("testing.csv")
```
  
#### Create a "tidy"  training and testing data set

```r
## Create a training data set based on the training_raw data set.
p_training <- 0.90

set.seed(3383)
inTrain <- createDataPartition(y=training_raw$classe, p=p_training, list=FALSE)

## Use only averages via the style of the paper
varsToUse <-c("avg_roll_belt","avg_pitch_belt","avg_yaw_belt","avg_roll_arm","avg_pitch_arm","avg_yaw_arm", "avg_roll_dumbbell", "avg_pitch_dumbbell", "avg_yaw_dumbbell", "avg_roll_forearm", "avg_pitch_forearm", "avg_yaw_forearm", "classe")

new_names <-c("roll_belt", "pitch_belt", "yaw_belt", "roll_arm", "pitch_arm", "yaw_arm", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "roll_forearm", "pitch_forearm", "yaw_forearm", "classe")

## Create a training data set based on the training_raw data set.
## Use only averages via the style of the paper. Remove any rows that has NA for avg_roll_belt.
## Remove any rows that has NA for avg_roll_belt. Change variable names to be compatible with testCases
training <- training_raw[inTrain,]
l <- is.na(training$avg_roll_belt)
tidy_training <- training[!l,varsToUse]
colnames(tidy_training) <- new_names
nrow(tidy_training)
```

```
## [1] 363
```

```r
ncol(tidy_training)
```

```
## [1] 13
```

```r
## Create a testing data set based on the training_raw data set.
## Use only averages via the style of the paper. Remove any rows that has NA for avg_roll_belt.
testing <- training_raw[-inTrain,]
l <- is.na(testing$avg_roll_belt)
tidy_testing <- testing[!l,varsToUse]
colnames(tidy_testing) <- new_names
nrow(tidy_testing)
```

```
## [1] 43
```

```r
ncol(tidy_testing)
```

```
## [1] 13
```
  
#### Build a Random Forest model using the tidy_training data set.


```r
modFit_rf <- train(classe ~ . , data=tidy_training, method="rf", prox = TRUE)
modFit_rf
```

```
## Random Forest 
## 
## 363 samples
##  12 predictor
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 363, 363, 363, 363, 363, 363, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.7657515  0.7040410
##    7    0.7468202  0.6802643
##   12    0.7324304  0.6625806
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
modFit_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 18.46%
## Confusion matrix:
##    A  B  C  D  E class.error
## A 89  4  3  2  0  0.09183673
## B 10 44  6  4  3  0.34328358
## C  2  4 54  1  1  0.12903226
## D  1  2  5 50  3  0.18032787
## E  1  9  4  2 59  0.21333333
```

```r
a <- modFit_rf$finalModel$confusion
n <- sum(a[,1:5])
d <- sum(diag(a[,1:5]))
myErr_rate <- round( (1 - (d/n)) * 100,2)
```
Results indicated that for my chosen predictors, the best fit model had an accuracy of 0.7657515 while the confusion matrix had an OOB (Out of Bag) error rate of 18.46 % (number misclassified divided by the total). 

#### Validate the model.

The model created with the tidy_training data set was used to predict "classe" on 2 subsets of the tidy_training data set. Data sets val1 and val2 are approximately 75% the size of the tidy_training data set.


```r
## Validation
##new_names <-c("roll_belt", "pitch_belt", "yaw_belt", "roll_arm", "pitch_arm", "yaw_arm", "roll_dumbbell", ##pitch_dumbbell", "yaw_dumbbell", "roll_forearm", "pitch_forearm", "yaw_forearm", "classe")
{
    print("-------------- Model validation 1 ----------------")
    set.seed(3383)
    p_val_training <- 0.75
    inTrain <- createDataPartition(y=tidy_training$classe, p=p_val_training, list=FALSE)
    val_training <- tidy_training[inTrain,]
    val_testing <- tidy_training[-inTrain,]

    modFit_rf_val1 <- train(classe ~ roll_belt+pitch_belt+yaw_belt+roll_arm+pitch_arm+yaw_arm, data=val_training,
                      method="rf", prox = TRUE)
    print(modFit_rf_val1)
    print(modFit_rf_val1$finalModel)
    
    predVal_Test1 <- predict(modFit_rf_val1, val_testing)
    tbl1 <- table(predVal_Test1, val_testing$classe)
    d <- sum(diag(tbl1))
    n <- sum(tbl1)
    myErr_rate_val1 <- round( (1 - (d/n)) * 100,2)
    print("Table of predition versus testing validation data")
    print(paste("My out of sample error rate estimate = ", myErr_rate_val1))
}
```

```
## [1] "-------------- Model validation 1 ----------------"
## Random Forest 
## 
## 275 samples
##   6 predictor
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 275, 275, 275, 275, 275, 275, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.6101375  0.5070128
##   4     0.6039715  0.5003773
##   6     0.5975600  0.4927171
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 32.36%
## Confusion matrix:
##    A  B  C  D  E class.error
## A 54  6  7  5  2   0.2702703
## B 11 35  3  0  2   0.3137255
## C  9 10 24  3  1   0.4893617
## D  7  1  5 30  3   0.3478261
## E  7  1  0  6 43   0.2456140
## [1] "Table of predition versus testing validation data"
## [1] "My out of sample error rate estimate =  22.73"
```

```r
{
    print("-------------- Model validation 2 ----------------")
    set.seed(3383)
    p_val_training <- 0.75
    inTrain <- createDataPartition(y=tidy_training$classe, p=p_val_training, list=FALSE)
    val_training <- tidy_training[inTrain,]
    val_testing <- tidy_training[-inTrain,]

    modFit_rf_val2 <- train(classe ~ ., data=val_training,method="rf", prox = TRUE)
    print(modFit_rf_val2)
    print(modFit_rf_val2$finalModel)
    
    predVal_Test2 <- predict(modFit_rf_val2, val_testing)
    tbl2 <- table(predVal_Test2, val_testing$classe)
    d <- sum(diag(tbl2))
    n <- sum(tbl2)
    myErr_rate_val2 <- round( (1 - (d/n)) * 100,2)
    print("Table of predition versus testing validation data")
    print(paste("My out of sample error rate estimate = ", myErr_rate_val2))
}
```

```
## [1] "-------------- Model validation 2 ----------------"
## Random Forest 
## 
## 275 samples
##  12 predictor
##   5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 275, 275, 275, 275, 275, 275, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.7166816  0.6430463
##    7    0.6924176  0.6124350
##   12    0.6730555  0.5881296
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 22.55%
## Confusion matrix:
##    A  B  C  D  E class.error
## A 67  3  2  2  0  0.09459459
## B  8 25  7  7  4  0.50980392
## C  1  5 39  1  1  0.17021277
## D  1  2  3 38  2  0.17391304
## E  3  4  3  3 44  0.22807018
## [1] "Table of predition versus testing validation data"
## [1] "My out of sample error rate estimate =  18.18"
```

#### Predict test cases"classe"" for the test cases data set using the Random Forest model

```r
set.seed(3383)
predTestCases <- predict(modFit_rf, testCases_raw)
predTestCases
```

```
##  [1] C A B A A E D D A A B C B A C E A D A B
## Levels: A B C D E
```
  
#### Determine "out of sample" error

```r
set.seed(3383)
predTest <- predict(modFit_rf, testing)

tbl <- table(predTest, testing$classe)
d <- sum(diag(tbl))
n <- sum(tbl)
myErr_rate <- round( (1 - (d/n)) * 100,2)
myErr_rate
```

```
## [1] 21.33
```
  
The out of sample error rate is 21.33 %.
