# Acadgild-Dataanalytics-session17-assignment
DATA ANALYTICS WITH R, EXCEL AND TABLEAU SESSION 17ASSIGNMENT 

 
Session 17 Assignment
Weight Lifting Exercise



 
Session 17 Assignment
Weight Lifting Exercise



2. Perform the below given activities: 
a. Create classification model using logistic regression model 
b. verify model goodness of fit 
c. Report the accuracy measures 
d. Report the variable importance 
e. Report the unimportant variables 
f. Interpret the results 
g. Visualize the results
setwd("C:/Users/Seshan/Desktop")
library(readr)
Weight_lift <- read.csv("Weight lift.csv")
View(Weight_lift)
 str(Weight_lift)
data<-Weight_lift
# load libraries
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lattice)
library(rattle)

library(C50)
#install.package('devtools') # Only needed if you dont have this installed.
library(devtools)
install_github('adam-m-mcelhinney/helpRFunctions')
library(helpRFunctions)
names(data)
dim(data)
pairs(data[1:10])
# enable multi-core processing
library(doParallel)
#cl <- makeCluster(detectCores())
registerDoParallel()
set.seed(12345)
dataTrain<-data[1:3020,]
dataTest<-data[3021:4024,]
head(dataTrain)
head(dataTest)
indexNA <- as.vector(sapply(dataTrain[,1:152],function(x) {length(which(is.na(x)))!=0}))
dataTrain <- dataTrain[,!indexNA]
train_control<- trainControl(method="cv", number=10)

model<- train( pitch_belt ~., data=dataTrain,trControl=train_control, method="rf")
model
# make predictions
predictions<- predict(model,dataTrain)
# append predictions
pred<- cbind(dataTrain,predictions)
# summarize results
confusionMatrix<- confusionMatrix(pred$predictions,pred$pitch_belt)
confusionMatrix

summary(data)
summary(validation)
dim(data)
dim(validation)
#Remove unnecessary columns
# first 7 columns don't contain useful info
data <- data[,-seq(1:7)]
validation <- validation[,-seq(1:7)]

#Remove columns with NAs This reduces de amount of predictors to 53
# select columns that don't have NAs
indexNA <- as.vector(sapply(data[,1:152],function(x) {length(which(is.na(x)))!=0}))
data <- data[,!indexNA]
validation <- validation[,!indexNA]
# set last (classe) and prior (- classe) column index
#last <- as.numeric(ncol(data))
#prior <- last - 1

# set variables to numerics for correlation check, except the "classe"
for (i in 1:prior) {
  data[,i] <- as.numeric(data[,i])
  validation[,i] <- as.numeric(validation[,i])
}

#check the correlations
cor.check <- cor(data[, -c(last)])
diag(cor.check) <- 0 
plot( levelplot(cor.check, main ="Correlation matrix for all WLE features in training set",
                scales=list(x=list(rot=90), cex=1.0)))
# find the highly correlated variables
highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)
# pre process variables
preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
dataPrep <- predict(preObj, data[,1:prior])
dataPrep$classe <- data$classe

valPrep <-predict(preObj,validation[,1:prior])
valPrep$problem_id <- validation$problem_id
# test near zero variance
myDataNZV <- nearZeroVar(dataPrep, saveMetrics=TRUE)
if (any(myDataNZV$nzv)) nzv else message("No variables with near zero variance")
dataPrep <- dataPrep[,myDataNZV$nzv==FALSE]
valPrep <- valPrep[,myDataNZV$nzv==FALSE]
# split dataset into training and test set
inTrain <- createDataPartition(y=dataPrep$classe, p=0.7, list=FALSE )
training <- dataPrep[inTrain,]
testing <- dataPrep[-inTrain,]
# set seed for reproducibility
set.seed(12345)

# get the best mtry
bestmtry <- tuneRF(training[-last],training$classe, ntreeTry=100, 
                   stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)

mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]

# Model 1: RandomForest
wle.rf <-randomForest(classe~.,data=training, mtry=mtry, ntree=501, 
                      keep.forest=TRUE, proximity=TRUE, 
                      importance=TRUE,test=testing)
# plot the Out of bag error estimates
layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(wle.rf, log="y", main ="Out-of-bag (OOB) error estimate per Number of Trees")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(wle.rf$err.rate),col=1:6,cex=0.8,fill=1:6)
# plot the accuracy and Gini
varImpPlot(wle.rf, main="Mean Decrease of Accuracy and Gini per variable")
# MDSplot (we couldn't execute this due to lack of memory)
MDSplot(wle.rf, training$classe)
# results with training set
predict1 <- predict(wle.rf, newdata=training)
confusionMatrix(predict1,training$classe)
#Confusion Matrix and Statistics
# results with test set
predict2 <- predict(wle.rf, newdata=testing)
confusionMatrix(predict2,testing$classe)

# Confusion Matrix and Statistics
#Train Model 2: Decision Tree
# Model 2: Decision Tree
dt <- rpart(classe ~ ., data=training, method="class")

# fancyRpartPlot works for small trees, but not for ours
fancyRpartPlot(dt)

> setwd("C:/Users/Seshan/Desktop")
> library(readr)
> Weight_lift <- read.csv("Weight lift.csv")
> View(Weight_lift)
> str(Weight_lift)
'data.frame':	4024 obs. of  152 variables:
 $ user_name               : Factor w/ 5 levels "adelmo","carlitos",..: 3 3 3 3 3 3 3 3 3 3 ...
 $ raw_timestamp_part_1    : int  1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 ...
 $ raw_timestamp_part_2    : int  34670 62641 70653 82654 90637 170626 190665 242723 267551 274689 ...
 $ cvtd_timestamp          : Factor w/ 7 levels "2/12/2011 13:35",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
 $ num_window              : int  1 1 1 1 1 1 1 1 1 1 ...
 $ roll_belt               : num  3.7 3.66 3.58 3.56 3.57 3.45 3.31 2.91 2.31 2 ...
 $ pitch_belt              : num  41.6 42.8 43.7 44.4 45.1 45.6 46.2 46.9 47.4 47.7 ...
 $ yaw_belt                : num  -82.8 -82.5 -82.3 -82.1 -81.9 -81.9 -81.9 -82.2 -82.6 -82.8 ...
 $ total_accel_belt        : int  3 2 1 1 1 1 3 4 2 3 ...
 $ kurtosis_roll_belt      : num  -1.04 -1.04 -1.04 -1.04 -1.04 ...
 $ kurtosis_picth_belt     : num  -0.391 -0.391 -0.391 -0.391 -0.391 ...
 $ skewness_roll_belt      : num  0.00541 0.00541 0.00541 0.00541 0.00541 ...
 $ skewness_roll_belt.1    : num  0.0451 0.0451 0.0451 0.0451 0.0451 ...
 $ max_roll_belt           : num  -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 ...
 $ max_picth_belt          : int  20 20 20 20 20 20 20 20 20 20 ...
 $ max_yaw_belt            : num  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
 $ min_roll_belt           : num  -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 ...
 $ min_pitch_belt          : int  18 18 18 18 18 18 18 18 18 18 ...
 $ min_yaw_belt            : num  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
 $ amplitude_roll_belt     : num  1.34 1.34 1.34 1.34 1.34 ...
 $ amplitude_pitch_belt    : int  2 2 2 2 2 2 2 2 2 2 ...
 $ amplitude_yaw_belt      : int  0 0 0 0 0 0 0 0 0 0 ...
 $ var_total_accel_belt    : num  0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 ...
 $ avg_roll_belt           : num  122 122 122 122 122 ...
 $ stddev_roll_belt        : num  0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 ...
 $ var_roll_belt           : num  0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 ...
 $ avg_pitch_belt          : num  25.8 25.8 25.8 25.8 25.8 ...
 $ stddev_pitch_belt       : num  0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 ...
 $ var_pitch_belt          : num  0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 ...
 $ avg_yaw_belt            : num  -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 ...
 $ stddev_yaw_belt         : num  0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 ...
 $ var_yaw_belt            : num  0.17 0.17 0.17 0.17 0.17 0.17 0.17 0.17 0.17 0.17 ...
 $ gyros_belt_x            : num  2.02 1.96 1.88 1.8 1.77 1.75 1.78 1.75 1.65 1.48 ...
 $ gyros_belt_y            : num  0.18 0.14 0.08 0.03 0 -0.03 -0.06 -0.06 -0.03 -0.06 ...
 $ gyros_belt_z            : num  0.02 0.05 0.05 0.08 0.13 0.16 0.15 0.23 0.33 0.21 ...
 $ accel_belt_x            : int  -3 -2 -2 -6 -4 1 1 2 -1 -18 ...
 $ accel_belt_y            : int  -18 -13 -6 -5 -9 -9 -24 -36 -19 18 ...
 $ accel_belt_z            : int  22 16 8 7 0 -5 -8 -9 -7 1 ...
 $ magnet_belt_x           : int  387 405 409 422 418 432 438 440 443 449 ...
 $ magnet_belt_y           : int  525 512 511 513 508 510 508 503 507 499 ...
 $ magnet_belt_z           : int  -267 -254 -244 -221 -208 -189 -176 -163 -140 -132 ...
 $ roll_arm                : num  132 129 125 120 115 110 104 98.6 93.2 88.5 ...
 $ pitch_arm               : num  -43.7 -45.3 -46.8 -48.1 -49.1 -49.6 -49.9 -49.7 -49 -48.1 ...
 $ yaw_arm                 : num  -53.6 -49 -43.7 -38.1 -31.7 -25.8 -18.5 -11.4 -4.49 1.82 ...
 $ total_accel_arm         : int  38 38 35 35 34 33 29 28 27 22 ...
 $ var_accel_arm           : num  65.1 65.1 65.1 65.1 65.1 ...
 $ avg_roll_arm            : num  76.2 76.2 76.2 76.2 76.2 ...
 $ stddev_roll_arm         : num  16.1 16.1 16.1 16.1 16.1 ...
 $ var_roll_arm            : num  259 259 259 259 259 ...
 $ avg_pitch_arm           : num  -10.2 -10.2 -10.2 -10.2 -10.2 ...
 $ stddev_pitch_arm        : num  10.7 10.7 10.7 10.7 10.7 ...
 $ var_pitch_arm           : num  114 114 114 114 114 ...
 $ avg_yaw_arm             : num  19.1 19.1 19.1 19.1 19.1 ...
 $ stddev_yaw_arm          : num  35.9 35.9 35.9 35.9 35.9 ...
 $ var_yaw_arm             : num  1287 1287 1287 1287 1287 ...
 $ gyros_arm_x             : num  2.65 2.79 2.91 3.08 3.2 3.31 3.5 3.53 3.4 3.48 ...
 $ gyros_arm_y             : num  -0.61 -0.64 -0.69 -0.72 -0.77 -0.83 -0.83 -0.83 -0.83 -0.8 ...
 $ gyros_arm_z             : num  -0.02 -0.11 -0.15 -0.23 -0.25 -0.3 -0.31 -0.21 -0.11 -0.15 ...
 $ accel_arm_x             : int  143 146 156 158 163 160 165 153 143 135 ...
 $ accel_arm_y             : int  30 35 44 52 55 59 67 70 78 96 ...
 $ accel_arm_z             : int  -346 -339 -307 -305 -288 -274 -225 -218 -205 -134 ...
 $ magnet_arm_x            : int  556 599 613 646 670 696 721 725 740 741 ...
 $ magnet_arm_y            : int  -205 -206 -198 -186 -175 -174 -161 -152 -133 -115 ...
 $ magnet_arm_z            : int  -374 -335 -319 -268 -241 -193 -121 -105 -43 14 ...
 $ kurtosis_roll_arm       : num  -1.18 -1.18 -1.18 -1.18 -1.18 ...
 $ kurtosis_picth_arm      : num  -0.969 -0.969 -0.969 -0.969 -0.969 ...
 $ kurtosis_yaw_arm        : num  -0.87 -0.87 -0.87 -0.87 -0.87 ...
 $ skewness_roll_arm       : num  0.124 0.124 0.124 0.124 0.124 ...
 $ skewness_pitch_arm      : num  -0.103 -0.103 -0.103 -0.103 -0.103 ...
 $ skewness_yaw_arm        : num  0.0598 0.0598 0.0598 0.0598 0.0598 ...
 $ max_roll_arm            : num  8.45 8.45 8.45 8.45 8.45 8.45 8.45 8.45 8.45 8.45 ...
 $ max_picth_arm           : num  77.2 77.2 77.2 77.2 77.2 ...
 $ max_yaw_arm             : int  38 38 38 38 38 38 38 38 38 38 ...
 $ min_roll_arm            : num  -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 ...
 $ min_pitch_arm           : num  -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 ...
 $ min_yaw_arm             : int  10 10 10 10 10 10 10 10 10 10 ...
 $ amplitude_roll_arm      : num  36.9 36.9 36.9 36.9 36.9 ...
 $ amplitude_pitch_arm     : num  122 122 122 122 122 ...
 $ amplitude_yaw_arm       : int  27 27 27 27 27 27 27 27 27 27 ...
 $ roll_dumbbell           : num  51.2 55.8 55.5 55.9 55.2 ...
 $ pitch_dumbbell          : num  11.7 9.65 6.88 11.08 11.43 ...
 $ yaw_dumbbell            : num  104.3 100.2 101.1 99.8 100.4 ...
 $ kurtosis_roll_dumbbell  : num  -0.0959 -0.0959 -0.0959 -0.0959 -0.0959 ...
 $ kurtosis_picth_dumbbell : num  -0.442 -0.442 -0.442 -0.442 -0.442 ...
 $ skewness_roll_dumbbell  : num  0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 ...
 $ skewness_pitch_dumbbell : num  -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 ...
 $ max_roll_dumbbell       : num  41.9 41.9 41.9 41.9 41.9 ...
 $ max_picth_dumbbell      : num  133 133 133 133 133 133 133 133 133 133 ...
 $ max_yaw_dumbbell        : num  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 ...
 $ min_roll_dumbbell       : num  -26.8 -26.8 -26.8 -26.8 -26.8 ...
 $ min_pitch_dumbbell      : num  20.2 20.2 20.2 20.2 20.2 20.2 20.2 20.2 20.2 20.2 ...
 $ min_yaw_dumbbell        : num  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 ...
 $ amplitude_roll_dumbbell : num  55.7 55.7 55.7 55.7 55.7 ...
 $ amplitude_pitch_dumbbell: num  54.7 54.7 54.7 54.7 54.7 ...
 $ amplitude_yaw_dumbbell  : int  0 0 0 0 0 0 0 0 0 0 ...
 $ total_accel_dumbbell    : int  4 4 4 5 4 4 4 4 4 4 ...
 $ var_accel_dumbbell      : num  2.42 2.42 2.42 2.42 2.42 ...
 $ avg_roll_dumbbell       : num  -5.12 -5.12 -5.12 -5.12 -5.12 ...
  [list output truncated]
> data<-Weight_lift
> # load libraries
> library(caret)
Loading required package: lattice
Loading required package: ggplot2
> library(randomForest)
randomForest 4.6-14
Type rfNews() to see new features/changes/bug fixes.

Attaching package: ‘randomForest’

The following object is masked from ‘package:ggplot2’:

    margin

> library(rpart)
> library(rpart.plot)
> library(ggplot2)
> library(lattice)
> library(rattle)
Rattle: A free graphical interface for data science with R.
Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
Type 'rattle()' to shake, rattle, and roll your data.

Attaching package: ‘rattle’

The following object is masked from ‘package:randomForest’:

    importance

> 
> library(C50)
> #install.package('devtools') # Only needed if you dont have this installed.
> library(devtools)
> install_github('adam-m-mcelhinney/helpRFunctions')
Skipping install of 'helpRFunctions' from a github remote, the SHA1 (9eb16e8c) has not changed since last install.
  Use `force = TRUE` to force installation
> library(helpRFunctions)
> names(data)
  [1] "user_name"                "raw_timestamp_part_1"    
  [3] "raw_timestamp_part_2"     "cvtd_timestamp"          
  [5] "new_window"               "num_window"              
  [7] "roll_belt"                "pitch_belt"              
  [9] "yaw_belt"                 "total_accel_belt"        
 [11] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
 [13] "skewness_roll_belt"       "skewness_roll_belt.1"    
 [15] "max_roll_belt"            "max_picth_belt"          
 [17] "max_yaw_belt"             "min_roll_belt"           
 [19] "min_pitch_belt"           "min_yaw_belt"            
 [21] "amplitude_roll_belt"      "amplitude_pitch_belt"    
 [23] "amplitude_yaw_belt"       "var_total_accel_belt"    
 [25] "avg_roll_belt"            "stddev_roll_belt"        
 [27] "var_roll_belt"            "avg_pitch_belt"          
 [29] "stddev_pitch_belt"        "var_pitch_belt"          
 [31] "avg_yaw_belt"             "stddev_yaw_belt"         
 [33] "var_yaw_belt"             "gyros_belt_x"            
 [35] "gyros_belt_y"             "gyros_belt_z"            
 [37] "accel_belt_x"             "accel_belt_y"            
 [39] "accel_belt_z"             "magnet_belt_x"           
 [41] "magnet_belt_y"            "magnet_belt_z"           
 [43] "roll_arm"                 "pitch_arm"               
 [45] "yaw_arm"                  "total_accel_arm"         
 [47] "var_accel_arm"            "avg_roll_arm"            
 [49] "stddev_roll_arm"          "var_roll_arm"            
 [51] "avg_pitch_arm"            "stddev_pitch_arm"        
 [53] "var_pitch_arm"            "avg_yaw_arm"             
 [55] "stddev_yaw_arm"           "var_yaw_arm"             
 [57] "gyros_arm_x"              "gyros_arm_y"             
 [59] "gyros_arm_z"              "accel_arm_x"             
 [61] "accel_arm_y"              "accel_arm_z"             
 [63] "magnet_arm_x"             "magnet_arm_y"            
 [65] "magnet_arm_z"             "kurtosis_roll_arm"       
 [67] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
 [69] "skewness_roll_arm"        "skewness_pitch_arm"      
 [71] "skewness_yaw_arm"         "max_roll_arm"            
 [73] "max_picth_arm"            "max_yaw_arm"             
 [75] "min_roll_arm"             "min_pitch_arm"           
 [77] "min_yaw_arm"              "amplitude_roll_arm"      
 [79] "amplitude_pitch_arm"      "amplitude_yaw_arm"       
 [81] "roll_dumbbell"            "pitch_dumbbell"          
 [83] "yaw_dumbbell"             "kurtosis_roll_dumbbell"  
 [85] "kurtosis_picth_dumbbell"  "skewness_roll_dumbbell"  
 [87] "skewness_pitch_dumbbell"  "max_roll_dumbbell"       
 [89] "max_picth_dumbbell"       "max_yaw_dumbbell"        
 [91] "min_roll_dumbbell"        "min_pitch_dumbbell"      
 [93] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
 [95] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
 [97] "total_accel_dumbbell"     "var_accel_dumbbell"      
 [99] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
[101] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
[103] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
[105] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
[107] "var_yaw_dumbbell"         "gyros_dumbbell_x"        
[109] "gyros_dumbbell_y"         "gyros_dumbbell_z"        
[111] "accel_dumbbell_x"         "accel_dumbbell_y"        
[113] "accel_dumbbell_z"         "magnet_dumbbell_x"       
[115] "magnet_dumbbell_y"        "magnet_dumbbell_z"       
[117] "roll_forearm"             "pitch_forearm"           
[119] "yaw_forearm"              "kurtosis_roll_forearm"   
[121] "kurtosis_picth_forearm"   "skewness_roll_forearm"   
[123] "skewness_pitch_forearm"   "max_roll_forearm"        
[125] "max_picth_forearm"        "max_yaw_forearm"         
[127] "min_roll_forearm"         "min_pitch_forearm"       
[129] "min_yaw_forearm"          "amplitude_roll_forearm"  
[131] "amplitude_pitch_forearm"  "amplitude_yaw_forearm"   
[133] "total_accel_forearm"      "var_accel_forearm"       
[135] "avg_roll_forearm"         "stddev_roll_forearm"     
[137] "var_roll_forearm"         "avg_pitch_forearm"       
[139] "stddev_pitch_forearm"     "var_pitch_forearm"       
[141] "avg_yaw_forearm"          "stddev_yaw_forearm"      
[143] "var_yaw_forearm"          "gyros_forearm_x"         
[145] "gyros_forearm_y"          "gyros_forearm_z"         
[147] "accel_forearm_x"          "accel_forearm_y"         
[149] "accel_forearm_z"          "magnet_forearm_x"        
[151] "magnet_forearm_y"         "magnet_forearm_z"        
> dim(data)
[1] 4024  152
> pairs(data[1:10])
> # enable multi-core processing
> library(doParallel)
Loading required package: foreach
Loading required package: iterators
Loading required package: parallel
> #cl <- makeCluster(detectCores())

registerDoParallel()
> set.seed(12345)
> dataTrain<-data[1:3020,]
> dataTest<-data[3021:4024,]
> head(dataTrain)
  user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
1    eurico           1322489729                34670 28/11/2011 14:15
2    eurico           1322489729                62641 28/11/2011 14:15
3    eurico           1322489729                70653 28/11/2011 14:15
4    eurico           1322489729                82654 28/11/2011 14:15
5    eurico           1322489729                90637 28/11/2011 14:15
6    eurico           1322489729               170626 28/11/2011 14:15
  new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
1         no          1      3.70       41.6    -82.8                3
2         no          1      3.66       42.8    -82.5                2
3         no          1      3.58       43.7    -82.3                1
4         no          1      3.56       44.4    -82.1                1
5         no          1      3.57       45.1    -81.9                1
6         no          1      3.45       45.6    -81.9                1
  kurtosis_roll_belt kurtosis_picth_belt skewness_roll_belt
1           -1.03566            -0.39133           0.005406
2           -1.03566            -0.39133           0.005406
3           -1.03566            -0.39133           0.005406
4           -1.03566            -0.39133           0.005406
5           -1.03566            -0.39133           0.005406
6           -1.03566            -0.39133           0.005406
  skewness_roll_belt.1 max_roll_belt max_picth_belt max_yaw_belt
1             0.045115          -4.1             20           -1
2             0.045115          -4.1             20           -1
3             0.045115          -4.1             20           -1
4             0.045115          -4.1             20           -1
5             0.045115          -4.1             20           -1
6             0.045115          -4.1             20           -1
  min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt
1         -7.25             18           -1               1.345
2         -7.25             18           -1               1.345
3         -7.25             18           -1               1.345
4         -7.25             18           -1               1.345
5         -7.25             18           -1               1.345
6         -7.25             18           -1               1.345
  amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt
1                    2                  0                  0.3
2                    2                  0                  0.3
3                    2                  0                  0.3
4                    2                  0                  0.3
5                    2                  0                  0.3
6                    2                  0                  0.3
  avg_roll_belt stddev_roll_belt var_roll_belt avg_pitch_belt
1         121.9              0.6          0.35          25.75
2         121.9              0.6          0.35          25.75
3         121.9              0.6          0.35          25.75
4         121.9              0.6          0.35          25.75
5         121.9              0.6          0.35          25.75
6         121.9              0.6          0.35          25.75
  stddev_pitch_belt var_pitch_belt avg_yaw_belt stddev_yaw_belt
1              0.35            0.1        -4.95             0.4
2              0.35            0.1        -4.95             0.4
3              0.35            0.1        -4.95             0.4
4              0.35            0.1        -4.95             0.4
5              0.35            0.1        -4.95             0.4
6              0.35            0.1        -4.95             0.4
  var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z accel_belt_x
1         0.17         2.02         0.18         0.02           -3
2         0.17         1.96         0.14         0.05           -2
3         0.17         1.88         0.08         0.05           -2
4         0.17         1.80         0.03         0.08           -6
5         0.17         1.77         0.00         0.13           -4
6         0.17         1.75        -0.03         0.16            1
  accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y magnet_belt_z
1          -18           22           387           525          -267
2          -13           16           405           512          -254
3           -6            8           409           511          -244
4           -5            7           422           513          -221
5           -9            0           418           508          -208
6           -9           -5           432           510          -189
  roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm avg_roll_arm
1      132     -43.7   -53.6              38       65.0977     76.22175
2      129     -45.3   -49.0              38       65.0977     76.22175
3      125     -46.8   -43.7              35       65.0977     76.22175
4      120     -48.1   -38.1              35       65.0977     76.22175
5      115     -49.1   -31.7              34       65.0977     76.22175
6      110     -49.6   -25.8              33       65.0977     76.22175
  stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm var_pitch_arm
1         16.1039     259.3599      -10.1695         10.66725      113.7978
2         16.1039     259.3599      -10.1695         10.66725      113.7978
3         16.1039     259.3599      -10.1695         10.66725      113.7978
4         16.1039     259.3599      -10.1695         10.66725      113.7978
5         16.1039     259.3599      -10.1695         10.66725      113.7978
6         16.1039     259.3599      -10.1695         10.66725      113.7978
  avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x gyros_arm_y
1     19.0615        35.8809    1287.463        2.65       -0.61
2     19.0615        35.8809    1287.463        2.79       -0.64
3     19.0615        35.8809    1287.463        2.91       -0.69
4     19.0615        35.8809    1287.463        3.08       -0.72
5     19.0615        35.8809    1287.463        3.20       -0.77
6     19.0615        35.8809    1287.463        3.31       -0.83
  gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y
1       -0.02         143          30        -346          556         -205
2       -0.11         146          35        -339          599         -206
3       -0.15         156          44        -307          613         -198
4       -0.23         158          52        -305          646         -186
5       -0.25         163          55        -288          670         -175
6       -0.30         160          59        -274          696         -174
  magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm kurtosis_yaw_arm
1         -374          -1.18224           -0.96912         -0.86977
2         -335          -1.18224           -0.96912         -0.86977
3         -319          -1.18224           -0.96912         -0.86977
4         -268          -1.18224           -0.96912         -0.86977
5         -241          -1.18224           -0.96912         -0.86977
6         -193          -1.18224           -0.96912         -0.86977
  skewness_roll_arm skewness_pitch_arm skewness_yaw_arm max_roll_arm
1           0.12353           -0.10319         0.059765         8.45
2           0.12353           -0.10319         0.059765         8.45
3           0.12353           -0.10319         0.059765         8.45
4           0.12353           -0.10319         0.059765         8.45
5           0.12353           -0.10319         0.059765         8.45
6           0.12353           -0.10319         0.059765         8.45
  max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm min_yaw_arm
1         77.25          38        -33.6         -58.6          10
2         77.25          38        -33.6         -58.6          10
3         77.25          38        -33.6         -58.6          10
4         77.25          38        -33.6         -58.6          10
5         77.25          38        -33.6         -58.6          10
6         77.25          38        -33.6         -58.6          10
  amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell
1             36.945               121.5                27      51.23554
2             36.945               121.5                27      55.82442
3             36.945               121.5                27      55.46983
4             36.945               121.5                27      55.94486
5             36.945               121.5                27      55.21174
6             36.945               121.5                27      54.24731
  pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
1      11.698847    104.26473               -0.09595
2       9.645819    100.22805               -0.09595
3       6.875244    101.08411               -0.09595
4      11.079297     99.78456               -0.09595
5      11.426833    100.42258               -0.09595
6      14.126636    100.61574               -0.09595
  kurtosis_picth_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
1                 -0.4422                 0.0819                  -0.216
2                 -0.4422                 0.0819                  -0.216
3                 -0.4422                 0.0819                  -0.216
4                 -0.4422                 0.0819                  -0.216
5                 -0.4422                 0.0819                  -0.216
6                 -0.4422                 0.0819                  -0.216
  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
1             41.85                133             -0.1            -26.75
2             41.85                133             -0.1            -26.75
3             41.85                133             -0.1            -26.75
4             41.85                133             -0.1            -26.75
5             41.85                133             -0.1            -26.75
6             41.85                133             -0.1            -26.75
  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
1               20.2             -0.1                   55.71
2               20.2             -0.1                   55.71
3               20.2             -0.1                   55.71
4               20.2             -0.1                   55.71
5               20.2             -0.1                   55.71
6               20.2             -0.1                   55.71
  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
1                    54.74                      0                    4
2                    54.74                      0                    4
3                    54.74                      0                    4
4                    54.74                      0                    5
5                    54.74                      0                    4
6                    54.74                      0                    4
  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
1            2.41635          -5.11805               17.058
2            2.41635          -5.11805               17.058
3            2.41635          -5.11805               17.058
4            2.41635          -5.11805               17.058
5            2.41635          -5.11805               17.058
6            2.41635          -5.11805               17.058
  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
1           291.001            13.9312               14.1062
2           291.001            13.9312               14.1062
3           291.001            13.9312               14.1062
4           291.001            13.9312               14.1062
5           291.001            13.9312               14.1062
6           291.001            13.9312               14.1062
  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
1           199.0775          64.7063             13.5747         184.5578
2           199.0775          64.7063             13.5747         184.5578
3           199.0775          64.7063             13.5747         184.5578
4           199.0775          64.7063             13.5747         184.5578
5           199.0775          64.7063             13.5747         184.5578
6           199.0775          64.7063             13.5747         184.5578
  gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
1            -0.31             0.16             0.08                5
2            -0.31             0.14             0.07                4
3            -0.31             0.16             0.05                3
4            -0.31             0.16             0.07                5
5            -0.31             0.14             0.07                5
6            -0.31             0.14             0.07                6
  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
1               21               37              -471               191
2               22               35              -472               184
3               23               37              -468               190
4               24               38              -469               184
5               23               37              -468               189
6               22               36              -473               188
  magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
1               277         -111          26.5         138
2               281         -112          26.2         138
3               275         -114          26.0         137
4               285         -115          25.8         137
5               292         -117          25.5         137
6               278         -118          25.1         137
  kurtosis_roll_forearm kurtosis_picth_forearm skewness_roll_forearm
1              -1.09475               -0.97525              -0.05065
2              -1.09475               -0.97525              -0.05065
3              -1.09475               -0.97525              -0.05065
4              -1.09475               -0.97525              -0.05065
5              -1.09475               -0.97525              -0.05065
6              -1.09475               -0.97525              -0.05065
  skewness_pitch_forearm max_roll_forearm max_picth_forearm max_yaw_forearm
1                0.17285             49.6               168            -1.1
2                0.17285             49.6               168            -1.1
3                0.17285             49.6               168            -1.1
4                0.17285             49.6               168            -1.1
5                0.17285             49.6               168            -1.1
6                0.17285             49.6               168            -1.1
  min_roll_forearm min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
1             4.65            -168.5            -1.1                   32.2
2             4.65            -168.5            -1.1                   32.2
3             4.65            -168.5            -1.1                   32.2
4             4.65            -168.5            -1.1                   32.2
5             4.65            -168.5            -1.1                   32.2
6             4.65            -168.5            -1.1                   32.2
  amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
1                   341.5                     0                  30
2                   341.5                     0                  31
3                   341.5                     0                  32
4                   341.5                     0                  33
5                   341.5                     0                  34
6                   341.5                     0                  36
  var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
1           14.0772         27.85936            45.16342         2749.163
2           14.0772         27.85936            45.16342         2749.163
3           14.0772         27.85936            45.16342         2749.163
4           14.0772         27.85936            45.16342         2749.163
5           14.0772         27.85936            45.16342         2749.163
6           14.0772         27.85936            45.16342         2749.163
  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
1          25.35597             8.906695          79.33451        17.09505
2          25.35597             8.906695          79.33451        17.09505
3          25.35597             8.906695          79.33451        17.09505
4          25.35597             8.906695          79.33451        17.09505
5          25.35597             8.906695          79.33451        17.09505
6          25.35597             8.906695          79.33451        17.09505
  stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
1           74.27584        5541.956           -0.05           -0.37
2           74.27584        5541.956           -0.06           -0.37
3           74.27584        5541.956           -0.05           -0.27
4           74.27584        5541.956            0.02           -0.24
5           74.27584        5541.956            0.08           -0.27
6           74.27584        5541.956            0.14           -0.29
  gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
1           -0.43            -170             155             184
2           -0.59            -178             164             182
3           -0.72            -182             172             185
4           -0.79            -185             182             188
5           -0.82            -188             195             188
6           -0.82            -208             207             190
  magnet_forearm_x magnet_forearm_y magnet_forearm_z
1            -1160             1400             -876
2            -1150             1410             -871
3            -1130             1400             -863
4            -1120             1400             -855
5            -1100             1400             -843
6            -1090             1400             -838
> head(dataTest)
     user_name raw_timestamp_part_1 raw_timestamp_part_2  cvtd_timestamp
3021     pedro           1323094996               656284 5/12/2011 14:23
3022     pedro           1323094996               664357 5/12/2011 14:23
3023     pedro           1323094996               672361 5/12/2011 14:23
3024     pedro           1323094996               692335 5/12/2011 14:23
3025     pedro           1323094996               700442 5/12/2011 14:23
3026     pedro           1323094996               712340 5/12/2011 14:23
     new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
3021         no         69       125       25.9    -3.24               20
3022         no         69       125       25.8    -3.13               20
3023         no         69       125       25.8    -3.07               20
3024         no         69       125       25.7    -3.01               20
3025         no         69       125       25.7    -2.96               20
3026         no         69       125       25.6    -2.80               20
     kurtosis_roll_belt kurtosis_picth_belt skewness_roll_belt
3021           -1.03566            -0.39133           0.005406
3022           -1.03566            -0.39133           0.005406
3023           -1.03566            -0.39133           0.005406
3024           -1.03566            -0.39133           0.005406
3025           -1.03566            -0.39133           0.005406
3026           -1.03566            -0.39133           0.005406
     skewness_roll_belt.1 max_roll_belt max_picth_belt max_yaw_belt
3021             0.045115          -4.1             20           -1
3022             0.045115          -4.1             20           -1
3023             0.045115          -4.1             20           -1
3024             0.045115          -4.1             20           -1
3025             0.045115          -4.1             20           -1
3026             0.045115          -4.1             20           -1
     min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt
3021         -7.25             18           -1               1.345
3022         -7.25             18           -1               1.345
3023         -7.25             18           -1               1.345
3024         -7.25             18           -1               1.345
3025         -7.25             18           -1               1.345
3026         -7.25             18           -1               1.345
     amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt
3021                    2                  0                  0.3
3022                    2                  0                  0.3
3023                    2                  0                  0.3
3024                    2                  0                  0.3
3025                    2                  0                  0.3
3026                    2                  0                  0.3
     avg_roll_belt stddev_roll_belt var_roll_belt avg_pitch_belt
3021         121.9              0.6          0.35          25.75
3022         121.9              0.6          0.35          25.75
3023         121.9              0.6          0.35          25.75
3024         121.9              0.6          0.35          25.75
3025         121.9              0.6          0.35          25.75
3026         121.9              0.6          0.35          25.75
     stddev_pitch_belt var_pitch_belt avg_yaw_belt stddev_yaw_belt
3021              0.35            0.1        -4.95             0.4
3022              0.35            0.1        -4.95             0.4
3023              0.35            0.1        -4.95             0.4
3024              0.35            0.1        -4.95             0.4
3025              0.35            0.1        -4.95             0.4
3026              0.35            0.1        -4.95             0.4
     var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z accel_belt_x
3021         0.17        -0.37        -0.02        -0.43          -40
3022         0.17        -0.37        -0.02        -0.43          -42
3023         0.17        -0.39        -0.02        -0.44          -43
3024         0.17        -0.39        -0.02        -0.44          -42
3025         0.17        -0.39        -0.02        -0.44          -40
3026         0.17        -0.39        -0.02        -0.43          -42
     accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y magnet_belt_z
3021           71         -175             2           579          -375
3022           72         -178            -2           581          -390
3023           71         -179             5           579          -387
3024           69         -178             4           575          -389
3025           68         -177             5           584          -368
3026           70         -177             7           575          -389
     roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm avg_roll_arm
3021    -24.7      6.42    -115               6       65.0977     76.22175
3022    -25.0      6.61    -115               5       65.0977     76.22175
3023    -25.2      6.71    -116               5       65.0977     76.22175
3024    -25.5      6.70    -116               4       65.0977     76.22175
3025    -25.6      6.61    -116               4       65.0977     76.22175
3026    -25.5      6.43    -117               3       65.0977     76.22175
     stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
3021         16.1039     259.3599      -10.1695         10.66725
3022         16.1039     259.3599      -10.1695         10.66725
3023         16.1039     259.3599      -10.1695         10.66725
3024         16.1039     259.3599      -10.1695         10.66725
3025         16.1039     259.3599      -10.1695         10.66725
3026         16.1039     259.3599      -10.1695         10.66725
     var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
3021      113.7978     19.0615        35.8809    1287.463        0.47
3022      113.7978     19.0615        35.8809    1287.463        0.34
3023      113.7978     19.0615        35.8809    1287.463        0.22
3024      113.7978     19.0615        35.8809    1287.463        0.14
3025      113.7978     19.0615        35.8809    1287.463        0.10
3026      113.7978     19.0615        35.8809    1287.463        0.03
     gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
3021       -0.37       -0.13          -4          14          56
3022       -0.27       -0.31          -2          15          49
3023       -0.19       -0.44           6           6          49
3024       -0.13       -0.49           7           3          42
3025       -0.10       -0.44           7           4          37
3026       -0.10       -0.39           5           3          33
     magnet_arm_x magnet_arm_y magnet_arm_z kurtosis_roll_arm
3021         -378          385          448          -1.18224
3022         -389          377          454          -1.18224
3023         -392          372          461          -1.18224
3024         -391          369          463          -1.18224
3025         -395          369          453          -1.18224
3026         -399          360          461          -1.18224
     kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm
3021           -0.96912         -0.86977           0.12353
3022           -0.96912         -0.86977           0.12353
3023           -0.96912         -0.86977           0.12353
3024           -0.96912         -0.86977           0.12353
3025           -0.96912         -0.86977           0.12353
3026           -0.96912         -0.86977           0.12353
     skewness_pitch_arm skewness_yaw_arm max_roll_arm max_picth_arm
3021           -0.10319         0.059765         8.45         77.25
3022           -0.10319         0.059765         8.45         77.25
3023           -0.10319         0.059765         8.45         77.25
3024           -0.10319         0.059765         8.45         77.25
3025           -0.10319         0.059765         8.45         77.25
3026           -0.10319         0.059765         8.45         77.25
     max_yaw_arm min_roll_arm min_pitch_arm min_yaw_arm amplitude_roll_arm
3021          38        -33.6         -58.6          10             36.945
3022          38        -33.6         -58.6          10             36.945
3023          38        -33.6         -58.6          10             36.945
3024          38        -33.6         -58.6          10             36.945
3025          38        -33.6         -58.6          10             36.945
3026          38        -33.6         -58.6          10             36.945
     amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell pitch_dumbbell
3021               121.5                27      2.239617       27.20759
3022               121.5                27     -2.321713       27.02507
3023               121.5                27      1.132156       27.51712
3024               121.5                27     -6.717451       30.68508
3025               121.5                27    -11.203061       29.49277
3026               121.5                27     -4.579686       31.41632
     yaw_dumbbell kurtosis_roll_dumbbell kurtosis_picth_dumbbell
3021     129.7753               -0.09595                 -0.4422
3022     129.9500               -0.09595                 -0.4422
3023     129.5380               -0.09595                 -0.4422
3024     125.6207               -0.09595                 -0.4422
3025     125.4100               -0.09595                 -0.4422
3026     125.3085               -0.09595                 -0.4422
     skewness_roll_dumbbell skewness_pitch_dumbbell max_roll_dumbbell
3021                 0.0819                  -0.216             41.85
3022                 0.0819                  -0.216             41.85
3023                 0.0819                  -0.216             41.85
3024                 0.0819                  -0.216             41.85
3025                 0.0819                  -0.216             41.85
3026                 0.0819                  -0.216             41.85
     max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
3021                133             -0.1            -26.75
3022                133             -0.1            -26.75
3023                133             -0.1            -26.75
3024                133             -0.1            -26.75
3025                133             -0.1            -26.75
3026                133             -0.1            -26.75
     min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
3021               20.2             -0.1                   55.71
3022               20.2             -0.1                   55.71
3023               20.2             -0.1                   55.71
3024               20.2             -0.1                   55.71
3025               20.2             -0.1                   55.71
3026               20.2             -0.1                   55.71
     amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
3021                    54.74                      0                    9
3022                    54.74                      0                    9
3023                    54.74                      0                    9
3024                    54.74                      0                    9
3025                    54.74                      0                    9
3026                    54.74                      0                    9
     var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
3021            2.41635          -5.11805               17.058
3022            2.41635          -5.11805               17.058
3023            2.41635          -5.11805               17.058
3024            2.41635          -5.11805               17.058
3025            2.41635          -5.11805               17.058
3026            2.41635          -5.11805               17.058
     var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
3021           291.001            13.9312               14.1062
3022           291.001            13.9312               14.1062
3023           291.001            13.9312               14.1062
3024           291.001            13.9312               14.1062
3025           291.001            13.9312               14.1062
3026           291.001            13.9312               14.1062
     var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell
3021           199.0775          64.7063             13.5747
3022           199.0775          64.7063             13.5747
3023           199.0775          64.7063             13.5747
3024           199.0775          64.7063             13.5747
3025           199.0775          64.7063             13.5747
3026           199.0775          64.7063             13.5747
     var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z
3021         184.5578             0.47            -0.19            -0.13
3022         184.5578             0.47            -0.19            -0.15
3023         184.5578             0.47            -0.22            -0.16
3024         184.5578             0.50            -0.21            -0.18
3025         184.5578             0.50            -0.18            -0.18
3026         184.5578             0.48            -0.14            -0.16
     accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
3021               24                2               86               529
3022               23               -2               83               540
3023               24                1               85               533
3024               27               -6               85               534
3025               26              -10               85               537
3026               27               -4               83               542
     magnet_dumbbell_y magnet_dumbbell_z roll_forearm pitch_forearm
3021              -521               -91          115          16.3
3022              -516               -99          115          17.2
3023              -518               -92          116          18.1
3024              -506               -90          116          19.2
3025              -512               -88          117          20.5
3026              -517               -94          117          21.9
     yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
3021        87.7              -1.09475               -0.97525
3022        89.0              -1.09475               -0.97525
3023        90.4              -1.09475               -0.97525
3024        91.9              -1.09475               -0.97525
3025        93.4              -1.09475               -0.97525
3026        94.8              -1.09475               -0.97525
     skewness_roll_forearm skewness_pitch_forearm max_roll_forearm
3021              -0.05065                0.17285             49.6
3022              -0.05065                0.17285             49.6
3023              -0.05065                0.17285             49.6
3024              -0.05065                0.17285             49.6
3025              -0.05065                0.17285             49.6
3026              -0.05065                0.17285             49.6
     max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
3021               168            -1.1             4.65            -168.5
3022               168            -1.1             4.65            -168.5
3023               168            -1.1             4.65            -168.5
3024               168            -1.1             4.65            -168.5
3025               168            -1.1             4.65            -168.5
3026               168            -1.1             4.65            -168.5
     min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
3021            -1.1                   32.2                   341.5
3022            -1.1                   32.2                   341.5
3023            -1.1                   32.2                   341.5
3024            -1.1                   32.2                   341.5
3025            -1.1                   32.2                   341.5
3026            -1.1                   32.2                   341.5
     amplitude_yaw_forearm total_accel_forearm var_accel_forearm
3021                     0                  36           14.0772
3022                     0                  36           14.0772
3023                     0                  37           14.0772
3024                     0                  37           14.0772
3025                     0                  37           14.0772
3026                     0                  38           14.0772
     avg_roll_forearm stddev_roll_forearm var_roll_forearm
3021         27.85936            45.16342         2749.163
3022         27.85936            45.16342         2749.163
3023         27.85936            45.16342         2749.163
3024         27.85936            45.16342         2749.163
3025         27.85936            45.16342         2749.163
3026         27.85936            45.16342         2749.163
     avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm
3021          25.35597             8.906695          79.33451
3022          25.35597             8.906695          79.33451
3023          25.35597             8.906695          79.33451
3024          25.35597             8.906695          79.33451
3025          25.35597             8.906695          79.33451
3026          25.35597             8.906695          79.33451
     avg_yaw_forearm stddev_yaw_forearm var_yaw_forearm gyros_forearm_x
3021        17.09505           74.27584        5541.956            0.42
3022        17.09505           74.27584        5541.956            0.50
3023        17.09505           74.27584        5541.956            0.55
3024        17.09505           74.27584        5541.956            0.58
3025        17.09505           74.27584        5541.956            0.51
3026        17.09505           74.27584        5541.956            0.40
     gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
3021           -0.87           -0.02             105             293
3022           -0.90            0.07              80             300
3023           -1.01            0.08              79             316
3024           -1.24            0.03              60             317
3025           -1.32           -0.10              42             317
3026           -1.30           -0.23              40             325
     accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
3021            -164             -275              791              694
3022            -161             -290              792              678
3023            -164             -300              794              682
3024            -169             -322              794              672
3025            -171             -341              787              676
3026            -170             -343              785              681
> train_control<- trainControl(method="cv", number=10)
> 
> model<- train( pitch_belt ~., data=dataTrain,trControl=train_control, method="rf")

Random Forest 

3020 samples
 151 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 2718, 2718, 2716, 2719, 2718, 2720, ... 
Resampling results across tuning parameters:

  mtry  RMSE       Rsquared   MAE      
    2   3.7719395  0.9858292  2.0072729
   80   0.4378226  0.9994475  0.1698078
  159   0.5446360  0.9991612  0.1890224

RMSE was used to select the optimal model using the smallest value.
The final value used for the model was mtry = 80.

> summary(data)
       X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
 Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
 1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
 Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
 Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
 3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
 Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
          cvtd_timestamp  new_window    num_window      roll_belt     
 28/11/2011 14:14: 1498   no :19216   Min.   :  1.0   Min.   :-28.90  
 05/12/2011 11:24: 1497   yes:  406   1st Qu.:222.0   1st Qu.:  1.10  
 30/11/2011 17:11: 1440               Median :424.0   Median :113.00  
 05/12/2011 11:25: 1425               Mean   :430.6   Mean   : 64.41  
 02/12/2011 14:57: 1380               3rd Qu.:644.0   3rd Qu.:123.00  
 02/12/2011 13:34: 1375               Max.   :864.0   Max.   :162.00  
   pitch_belt          yaw_belt       total_accel_belt kurtosis_roll_belt
 Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00    Min.   :-2.121    
 1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00    1st Qu.:-1.329    
 Median :  5.2800   Median : -13.00   Median :17.00    Median :-0.899    
 Mean   :  0.3053   Mean   : -11.21   Mean   :11.31    Mean   :-0.220    
 3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00    3rd Qu.:-0.219    
 Max.   : 60.3000   Max.   : 179.00   Max.   :29.00    Max.   :33.000    
 kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
 Min.   :-2.190      Mode:logical      Min.   :-5.745    
 1st Qu.:-1.107      NA's:19622        1st Qu.:-0.444    
 Median :-0.151                        Median : 0.000    
 Mean   : 4.334                        Mean   :-0.026    
 3rd Qu.: 3.178                        3rd Qu.: 0.417    
 Max.   :58.000                        Max.   : 3.595    
 skewness_roll_belt.1 skewness_yaw_belt max_roll_belt     max_picth_belt 
 Min.   :-7.616       Mode:logical      Min.   :-94.300   Min.   : 3.00  
 1st Qu.:-1.114       NA's:19622        1st Qu.:-88.000   1st Qu.: 5.00  
 Median :-0.068                         Median : -5.100   Median :18.00  
 Mean   :-0.296                         Mean   : -6.667   Mean   :12.92  
 3rd Qu.: 0.661                         3rd Qu.: 18.500   3rd Qu.:19.00  
 Max.   : 7.348                         Max.   :180.000   Max.   :30.00  
  max_yaw_belt   min_roll_belt     min_pitch_belt   min_yaw_belt  
 Min.   :-2.10   Min.   :-180.00   Min.   : 0.00   Min.   :-2.10  
 1st Qu.:-1.30   1st Qu.: -88.40   1st Qu.: 3.00   1st Qu.:-1.30  
 Median :-0.90   Median :  -7.85   Median :16.00   Median :-0.90  
 Mean   :-0.22   Mean   : -10.44   Mean   :10.76   Mean   :-0.22  
 3rd Qu.:-0.20   3rd Qu.:   9.05   3rd Qu.:17.00   3rd Qu.:-0.20  
 Max.   :33.00   Max.   : 173.00   Max.   :23.00   Max.   :33.00  
 amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
 Min.   :  0.000     Min.   : 0.000       Min.   :0         
 1st Qu.:  0.300     1st Qu.: 1.000       1st Qu.:0         
 Median :  1.000     Median : 1.000       Median :0         
 Mean   :  3.769     Mean   : 2.167       Mean   :0         
 3rd Qu.:  2.083     3rd Qu.: 2.000       3rd Qu.:0         
 Max.   :360.000     Max.   :12.000       Max.   :0         
 var_total_accel_belt avg_roll_belt    stddev_roll_belt var_roll_belt    
 Min.   : 0.000       Min.   :-27.40   Min.   : 0.000   Min.   :  0.000  
 1st Qu.: 0.100       1st Qu.:  1.10   1st Qu.: 0.200   1st Qu.:  0.000  
 Median : 0.200       Median :116.35   Median : 0.400   Median :  0.100  
 Mean   : 0.926       Mean   : 68.06   Mean   : 1.337   Mean   :  7.699  
 3rd Qu.: 0.300       3rd Qu.:123.38   3rd Qu.: 0.700   3rd Qu.:  0.500  
 Max.   :16.500       Max.   :157.40   Max.   :14.200   Max.   :200.700  
 avg_pitch_belt    stddev_pitch_belt var_pitch_belt    avg_yaw_belt     
 Min.   :-51.400   Min.   :0.000     Min.   : 0.000   Min.   :-138.300  
 1st Qu.:  2.025   1st Qu.:0.200     1st Qu.: 0.000   1st Qu.: -88.175  
 Median :  5.200   Median :0.400     Median : 0.100   Median :  -6.550  
 Mean   :  0.520   Mean   :0.603     Mean   : 0.766   Mean   :  -8.831  
 3rd Qu.: 15.775   3rd Qu.:0.700     3rd Qu.: 0.500   3rd Qu.:  14.125  
 Max.   : 59.700   Max.   :4.000     Max.   :16.200   Max.   : 173.500  
 stddev_yaw_belt    var_yaw_belt        gyros_belt_x      
 Min.   :  0.000   Min.   :    0.000   Min.   :-1.040000  
 1st Qu.:  0.100   1st Qu.:    0.010   1st Qu.:-0.030000  
 Median :  0.300   Median :    0.090   Median : 0.030000  
 Mean   :  1.341   Mean   :  107.487   Mean   :-0.005592  
 3rd Qu.:  0.700   3rd Qu.:    0.475   3rd Qu.: 0.110000  
 Max.   :176.600   Max.   :31183.240   Max.   : 2.220000  
  gyros_belt_y       gyros_belt_z      accel_belt_x       accel_belt_y   
 Min.   :-0.64000   Min.   :-1.4600   Min.   :-120.000   Min.   :-69.00  
 1st Qu.: 0.00000   1st Qu.:-0.2000   1st Qu.: -21.000   1st Qu.:  3.00  
 Median : 0.02000   Median :-0.1000   Median : -15.000   Median : 35.00  
 Mean   : 0.03959   Mean   :-0.1305   Mean   :  -5.595   Mean   : 30.15  
 3rd Qu.: 0.11000   3rd Qu.:-0.0200   3rd Qu.:  -5.000   3rd Qu.: 61.00  
 Max.   : 0.64000   Max.   : 1.6200   Max.   :  85.000   Max.   :164.00  
  accel_belt_z     magnet_belt_x   magnet_belt_y   magnet_belt_z   
 Min.   :-275.00   Min.   :-52.0   Min.   :354.0   Min.   :-623.0  
 1st Qu.:-162.00   1st Qu.:  9.0   1st Qu.:581.0   1st Qu.:-375.0  
 Median :-152.00   Median : 35.0   Median :601.0   Median :-320.0  
 Mean   : -72.59   Mean   : 55.6   Mean   :593.7   Mean   :-345.5  
 3rd Qu.:  27.00   3rd Qu.: 59.0   3rd Qu.:610.0   3rd Qu.:-306.0  
 Max.   : 105.00   Max.   :485.0   Max.   :673.0   Max.   : 293.0  
    roll_arm         pitch_arm          yaw_arm          total_accel_arm
 Min.   :-180.00   Min.   :-88.800   Min.   :-180.0000   Min.   : 1.00  
 1st Qu.: -31.77   1st Qu.:-25.900   1st Qu.: -43.1000   1st Qu.:17.00  
 Median :   0.00   Median :  0.000   Median :   0.0000   Median :27.00  
 Mean   :  17.83   Mean   : -4.612   Mean   :  -0.6188   Mean   :25.51  
 3rd Qu.:  77.30   3rd Qu.: 11.200   3rd Qu.:  45.8750   3rd Qu.:33.00  
 Max.   : 180.00   Max.   : 88.500   Max.   : 180.0000   Max.   :66.00  
 var_accel_arm     avg_roll_arm     stddev_roll_arm    var_roll_arm      
 Min.   :  0.00   Min.   :-166.67   Min.   :  0.000   Min.   :    0.000  
 1st Qu.:  9.03   1st Qu.: -38.37   1st Qu.:  1.376   1st Qu.:    1.898  
 Median : 40.61   Median :   0.00   Median :  5.702   Median :   32.517  
 Mean   : 53.23   Mean   :  12.68   Mean   : 11.201   Mean   :  417.264  
 3rd Qu.: 75.62   3rd Qu.:  76.33   3rd Qu.: 14.921   3rd Qu.:  222.647  
 Max.   :331.70   Max.   : 163.33   Max.   :161.964   Max.   :26232.208  
 avg_pitch_arm     stddev_pitch_arm var_pitch_arm       avg_yaw_arm      
 Min.   :-81.773   Min.   : 0.000   Min.   :   0.000   Min.   :-173.440  
 1st Qu.:-22.770   1st Qu.: 1.642   1st Qu.:   2.697   1st Qu.: -29.198  
 Median :  0.000   Median : 8.133   Median :  66.146   Median :   0.000  
 Mean   : -4.901   Mean   :10.383   Mean   : 195.864   Mean   :   2.359  
 3rd Qu.:  8.277   3rd Qu.:16.327   3rd Qu.: 266.576   3rd Qu.:  38.185  
 Max.   : 75.659   Max.   :43.412   Max.   :1884.565   Max.   : 152.000  
 stddev_yaw_arm     var_yaw_arm         gyros_arm_x        gyros_arm_y     
 Min.   :  0.000   Min.   :    0.000   Min.   :-6.37000   Min.   :-3.4400  
 1st Qu.:  2.577   1st Qu.:    6.642   1st Qu.:-1.33000   1st Qu.:-0.8000  
 Median : 16.682   Median :  278.309   Median : 0.08000   Median :-0.2400  
 Mean   : 22.270   Mean   : 1055.933   Mean   : 0.04277   Mean   :-0.2571  
 3rd Qu.: 35.984   3rd Qu.: 1294.850   3rd Qu.: 1.57000   3rd Qu.: 0.1400  
 Max.   :177.044   Max.   :31344.568   Max.   : 4.87000   Max.   : 2.8400  
  gyros_arm_z       accel_arm_x       accel_arm_y      accel_arm_z     
 Min.   :-2.3300   Min.   :-404.00   Min.   :-318.0   Min.   :-636.00  
 1st Qu.:-0.0700   1st Qu.:-242.00   1st Qu.: -54.0   1st Qu.:-143.00  
 Median : 0.2300   Median : -44.00   Median :  14.0   Median : -47.00  
 Mean   : 0.2695   Mean   : -60.24   Mean   :  32.6   Mean   : -71.25  
 3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.0   3rd Qu.:  23.00  
 Max.   : 3.0200   Max.   : 437.00   Max.   : 308.0   Max.   : 292.00  
  magnet_arm_x     magnet_arm_y     magnet_arm_z    kurtosis_roll_arm
 Min.   :-584.0   Min.   :-392.0   Min.   :-597.0   Min.   :-1.809   
 1st Qu.:-300.0   1st Qu.:  -9.0   1st Qu.: 131.2   1st Qu.:-1.345   
 Median : 289.0   Median : 202.0   Median : 444.0   Median :-0.894   
 Mean   : 191.7   Mean   : 156.6   Mean   : 306.5   Mean   :-0.366   
 3rd Qu.: 637.0   3rd Qu.: 323.0   3rd Qu.: 545.0   3rd Qu.:-0.038   
 Max.   : 782.0   Max.   : 583.0   Max.   : 694.0   Max.   :21.456   
 kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
 Min.   :-2.084     Min.   :-2.103   Min.   :-2.541    Min.   :-4.565    
 1st Qu.:-1.280     1st Qu.:-1.220   1st Qu.:-0.561    1st Qu.:-0.618    
 Median :-1.010     Median :-0.733   Median : 0.040    Median :-0.035    
 Mean   :-0.542     Mean   : 0.406   Mean   : 0.068    Mean   :-0.065    
 3rd Qu.:-0.379     3rd Qu.: 0.115   3rd Qu.: 0.671    3rd Qu.: 0.454    
 Max.   :19.751     Max.   :56.000   Max.   : 4.394    Max.   : 3.043    
 skewness_yaw_arm  max_roll_arm     max_picth_arm       max_yaw_arm   
 Min.   :-6.708   Min.   :-73.100   Min.   :-173.000   Min.   : 4.00  
 1st Qu.:-0.743   1st Qu.: -0.175   1st Qu.:  -1.975   1st Qu.:29.00  
 Median :-0.133   Median :  4.950   Median :  23.250   Median :34.00  
 Mean   :-0.229   Mean   : 11.236   Mean   :  35.751   Mean   :35.46  
 3rd Qu.: 0.344   3rd Qu.: 26.775   3rd Qu.:  95.975   3rd Qu.:41.00  
 Max.   : 7.483   Max.   : 85.500   Max.   : 180.000   Max.   :65.00  
  min_roll_arm    min_pitch_arm      min_yaw_arm    amplitude_roll_arm
 Min.   :-89.10   Min.   :-180.00   Min.   : 1.00   Min.   :  0.000   
 1st Qu.:-41.98   1st Qu.: -72.62   1st Qu.: 8.00   1st Qu.:  5.425   
 Median :-22.45   Median : -33.85   Median :13.00   Median : 28.450   
 Mean   :-21.22   Mean   : -33.92   Mean   :14.66   Mean   : 32.452   
 3rd Qu.:  0.00   3rd Qu.:   0.00   3rd Qu.:19.00   3rd Qu.: 50.960   
 Max.   : 66.40   Max.   : 152.00   Max.   :38.00   Max.   :119.500   
 amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell     pitch_dumbbell   
 Min.   :  0.000     Min.   : 0.00     Min.   :-153.71   Min.   :-149.59  
 1st Qu.:  9.925     1st Qu.:13.00     1st Qu.: -18.49   1st Qu.: -40.89  
 Median : 54.900     Median :22.00     Median :  48.17   Median : -20.96  
 Mean   : 69.677     Mean   :20.79     Mean   :  23.84   Mean   : -10.78  
 3rd Qu.:115.175     3rd Qu.:28.75     3rd Qu.:  67.61   3rd Qu.:  17.50  
 Max.   :360.000     Max.   :52.00     Max.   : 153.55   Max.   : 149.40  
  yaw_dumbbell      kurtosis_roll_dumbbell kurtosis_picth_dumbbell
 Min.   :-150.871   Min.   :-2.174         Min.   :-2.200         
 1st Qu.: -77.644   1st Qu.:-0.682         1st Qu.:-0.721         
 Median :  -3.324   Median :-0.033         Median :-0.133         
 Mean   :   1.674   Mean   : 0.452         Mean   : 0.286         
 3rd Qu.:  79.643   3rd Qu.: 0.940         3rd Qu.: 0.584         
 Max.   : 154.952   Max.   :54.998         Max.   :55.628         
 kurtosis_yaw_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
 Mode:logical          Min.   :-7.384         Min.   :-7.447         
 NA's:19622            1st Qu.:-0.581         1st Qu.:-0.526         
                       Median :-0.076         Median :-0.091         
                       Mean   :-0.115         Mean   :-0.035         
                       3rd Qu.: 0.400         3rd Qu.: 0.505         
                       Max.   : 1.958         Max.   : 3.769         
 skewness_yaw_dumbbell max_roll_dumbbell max_picth_dumbbell
 Mode:logical          Min.   :-70.10    Min.   :-112.90   
 NA's:19622            1st Qu.:-27.15    1st Qu.: -66.70   
                       Median : 14.85    Median :  40.05   
                       Mean   : 13.76    Mean   :  32.75   
                       3rd Qu.: 50.58    3rd Qu.: 133.22   
                       Max.   :137.00    Max.   : 155.00   
 max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell min_yaw_dumbbell
 Min.   :-2.20    Min.   :-149.60   Min.   :-147.00    Min.   :-2.20   
 1st Qu.:-0.70    1st Qu.: -59.67   1st Qu.: -91.80    1st Qu.:-0.70   
 Median : 0.00    Median : -43.55   Median : -66.15    Median : 0.00   
 Mean   : 0.45    Mean   : -41.24   Mean   : -33.18    Mean   : 0.45   
 3rd Qu.: 0.90    3rd Qu.: -25.20   3rd Qu.:  21.20    3rd Qu.: 0.90   
 Max.   :55.00    Max.   :  73.20   Max.   : 120.90    Max.   :55.00   
 amplitude_roll_dumbbell amplitude_pitch_dumbbell amplitude_yaw_dumbbell
 Min.   :  0.00          Min.   :  0.00           Min.   :0             
 1st Qu.: 14.97          1st Qu.: 17.06           1st Qu.:0             
 Median : 35.05          Median : 41.73           Median :0             
 Mean   : 55.00          Mean   : 65.93           Mean   :0             
 3rd Qu.: 81.04          3rd Qu.: 99.55           3rd Qu.:0             
 Max.   :256.48          Max.   :273.59           Max.   :0             
 total_accel_dumbbell var_accel_dumbbell avg_roll_dumbbell
 Min.   : 0.00        Min.   :  0.000    Min.   :-128.96  
 1st Qu.: 4.00        1st Qu.:  0.378    1st Qu.: -12.33  
 Median :10.00        Median :  1.000    Median :  48.23  
 Mean   :13.72        Mean   :  4.388    Mean   :  23.86  
 3rd Qu.:19.00        3rd Qu.:  3.434    3rd Qu.:  64.37  
 Max.   :58.00        Max.   :230.428    Max.   : 125.99  
 stddev_roll_dumbbell var_roll_dumbbell  avg_pitch_dumbbell
 Min.   :  0.000      Min.   :    0.00   Min.   :-70.73    
 1st Qu.:  4.639      1st Qu.:   21.52   1st Qu.:-42.00    
 Median : 12.204      Median :  148.95   Median :-19.91    
 Mean   : 20.761      Mean   : 1020.27   Mean   :-12.33    
 3rd Qu.: 26.356      3rd Qu.:  694.65   3rd Qu.: 13.21    
 Max.   :123.778      Max.   :15321.01   Max.   : 94.28    
 stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell  
 Min.   : 0.000        Min.   :   0.00    Min.   :-117.950  
 1st Qu.: 3.482        1st Qu.:  12.12    1st Qu.: -76.696  
 Median : 8.089        Median :  65.44    Median :  -4.505  
 Mean   :13.147        Mean   : 350.31    Mean   :   0.202  
 3rd Qu.:19.238        3rd Qu.: 370.11    3rd Qu.:  71.234  
 Max.   :82.680        Max.   :6836.02    Max.   : 134.905  
 stddev_yaw_dumbbell var_yaw_dumbbell   gyros_dumbbell_x   
 Min.   :  0.000     Min.   :    0.00   Min.   :-204.0000  
 1st Qu.:  3.885     1st Qu.:   15.09   1st Qu.:  -0.0300  
 Median : 10.264     Median :  105.35   Median :   0.1300  
 Mean   : 16.647     Mean   :  589.84   Mean   :   0.1611  
 3rd Qu.: 24.674     3rd Qu.:  608.79   3rd Qu.:   0.3500  
 Max.   :107.088     Max.   :11467.91   Max.   :   2.2200  
 gyros_dumbbell_y   gyros_dumbbell_z  accel_dumbbell_x  accel_dumbbell_y 
 Min.   :-2.10000   Min.   : -2.380   Min.   :-419.00   Min.   :-189.00  
 1st Qu.:-0.14000   1st Qu.: -0.310   1st Qu.: -50.00   1st Qu.:  -8.00  
 Median : 0.03000   Median : -0.130   Median :  -8.00   Median :  41.50  
 Mean   : 0.04606   Mean   : -0.129   Mean   : -28.62   Mean   :  52.63  
 3rd Qu.: 0.21000   3rd Qu.:  0.030   3rd Qu.:  11.00   3rd Qu.: 111.00  
 Max.   :52.00000   Max.   :317.000   Max.   : 235.00   Max.   : 315.00  
 accel_dumbbell_z  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
 Min.   :-334.00   Min.   :-643.0    Min.   :-3600     Min.   :-262.00  
 1st Qu.:-142.00   1st Qu.:-535.0    1st Qu.:  231     1st Qu.: -45.00  
 Median :  -1.00   Median :-479.0    Median :  311     Median :  13.00  
 Mean   : -38.32   Mean   :-328.5    Mean   :  221     Mean   :  46.05  
 3rd Qu.:  38.00   3rd Qu.:-304.0    3rd Qu.:  390     3rd Qu.:  95.00  
 Max.   : 318.00   Max.   : 592.0    Max.   :  633     Max.   : 452.00  
  roll_forearm       pitch_forearm     yaw_forearm     
 Min.   :-180.0000   Min.   :-72.50   Min.   :-180.00  
 1st Qu.:  -0.7375   1st Qu.:  0.00   1st Qu.: -68.60  
 Median :  21.7000   Median :  9.24   Median :   0.00  
 Mean   :  33.8265   Mean   : 10.71   Mean   :  19.21  
 3rd Qu.: 140.0000   3rd Qu.: 28.40   3rd Qu.: 110.00  
 Max.   : 180.0000   Max.   : 89.80   Max.   : 180.00  
 kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
 Min.   :-1.879        Min.   :-2.098         Mode:logical        
 1st Qu.:-1.398        1st Qu.:-1.376         NA's:19622          
 Median :-1.119        Median :-0.890                             
 Mean   :-0.689        Mean   : 0.419                             
 3rd Qu.:-0.618        3rd Qu.: 0.054                             
 Max.   :40.060        Max.   :33.626                             
 skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
 Min.   :-2.297        Min.   :-5.241         Mode:logical        
 1st Qu.:-0.402        1st Qu.:-0.881         NA's:19622          
 Median : 0.003        Median :-0.156                             
 Mean   :-0.009        Mean   :-0.223                             
 3rd Qu.: 0.370        3rd Qu.: 0.514                             
 Max.   : 5.856        Max.   : 4.464                             
 max_roll_forearm max_picth_forearm max_yaw_forearm  min_roll_forearm 
 Min.   :-66.60   Min.   :-151.00   Min.   :-1.900   Min.   :-72.500  
 1st Qu.:  0.00   1st Qu.:   0.00   1st Qu.:-1.400   1st Qu.: -6.075  
 Median : 26.80   Median : 113.00   Median :-1.100   Median :  0.000  
 Mean   : 24.49   Mean   :  81.49   Mean   :-0.689   Mean   : -0.167  
 3rd Qu.: 45.95   3rd Qu.: 174.75   3rd Qu.:-0.600   3rd Qu.: 12.075  
 Max.   : 89.80   Max.   : 180.00   Max.   :40.100   Max.   : 62.100  
 min_pitch_forearm min_yaw_forearm  amplitude_roll_forearm
 Min.   :-180.00   Min.   :-1.900   Min.   :  0.000       
 1st Qu.:-175.00   1st Qu.:-1.400   1st Qu.:  1.125       
 Median : -61.00   Median :-1.100   Median : 17.770       
 Mean   : -57.57   Mean   :-0.689   Mean   : 24.653       
 3rd Qu.:   0.00   3rd Qu.:-0.600   3rd Qu.: 39.875       
 Max.   : 167.00   Max.   :40.100   Max.   :126.000       
 amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
 Min.   :  0.0           Min.   :0             Min.   :  0.00     
 1st Qu.:  2.0           1st Qu.:0             1st Qu.: 29.00     
 Median : 83.7           Median :0             Median : 36.00     
 Mean   :139.1           Mean   :0             Mean   : 34.72     
 3rd Qu.:350.0           3rd Qu.:0             3rd Qu.: 41.00     
 Max.   :360.0           Max.   :0             Max.   :108.00     
 var_accel_forearm avg_roll_forearm   stddev_roll_forearm
 Min.   :  0.000   Min.   :-177.234   Min.   :  0.000    
 1st Qu.:  6.759   1st Qu.:  -0.909   1st Qu.:  0.428    
 Median : 21.165   Median :  11.172   Median :  8.030    
 Mean   : 33.502   Mean   :  33.165   Mean   : 41.986    
 3rd Qu.: 51.240   3rd Qu.: 107.132   3rd Qu.: 85.373    
 Max.   :172.606   Max.   : 177.256   Max.   :179.171    
 var_roll_forearm   avg_pitch_forearm stddev_pitch_forearm
 Min.   :    0.00   Min.   :-68.17    Min.   : 0.000      
 1st Qu.:    0.18   1st Qu.:  0.00    1st Qu.: 0.336      
 Median :   64.48   Median : 12.02    Median : 5.516      
 Mean   : 5274.10   Mean   : 11.79    Mean   : 7.977      
 3rd Qu.: 7289.08   3rd Qu.: 28.48    3rd Qu.:12.866      
 Max.   :32102.24   Max.   : 72.09    Max.   :47.745      
 var_pitch_forearm  avg_yaw_forearm   stddev_yaw_forearm var_yaw_forearm   
 Min.   :   0.000   Min.   :-155.06   Min.   :  0.000    Min.   :    0.00  
 1st Qu.:   0.113   1st Qu.: -26.26   1st Qu.:  0.524    1st Qu.:    0.27  
 Median :  30.425   Median :   0.00   Median : 24.743    Median :  612.21  
 Mean   : 139.593   Mean   :  18.00   Mean   : 44.854    Mean   : 4639.85  
 3rd Qu.: 165.532   3rd Qu.:  85.79   3rd Qu.: 85.817    3rd Qu.: 7368.41  
 Max.   :2279.617   Max.   : 169.24   Max.   :197.508    Max.   :39009.33  
 gyros_forearm_x   gyros_forearm_y     gyros_forearm_z    accel_forearm_x  
 Min.   :-22.000   Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00  
 1st Qu.: -0.220   1st Qu.: -1.46000   1st Qu.: -0.1800   1st Qu.:-178.00  
 Median :  0.050   Median :  0.03000   Median :  0.0800   Median : -57.00  
 Mean   :  0.158   Mean   :  0.07517   Mean   :  0.1512   Mean   : -61.65  
 3rd Qu.:  0.560   3rd Qu.:  1.62000   3rd Qu.:  0.4900   3rd Qu.:  76.00  
 Max.   :  3.970   Max.   :311.00000   Max.   :231.0000   Max.   : 477.00  
 accel_forearm_y  accel_forearm_z   magnet_forearm_x  magnet_forearm_y
 Min.   :-632.0   Min.   :-446.00   Min.   :-1280.0   Min.   :-896.0  
 1st Qu.:  57.0   1st Qu.:-182.00   1st Qu.: -616.0   1st Qu.:   2.0  
 Median : 201.0   Median : -39.00   Median : -378.0   Median : 591.0  
 Mean   : 163.7   Mean   : -55.29   Mean   : -312.6   Mean   : 380.1  
 3rd Qu.: 312.0   3rd Qu.:  26.00   3rd Qu.:  -73.0   3rd Qu.: 737.0  
 Max.   : 923.0   Max.   : 291.00   Max.   :  672.0   Max.   :1480.0  
 magnet_forearm_z classe  
 Min.   :-973.0   A:5580  
 1st Qu.: 191.0   B:3797  
 Median : 511.0   C:3422  
 Mean   : 393.6   D:3216  
 3rd Qu.: 653.0   E:3607  
 Max.   :1090.0           
 [ reached getOption("max.print") -- omitted 1 row ]
> summary(validation)
       X            user_name raw_timestamp_part_1 raw_timestamp_part_2
 Min.   : 1.00   adelmo  :1   Min.   :1.322e+09    Min.   : 36553      
 1st Qu.: 5.75   carlitos:3   1st Qu.:1.323e+09    1st Qu.:268655      
 Median :10.50   charles :1   Median :1.323e+09    Median :530706      
 Mean   :10.50   eurico  :4   Mean   :1.323e+09    Mean   :512167      
 3rd Qu.:15.25   jeremy  :8   3rd Qu.:1.323e+09    3rd Qu.:787738      
 Max.   :20.00   pedro   :3   Max.   :1.323e+09    Max.   :920315      
          cvtd_timestamp new_window   num_window      roll_belt       
 30/11/2011 17:11:4      no:20      Min.   : 48.0   Min.   : -5.9200  
 05/12/2011 11:24:3                 1st Qu.:250.0   1st Qu.:  0.9075  
 30/11/2011 17:12:3                 Median :384.5   Median :  1.1100  
 05/12/2011 14:23:2                 Mean   :379.6   Mean   : 31.3055  
 28/11/2011 14:14:2                 3rd Qu.:467.0   3rd Qu.: 32.5050  
 02/12/2011 13:33:1                 Max.   :859.0   Max.   :129.0000  
   pitch_belt         yaw_belt      total_accel_belt kurtosis_roll_belt
 Min.   :-41.600   Min.   :-93.70   Min.   : 2.00    Mode:logical      
 1st Qu.:  3.013   1st Qu.:-88.62   1st Qu.: 3.00    NA's:20           
 Median :  4.655   Median :-87.85   Median : 4.00                      
 Mean   :  5.824   Mean   :-59.30   Mean   : 7.55                      
 3rd Qu.:  6.135   3rd Qu.:-63.50   3rd Qu.: 8.00                      
 Max.   : 27.800   Max.   :162.00   Max.   :21.00                      
 kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
 Mode:logical        Mode:logical      Mode:logical      
 NA's:20             NA's:20           NA's:20           
                                                         
                                                         
                                                         
                                                         
 skewness_roll_belt.1 skewness_yaw_belt max_roll_belt  max_picth_belt
 Mode:logical         Mode:logical      Mode:logical   Mode:logical  
 NA's:20              NA's:20           NA's:20        NA's:20       
                                                                     
                                                                     
                                                                     
                                                                     
 max_yaw_belt   min_roll_belt  min_pitch_belt min_yaw_belt  
 Mode:logical   Mode:logical   Mode:logical   Mode:logical  
 NA's:20        NA's:20        NA's:20        NA's:20       
                                                            
                                                            
                                                            
                                                            
 amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
 Mode:logical        Mode:logical         Mode:logical      
 NA's:20             NA's:20              NA's:20           
                                                            
                                                            
                                                            
                                                            
 var_total_accel_belt avg_roll_belt  stddev_roll_belt var_roll_belt 
 Mode:logical         Mode:logical   Mode:logical     Mode:logical  
 NA's:20              NA's:20        NA's:20          NA's:20       
                                                                    
                                                                    
                                                                    
                                                                    
 avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt  
 Mode:logical   Mode:logical      Mode:logical   Mode:logical  
 NA's:20        NA's:20           NA's:20        NA's:20       
                                                               
                                                               
                                                               
                                                               
 stddev_yaw_belt var_yaw_belt    gyros_belt_x     gyros_belt_y   
 Mode:logical    Mode:logical   Min.   :-0.500   Min.   :-0.050  
 NA's:20         NA's:20        1st Qu.:-0.070   1st Qu.:-0.005  
                                Median : 0.020   Median : 0.000  
                                Mean   :-0.045   Mean   : 0.010  
                                3rd Qu.: 0.070   3rd Qu.: 0.020  
                                Max.   : 0.240   Max.   : 0.110  
  gyros_belt_z      accel_belt_x     accel_belt_y     accel_belt_z    
 Min.   :-0.4800   Min.   :-48.00   Min.   :-16.00   Min.   :-187.00  
 1st Qu.:-0.1375   1st Qu.:-19.00   1st Qu.:  2.00   1st Qu.: -24.00  
 Median :-0.0250   Median :-13.00   Median :  4.50   Median :  27.00  
 Mean   :-0.1005   Mean   :-13.50   Mean   : 18.35   Mean   : -17.60  
 3rd Qu.: 0.0000   3rd Qu.: -8.75   3rd Qu.: 25.50   3rd Qu.:  38.25  
 Max.   : 0.0500   Max.   : 46.00   Max.   : 72.00   Max.   :  49.00  
 magnet_belt_x    magnet_belt_y   magnet_belt_z       roll_arm      
 Min.   :-13.00   Min.   :566.0   Min.   :-426.0   Min.   :-137.00  
 1st Qu.:  5.50   1st Qu.:578.5   1st Qu.:-398.5   1st Qu.:   0.00  
 Median : 33.50   Median :600.5   Median :-313.5   Median :   0.00  
 Mean   : 35.15   Mean   :601.5   Mean   :-346.9   Mean   :  16.42  
 3rd Qu.: 46.25   3rd Qu.:631.2   3rd Qu.:-305.0   3rd Qu.:  71.53  
 Max.   :169.00   Max.   :638.0   Max.   :-291.0   Max.   : 152.00  
   pitch_arm          yaw_arm        total_accel_arm var_accel_arm 
 Min.   :-63.800   Min.   :-167.00   Min.   : 3.00   Mode:logical  
 1st Qu.: -9.188   1st Qu.: -60.15   1st Qu.:20.25   NA's:20       
 Median :  0.000   Median :   0.00   Median :29.50                 
 Mean   : -3.950   Mean   :  -2.80   Mean   :26.40                 
 3rd Qu.:  3.465   3rd Qu.:  25.50   3rd Qu.:33.25                 
 Max.   : 55.000   Max.   : 178.00   Max.   :44.00                 
 avg_roll_arm   stddev_roll_arm var_roll_arm   avg_pitch_arm 
 Mode:logical   Mode:logical    Mode:logical   Mode:logical  
 NA's:20        NA's:20         NA's:20        NA's:20       
                                                             
                                                             
                                                             
                                                             
 stddev_pitch_arm var_pitch_arm  avg_yaw_arm    stddev_yaw_arm
 Mode:logical     Mode:logical   Mode:logical   Mode:logical  
 NA's:20          NA's:20        NA's:20        NA's:20       
                                                              
                                                              
                                                              
                                                              
 var_yaw_arm     gyros_arm_x      gyros_arm_y       gyros_arm_z     
 Mode:logical   Min.   :-3.710   Min.   :-2.0900   Min.   :-0.6900  
 NA's:20        1st Qu.:-0.645   1st Qu.:-0.6350   1st Qu.:-0.1800  
                Median : 0.020   Median :-0.0400   Median :-0.0250  
                Mean   : 0.077   Mean   :-0.1595   Mean   : 0.1205  
                3rd Qu.: 1.248   3rd Qu.: 0.2175   3rd Qu.: 0.5650  
                Max.   : 3.660   Max.   : 1.8500   Max.   : 1.1300  
  accel_arm_x      accel_arm_y      accel_arm_z       magnet_arm_x    
 Min.   :-341.0   Min.   :-65.00   Min.   :-404.00   Min.   :-428.00  
 1st Qu.:-277.0   1st Qu.: 52.25   1st Qu.:-128.50   1st Qu.:-373.75  
 Median :-194.5   Median :112.00   Median : -83.50   Median :-265.00  
 Mean   :-134.6   Mean   :103.10   Mean   : -87.85   Mean   : -38.95  
 3rd Qu.:   5.5   3rd Qu.:168.25   3rd Qu.: -27.25   3rd Qu.: 250.50  
 Max.   : 106.0   Max.   :245.00   Max.   :  93.00   Max.   : 750.00  
  magnet_arm_y     magnet_arm_z    kurtosis_roll_arm kurtosis_picth_arm
 Min.   :-307.0   Min.   :-499.0   Mode:logical      Mode:logical      
 1st Qu.: 205.2   1st Qu.: 403.0   NA's:20           NA's:20           
 Median : 291.0   Median : 476.5                                       
 Mean   : 239.4   Mean   : 369.8                                       
 3rd Qu.: 358.8   3rd Qu.: 517.0                                       
 Max.   : 474.0   Max.   : 633.0                                       
 kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
 Mode:logical     Mode:logical      Mode:logical       Mode:logical    
 NA's:20          NA's:20           NA's:20            NA's:20         
                                                                       
                                                                       
                                                                       
                                                                       
 max_roll_arm   max_picth_arm  max_yaw_arm    min_roll_arm   min_pitch_arm 
 Mode:logical   Mode:logical   Mode:logical   Mode:logical   Mode:logical  
 NA's:20        NA's:20        NA's:20        NA's:20        NA's:20       
                                                                           
                                                                           
                                                                           
                                                                           
 min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
 Mode:logical   Mode:logical       Mode:logical        Mode:logical     
 NA's:20        NA's:20            NA's:20             NA's:20          
                                                                        
                                                                        
                                                                        
                                                                        
 roll_dumbbell      pitch_dumbbell    yaw_dumbbell      
 Min.   :-111.118   Min.   :-54.97   Min.   :-103.3200  
 1st Qu.:   7.494   1st Qu.:-51.89   1st Qu.: -75.2809  
 Median :  50.403   Median :-40.81   Median :  -8.2863  
 Mean   :  33.760   Mean   :-19.47   Mean   :  -0.9385  
 3rd Qu.:  58.129   3rd Qu.: 16.12   3rd Qu.:  55.8335  
 Max.   : 123.984   Max.   : 96.87   Max.   : 132.2337  
 kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
 Mode:logical           Mode:logical            Mode:logical         
 NA's:20                NA's:20                 NA's:20              
                                                                     
                                                                     
                                                                     
                                                                     
 skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
 Mode:logical           Mode:logical            Mode:logical         
 NA's:20                NA's:20                 NA's:20              
                                                                     
                                                                     
                                                                     
                                                                     
 max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
 Mode:logical      Mode:logical       Mode:logical     Mode:logical     
 NA's:20           NA's:20            NA's:20          NA's:20          
                                                                        
                                                                        
                                                                        
                                                                        
 min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
 Mode:logical       Mode:logical     Mode:logical           
 NA's:20            NA's:20          NA's:20                
                                                            
                                                            
                                                            
                                                            
 amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
 Mode:logical             Mode:logical           Min.   : 1.0        
 NA's:20                  NA's:20                1st Qu.: 7.0        
                                                 Median :15.5        
                                                 Mean   :17.2        
                                                 3rd Qu.:29.0        
                                                 Max.   :31.0        
 var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
 Mode:logical       Mode:logical      Mode:logical        
 NA's:20            NA's:20           NA's:20             
                                                          
                                                          
                                                          
                                                          
 var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
 Mode:logical      Mode:logical       Mode:logical         
 NA's:20           NA's:20            NA's:20              
                                                           
                                                           
                                                           
                                                           
 var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
 Mode:logical       Mode:logical     Mode:logical        Mode:logical    
 NA's:20            NA's:20          NA's:20             NA's:20         
                                                                         
                                                                         
                                                                         
                                                                         
 gyros_dumbbell_x  gyros_dumbbell_y  gyros_dumbbell_z accel_dumbbell_x 
 Min.   :-1.0300   Min.   :-1.1100   Min.   :-1.180   Min.   :-159.00  
 1st Qu.: 0.1600   1st Qu.:-0.2100   1st Qu.:-0.485   1st Qu.:-140.25  
 Median : 0.3600   Median : 0.0150   Median :-0.280   Median : -19.00  
 Mean   : 0.2690   Mean   : 0.0605   Mean   :-0.266   Mean   : -47.60  
 3rd Qu.: 0.4625   3rd Qu.: 0.1450   3rd Qu.:-0.165   3rd Qu.:  15.75  
 Max.   : 1.0600   Max.   : 1.9100   Max.   : 1.100   Max.   : 185.00  
 accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
 Min.   :-30.00   Min.   :-221.0   Min.   :-576.0    Min.   :-558.0   
 1st Qu.:  5.75   1st Qu.:-192.2   1st Qu.:-528.0    1st Qu.: 259.5   
 Median : 71.50   Median :  -3.0   Median :-508.5    Median : 316.0   
 Mean   : 70.55   Mean   : -60.0   Mean   :-304.2    Mean   : 189.3   
 3rd Qu.:151.25   3rd Qu.:  76.5   3rd Qu.:-317.0    3rd Qu.: 348.2   
 Max.   :166.00   Max.   : 100.0   Max.   : 523.0    Max.   : 403.0   
 magnet_dumbbell_z  roll_forearm     pitch_forearm      yaw_forearm      
 Min.   :-164.00   Min.   :-176.00   Min.   :-63.500   Min.   :-168.000  
 1st Qu.: -33.00   1st Qu.: -40.25   1st Qu.:-11.457   1st Qu.: -93.375  
 Median :  49.50   Median :  94.20   Median :  8.830   Median : -19.250  
 Mean   :  71.40   Mean   :  38.66   Mean   :  7.099   Mean   :   2.195  
 3rd Qu.:  96.25   3rd Qu.: 143.25   3rd Qu.: 28.500   3rd Qu.: 104.500  
 Max.   : 368.00   Max.   : 176.00   Max.   : 59.300   Max.   : 159.000  
 kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
 Mode:logical          Mode:logical           Mode:logical        
 NA's:20               NA's:20                NA's:20             
                                                                  
                                                                  
                                                                  
                                                                  
 skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
 Mode:logical          Mode:logical           Mode:logical        
 NA's:20               NA's:20                NA's:20             
                                                                  
                                                                  
                                                                  
                                                                  
 max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm
 Mode:logical     Mode:logical      Mode:logical    Mode:logical    
 NA's:20          NA's:20           NA's:20         NA's:20         
                                                                    
                                                                    
                                                                    
                                                                    
 min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
 Mode:logical      Mode:logical    Mode:logical          
 NA's:20           NA's:20         NA's:20               
                                                         
                                                         
                                                         
                                                         
 amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
 Mode:logical            Mode:logical          Min.   :21.00      
 NA's:20                 NA's:20               1st Qu.:24.00      
                                               Median :32.50      
                                               Mean   :32.05      
                                               3rd Qu.:36.75      
                                               Max.   :47.00      
 var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
 Mode:logical      Mode:logical     Mode:logical        Mode:logical    
 NA's:20           NA's:20          NA's:20             NA's:20         
                                                                        
                                                                        
                                                                        
                                                                        
 avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
 Mode:logical      Mode:logical         Mode:logical      Mode:logical   
 NA's:20           NA's:20              NA's:20           NA's:20        
                                                                         
                                                                         
                                                                         
                                                                         
 stddev_yaw_forearm var_yaw_forearm gyros_forearm_x   gyros_forearm_y  
 Mode:logical       Mode:logical    Min.   :-1.0600   Min.   :-5.9700  
 NA's:20            NA's:20         1st Qu.:-0.5850   1st Qu.:-1.2875  
                                    Median : 0.0200   Median : 0.0350  
                                    Mean   :-0.0200   Mean   :-0.0415  
                                    3rd Qu.: 0.2925   3rd Qu.: 2.0475  
                                    Max.   : 1.3800   Max.   : 4.2600  
 gyros_forearm_z   accel_forearm_x  accel_forearm_y  accel_forearm_z 
 Min.   :-1.2600   Min.   :-212.0   Min.   :-331.0   Min.   :-282.0  
 1st Qu.:-0.0975   1st Qu.:-114.8   1st Qu.:   8.5   1st Qu.:-199.0  
 Median : 0.2300   Median :  86.0   Median : 138.0   Median :-148.5  
 Mean   : 0.2610   Mean   :  38.8   Mean   : 125.3   Mean   : -93.7  
 3rd Qu.: 0.7625   3rd Qu.: 166.2   3rd Qu.: 268.0   3rd Qu.: -31.0  
 Max.   : 1.8000   Max.   : 232.0   Max.   : 406.0   Max.   : 179.0  
 magnet_forearm_x magnet_forearm_y magnet_forearm_z   problem_id   
 Min.   :-714.0   Min.   :-787.0   Min.   :-32.0    Min.   : 1.00  
 1st Qu.:-427.2   1st Qu.:-328.8   1st Qu.:275.2    1st Qu.: 5.75  
 Median :-189.5   Median : 487.0   Median :491.5    Median :10.50  
 Mean   :-159.2   Mean   : 191.8   Mean   :460.2    Mean   :10.50  
 3rd Qu.:  41.5   3rd Qu.: 720.8   3rd Qu.:661.5    3rd Qu.:15.25  
 Max.   : 532.0   Max.   : 800.0   Max.   :884.0    Max.   :20.00  
 [ reached getOption("max.print") -- omitted 1 row ]
> dim(data)
[1] 19622   160
> dim(validation)
[1]  20 160
> #Remove unnecessary columns
> # first 7 columns don't contain useful info
> data <- data[,-seq(1:7)]
> validation <- validation[,-seq(1:7)]
> 
> #Remove columns with NAs This reduces de amount of predictors to 53
> # select columns that don't have NAs
> indexNA <- as.vector(sapply(data[,1:152],function(x) {length(which(is.na(x)))!=0}))
> data <- data[,!indexNA]
> validation <- validation[,!indexNA]
> # set last (classe) and prior (- classe) column index
> #last <- as.numeric(ncol(data))
> #prior <- last - 1
> 
> # set variables to numerics for correlation check, except the "classe"
> for (i in 1:prior) {
+   data[,i] <- as.numeric(data[,i])
+   validation[,i] <- as.numeric(validation[,i])
+ }
> 
> #check the correlations
> cor.check <- cor(data[, -c(last)])
> diag(cor.check) <- 0 
> plot( levelplot(cor.check, main ="Correlation matrix for all WLE features in training set",
+                 scales=list(x=list(rot=90), cex=1.0)))
> # find the highly correlated variables
> highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)
> # pre process variables
> preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
> dataPrep <- predict(preObj, data[,1:prior])
> dataPrep$classe <- data$classe
> 
> valPrep <-predict(preObj,validation[,1:prior])
> valPrep$problem_id <- validation$problem_id
> # test near zero variance
> myDataNZV <- nearZeroVar(dataPrep, saveMetrics=TRUE)
> if (any(myDataNZV$nzv)) nzv else message("No variables with near zero variance")
No variables with near zero variance
> dataPrep <- dataPrep[,myDataNZV$nzv==FALSE]
> valPrep <- valPrep[,myDataNZV$nzv==FALSE]
> # split dataset into training and test set
> inTrain <- createDataPartition(y=dataPrep$classe, p=0.7, list=FALSE )
> training <- dataPrep[inTrain,]
> testing <- dataPrep[-inTrain,]
> # set seed for reproducibility
> set.seed(12345)
> 
> # get the best mtry
> bestmtry <- tuneRF(training[-last],training$classe, ntreeTry=100, 
+                    stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
mtry = 7  OOB error = 0.65% 
Searching left ...
mtry = 5 	OOB error = 0.72% 
-0.1123596 0.01 
Searching right ...
mtry = 10 	OOB error = 0.55% 
0.1573034 0.01 
mtry = 15 	OOB error = 0.57% 
-0.04 0.01 
> 
> mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]
> 
> # Model 1: RandomForest
> wle.rf <-randomForest(classe~.,data=training, mtry=mtry, ntree=501, 
+                       keep.forest=TRUE, proximity=TRUE, 
+                       importance=TRUE,test=testing)
> # plot the Out of bag error estimates
> layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
> par(mar=c(5,4,4,0)) #No margin on the right side
> plot(wle.rf, log="y", main ="Out-of-bag (OOB) error estimate per Number of Trees")
> par(mar=c(5,0,4,2)) #No margin on the left side
> plot(c(0,1),type="n", axes=F, xlab="", ylab="")
> legend("top", colnames(wle.rf$err.rate),col=1:6,cex=0.8,fill=1:6)
> # plot the accuracy and Gini
> varImpPlot(wle.rf, main="Mean Decrease of Accuracy and Gini per variable")
> # MDSplot (we couldn't execute this due to lack of memory)
> MDSplot(wle.rf, training$classe)
Error: cannot allocate vector of size 1.4 Gb
> # results with training set
> predict1 <- predict(wle.rf, newdata=training)
> confusionMatrix(predict1,training$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 3906    0    0    0    0
         B    0 2658    0    0    0
         C    0    0 2396    0    0
         D    0    0    0 2252    0
         E    0    0    0    0 2525

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9997, 1)
    No Information Rate : 0.2843     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
> #Confusion Matrix and Statistics
> # results with test set
> predict2 <- predict(wle.rf, newdata=testing)
> confusionMatrix(predict2,testing$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673   11    0    0    0
         B    1 1125    8    0    0
         C    0    3 1016    8    1
         D    0    0    2  955    2
         E    0    0    0    1 1079

Overall Statistics
                                          
               Accuracy : 0.9937          
                 95% CI : (0.9913, 0.9956)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.992           
 Mcnemar's Test P-Value : NA              


Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   0.9877   0.9903   0.9907   0.9972
Specificity            0.9974   0.9981   0.9975   0.9992   0.9998
Pos Pred Value         0.9935   0.9921   0.9883   0.9958   0.9991
Neg Pred Value         0.9998   0.9971   0.9979   0.9982   0.9994
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1912   0.1726   0.1623   0.1833
Detection Prevalence   0.2862   0.1927   0.1747   0.1630   0.1835
Balanced Accuracy      0.9984   0.9929   0.9939   0.9949   0.9985
> 
> # Confusion Matrix and Statistics
> #Train Model 2: Decision Tree
> # Model 2: Decision Tree
> dt <- rpart(classe ~ ., data=training, method="class")
> 
> # fancyRpartPlot works for small trees, but not for ours
> fancyRpartPlot(dt)
Warning message:
labs do not fit even at cex 0.15, there may be some overplotting













setwd("C:/Users/Seshan/Desktop")
library(readr)
Weight_lift <- read.csv("Weight lift.csv")
View(Weight_lift)
 str(Weight_lift)
data<-Weight_lift
# load libraries
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lattice)
library(rattle)

library(C50)
#install.package('devtools') # Only needed if you dont have this installed.
library(devtools)
install_github('adam-m-mcelhinney/helpRFunctions')
library(helpRFunctions)
names(data)
dim(data)
pairs(data[1:10])
# enable multi-core processing
library(doParallel)
#cl <- makeCluster(detectCores())
registerDoParallel()
set.seed(12345)
dataTrain<-data[1:3020,]
dataTest<-data[3021:4024,]
head(dataTrain)
head(dataTest)
indexNA <- as.vector(sapply(dataTrain[,1:152],function(x) {length(which(is.na(x)))!=0}))
dataTrain <- dataTrain[,!indexNA]
train_control<- trainControl(method="cv", number=10)

model<- train( pitch_belt ~., data=dataTrain,trControl=train_control, method="rf")
model
# make predictions
predictions<- predict(model,dataTrain)
# append predictions
pred<- cbind(dataTrain,predictions)
# summarize results
confusionMatrix<- confusionMatrix(pred$predictions,pred$pitch_belt)
confusionMatrix

summary(data)
summary(validation)
dim(data)
dim(validation)
#Remove unnecessary columns
# first 7 columns don't contain useful info
data <- data[,-seq(1:7)]
validation <- validation[,-seq(1:7)]

#Remove columns with NAs This reduces de amount of predictors to 53
# select columns that don't have NAs
indexNA <- as.vector(sapply(data[,1:152],function(x) {length(which(is.na(x)))!=0}))
data <- data[,!indexNA]
validation <- validation[,!indexNA]
# set last (classe) and prior (- classe) column index
#last <- as.numeric(ncol(data))
#prior <- last - 1

# set variables to numerics for correlation check, except the "classe"
for (i in 1:prior) {
  data[,i] <- as.numeric(data[,i])
  validation[,i] <- as.numeric(validation[,i])
}

#check the correlations
cor.check <- cor(data[, -c(last)])
diag(cor.check) <- 0 
plot( levelplot(cor.check, main ="Correlation matrix for all WLE features in training set",
                scales=list(x=list(rot=90), cex=1.0)))
# find the highly correlated variables
highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)
# pre process variables
preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
dataPrep <- predict(preObj, data[,1:prior])
dataPrep$classe <- data$classe

valPrep <-predict(preObj,validation[,1:prior])
valPrep$problem_id <- validation$problem_id
# test near zero variance
myDataNZV <- nearZeroVar(dataPrep, saveMetrics=TRUE)
if (any(myDataNZV$nzv)) nzv else message("No variables with near zero variance")
dataPrep <- dataPrep[,myDataNZV$nzv==FALSE]
valPrep <- valPrep[,myDataNZV$nzv==FALSE]
# split dataset into training and test set
inTrain <- createDataPartition(y=dataPrep$classe, p=0.7, list=FALSE )
training <- dataPrep[inTrain,]
testing <- dataPrep[-inTrain,]
# set seed for reproducibility
set.seed(12345)

# get the best mtry
bestmtry <- tuneRF(training[-last],training$classe, ntreeTry=100, 
                   stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)

mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]

# Model 1: RandomForest
wle.rf <-randomForest(classe~.,data=training, mtry=mtry, ntree=501, 
                      keep.forest=TRUE, proximity=TRUE, 
                      importance=TRUE,test=testing)
# plot the Out of bag error estimates
layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(wle.rf, log="y", main ="Out-of-bag (OOB) error estimate per Number of Trees")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(wle.rf$err.rate),col=1:6,cex=0.8,fill=1:6)
# plot the accuracy and Gini
varImpPlot(wle.rf, main="Mean Decrease of Accuracy and Gini per variable")
# MDSplot (we couldn't execute this due to lack of memory)
MDSplot(wle.rf, training$classe)
# results with training set
predict1 <- predict(wle.rf, newdata=training)
confusionMatrix(predict1,training$classe)
#Confusion Matrix and Statistics
# results with test set
predict2 <- predict(wle.rf, newdata=testing)
confusionMatrix(predict2,testing$classe)

# Confusion Matrix and Statistics
#Train Model 2: Decision Tree
# Model 2: Decision Tree
dt <- rpart(classe ~ ., data=training, method="class")

# fancyRpartPlot works for small trees, but not for ours
fancyRpartPlot(dt)

> setwd("C:/Users/Seshan/Desktop")
> library(readr)
> Weight_lift <- read.csv("Weight lift.csv")
> View(Weight_lift)
> str(Weight_lift)
'data.frame':	4024 obs. of  152 variables:
 $ user_name               : Factor w/ 5 levels "adelmo","carlitos",..: 3 3 3 3 3 3 3 3 3 3 ...
 $ raw_timestamp_part_1    : int  1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 1322489729 ...
 $ raw_timestamp_part_2    : int  34670 62641 70653 82654 90637 170626 190665 242723 267551 274689 ...
 $ cvtd_timestamp          : Factor w/ 7 levels "2/12/2011 13:35",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
 $ num_window              : int  1 1 1 1 1 1 1 1 1 1 ...
 $ roll_belt               : num  3.7 3.66 3.58 3.56 3.57 3.45 3.31 2.91 2.31 2 ...
 $ pitch_belt              : num  41.6 42.8 43.7 44.4 45.1 45.6 46.2 46.9 47.4 47.7 ...
 $ yaw_belt                : num  -82.8 -82.5 -82.3 -82.1 -81.9 -81.9 -81.9 -82.2 -82.6 -82.8 ...
 $ total_accel_belt        : int  3 2 1 1 1 1 3 4 2 3 ...
 $ kurtosis_roll_belt      : num  -1.04 -1.04 -1.04 -1.04 -1.04 ...
 $ kurtosis_picth_belt     : num  -0.391 -0.391 -0.391 -0.391 -0.391 ...
 $ skewness_roll_belt      : num  0.00541 0.00541 0.00541 0.00541 0.00541 ...
 $ skewness_roll_belt.1    : num  0.0451 0.0451 0.0451 0.0451 0.0451 ...
 $ max_roll_belt           : num  -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 -4.1 ...
 $ max_picth_belt          : int  20 20 20 20 20 20 20 20 20 20 ...
 $ max_yaw_belt            : num  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
 $ min_roll_belt           : num  -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 -7.25 ...
 $ min_pitch_belt          : int  18 18 18 18 18 18 18 18 18 18 ...
 $ min_yaw_belt            : num  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 ...
 $ amplitude_roll_belt     : num  1.34 1.34 1.34 1.34 1.34 ...
 $ amplitude_pitch_belt    : int  2 2 2 2 2 2 2 2 2 2 ...
 $ amplitude_yaw_belt      : int  0 0 0 0 0 0 0 0 0 0 ...
 $ var_total_accel_belt    : num  0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 ...
 $ avg_roll_belt           : num  122 122 122 122 122 ...
 $ stddev_roll_belt        : num  0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 ...
 $ var_roll_belt           : num  0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 ...
 $ avg_pitch_belt          : num  25.8 25.8 25.8 25.8 25.8 ...
 $ stddev_pitch_belt       : num  0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 ...
 $ var_pitch_belt          : num  0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 ...
 $ avg_yaw_belt            : num  -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 -4.95 ...
 $ stddev_yaw_belt         : num  0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 0.4 ...
 $ var_yaw_belt            : num  0.17 0.17 0.17 0.17 0.17 0.17 0.17 0.17 0.17 0.17 ...
 $ gyros_belt_x            : num  2.02 1.96 1.88 1.8 1.77 1.75 1.78 1.75 1.65 1.48 ...
 $ gyros_belt_y            : num  0.18 0.14 0.08 0.03 0 -0.03 -0.06 -0.06 -0.03 -0.06 ...
 $ gyros_belt_z            : num  0.02 0.05 0.05 0.08 0.13 0.16 0.15 0.23 0.33 0.21 ...
 $ accel_belt_x            : int  -3 -2 -2 -6 -4 1 1 2 -1 -18 ...
 $ accel_belt_y            : int  -18 -13 -6 -5 -9 -9 -24 -36 -19 18 ...
 $ accel_belt_z            : int  22 16 8 7 0 -5 -8 -9 -7 1 ...
 $ magnet_belt_x           : int  387 405 409 422 418 432 438 440 443 449 ...
 $ magnet_belt_y           : int  525 512 511 513 508 510 508 503 507 499 ...
 $ magnet_belt_z           : int  -267 -254 -244 -221 -208 -189 -176 -163 -140 -132 ...
 $ roll_arm                : num  132 129 125 120 115 110 104 98.6 93.2 88.5 ...
 $ pitch_arm               : num  -43.7 -45.3 -46.8 -48.1 -49.1 -49.6 -49.9 -49.7 -49 -48.1 ...
 $ yaw_arm                 : num  -53.6 -49 -43.7 -38.1 -31.7 -25.8 -18.5 -11.4 -4.49 1.82 ...
 $ total_accel_arm         : int  38 38 35 35 34 33 29 28 27 22 ...
 $ var_accel_arm           : num  65.1 65.1 65.1 65.1 65.1 ...
 $ avg_roll_arm            : num  76.2 76.2 76.2 76.2 76.2 ...
 $ stddev_roll_arm         : num  16.1 16.1 16.1 16.1 16.1 ...
 $ var_roll_arm            : num  259 259 259 259 259 ...
 $ avg_pitch_arm           : num  -10.2 -10.2 -10.2 -10.2 -10.2 ...
 $ stddev_pitch_arm        : num  10.7 10.7 10.7 10.7 10.7 ...
 $ var_pitch_arm           : num  114 114 114 114 114 ...
 $ avg_yaw_arm             : num  19.1 19.1 19.1 19.1 19.1 ...
 $ stddev_yaw_arm          : num  35.9 35.9 35.9 35.9 35.9 ...
 $ var_yaw_arm             : num  1287 1287 1287 1287 1287 ...
 $ gyros_arm_x             : num  2.65 2.79 2.91 3.08 3.2 3.31 3.5 3.53 3.4 3.48 ...
 $ gyros_arm_y             : num  -0.61 -0.64 -0.69 -0.72 -0.77 -0.83 -0.83 -0.83 -0.83 -0.8 ...
 $ gyros_arm_z             : num  -0.02 -0.11 -0.15 -0.23 -0.25 -0.3 -0.31 -0.21 -0.11 -0.15 ...
 $ accel_arm_x             : int  143 146 156 158 163 160 165 153 143 135 ...
 $ accel_arm_y             : int  30 35 44 52 55 59 67 70 78 96 ...
 $ accel_arm_z             : int  -346 -339 -307 -305 -288 -274 -225 -218 -205 -134 ...
 $ magnet_arm_x            : int  556 599 613 646 670 696 721 725 740 741 ...
 $ magnet_arm_y            : int  -205 -206 -198 -186 -175 -174 -161 -152 -133 -115 ...
 $ magnet_arm_z            : int  -374 -335 -319 -268 -241 -193 -121 -105 -43 14 ...
 $ kurtosis_roll_arm       : num  -1.18 -1.18 -1.18 -1.18 -1.18 ...
 $ kurtosis_picth_arm      : num  -0.969 -0.969 -0.969 -0.969 -0.969 ...
 $ kurtosis_yaw_arm        : num  -0.87 -0.87 -0.87 -0.87 -0.87 ...
 $ skewness_roll_arm       : num  0.124 0.124 0.124 0.124 0.124 ...
 $ skewness_pitch_arm      : num  -0.103 -0.103 -0.103 -0.103 -0.103 ...
 $ skewness_yaw_arm        : num  0.0598 0.0598 0.0598 0.0598 0.0598 ...
 $ max_roll_arm            : num  8.45 8.45 8.45 8.45 8.45 8.45 8.45 8.45 8.45 8.45 ...
 $ max_picth_arm           : num  77.2 77.2 77.2 77.2 77.2 ...
 $ max_yaw_arm             : int  38 38 38 38 38 38 38 38 38 38 ...
 $ min_roll_arm            : num  -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 -33.6 ...
 $ min_pitch_arm           : num  -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 -58.6 ...
 $ min_yaw_arm             : int  10 10 10 10 10 10 10 10 10 10 ...
 $ amplitude_roll_arm      : num  36.9 36.9 36.9 36.9 36.9 ...
 $ amplitude_pitch_arm     : num  122 122 122 122 122 ...
 $ amplitude_yaw_arm       : int  27 27 27 27 27 27 27 27 27 27 ...
 $ roll_dumbbell           : num  51.2 55.8 55.5 55.9 55.2 ...
 $ pitch_dumbbell          : num  11.7 9.65 6.88 11.08 11.43 ...
 $ yaw_dumbbell            : num  104.3 100.2 101.1 99.8 100.4 ...
 $ kurtosis_roll_dumbbell  : num  -0.0959 -0.0959 -0.0959 -0.0959 -0.0959 ...
 $ kurtosis_picth_dumbbell : num  -0.442 -0.442 -0.442 -0.442 -0.442 ...
 $ skewness_roll_dumbbell  : num  0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 0.0819 ...
 $ skewness_pitch_dumbbell : num  -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 -0.216 ...
 $ max_roll_dumbbell       : num  41.9 41.9 41.9 41.9 41.9 ...
 $ max_picth_dumbbell      : num  133 133 133 133 133 133 133 133 133 133 ...
 $ max_yaw_dumbbell        : num  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 ...
 $ min_roll_dumbbell       : num  -26.8 -26.8 -26.8 -26.8 -26.8 ...
 $ min_pitch_dumbbell      : num  20.2 20.2 20.2 20.2 20.2 20.2 20.2 20.2 20.2 20.2 ...
 $ min_yaw_dumbbell        : num  -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 ...
 $ amplitude_roll_dumbbell : num  55.7 55.7 55.7 55.7 55.7 ...
 $ amplitude_pitch_dumbbell: num  54.7 54.7 54.7 54.7 54.7 ...
 $ amplitude_yaw_dumbbell  : int  0 0 0 0 0 0 0 0 0 0 ...
 $ total_accel_dumbbell    : int  4 4 4 5 4 4 4 4 4 4 ...
 $ var_accel_dumbbell      : num  2.42 2.42 2.42 2.42 2.42 ...
 $ avg_roll_dumbbell       : num  -5.12 -5.12 -5.12 -5.12 -5.12 ...
  [list output truncated]
> data<-Weight_lift
> # load libraries
> library(caret)
Loading required package: lattice
Loading required package: ggplot2
> library(randomForest)
randomForest 4.6-14
Type rfNews() to see new features/changes/bug fixes.

Attaching package: ‘randomForest’

The following object is masked from ‘package:ggplot2’:

    margin

> library(rpart)
> library(rpart.plot)
> library(ggplot2)
> library(lattice)
> library(rattle)
Rattle: A free graphical interface for data science with R.
Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
Type 'rattle()' to shake, rattle, and roll your data.

Attaching package: ‘rattle’

The following object is masked from ‘package:randomForest’:

    importance

> 
> library(C50)
> #install.package('devtools') # Only needed if you dont have this installed.
> library(devtools)
> install_github('adam-m-mcelhinney/helpRFunctions')
Skipping install of 'helpRFunctions' from a github remote, the SHA1 (9eb16e8c) has not changed since last install.
  Use `force = TRUE` to force installation
> library(helpRFunctions)
> names(data)
  [1] "user_name"                "raw_timestamp_part_1"    
  [3] "raw_timestamp_part_2"     "cvtd_timestamp"          
  [5] "new_window"               "num_window"              
  [7] "roll_belt"                "pitch_belt"              
  [9] "yaw_belt"                 "total_accel_belt"        
 [11] "kurtosis_roll_belt"       "kurtosis_picth_belt"     
 [13] "skewness_roll_belt"       "skewness_roll_belt.1"    
 [15] "max_roll_belt"            "max_picth_belt"          
 [17] "max_yaw_belt"             "min_roll_belt"           
 [19] "min_pitch_belt"           "min_yaw_belt"            
 [21] "amplitude_roll_belt"      "amplitude_pitch_belt"    
 [23] "amplitude_yaw_belt"       "var_total_accel_belt"    
 [25] "avg_roll_belt"            "stddev_roll_belt"        
 [27] "var_roll_belt"            "avg_pitch_belt"          
 [29] "stddev_pitch_belt"        "var_pitch_belt"          
 [31] "avg_yaw_belt"             "stddev_yaw_belt"         
 [33] "var_yaw_belt"             "gyros_belt_x"            
 [35] "gyros_belt_y"             "gyros_belt_z"            
 [37] "accel_belt_x"             "accel_belt_y"            
 [39] "accel_belt_z"             "magnet_belt_x"           
 [41] "magnet_belt_y"            "magnet_belt_z"           
 [43] "roll_arm"                 "pitch_arm"               
 [45] "yaw_arm"                  "total_accel_arm"         
 [47] "var_accel_arm"            "avg_roll_arm"            
 [49] "stddev_roll_arm"          "var_roll_arm"            
 [51] "avg_pitch_arm"            "stddev_pitch_arm"        
 [53] "var_pitch_arm"            "avg_yaw_arm"             
 [55] "stddev_yaw_arm"           "var_yaw_arm"             
 [57] "gyros_arm_x"              "gyros_arm_y"             
 [59] "gyros_arm_z"              "accel_arm_x"             
 [61] "accel_arm_y"              "accel_arm_z"             
 [63] "magnet_arm_x"             "magnet_arm_y"            
 [65] "magnet_arm_z"             "kurtosis_roll_arm"       
 [67] "kurtosis_picth_arm"       "kurtosis_yaw_arm"        
 [69] "skewness_roll_arm"        "skewness_pitch_arm"      
 [71] "skewness_yaw_arm"         "max_roll_arm"            
 [73] "max_picth_arm"            "max_yaw_arm"             
 [75] "min_roll_arm"             "min_pitch_arm"           
 [77] "min_yaw_arm"              "amplitude_roll_arm"      
 [79] "amplitude_pitch_arm"      "amplitude_yaw_arm"       
 [81] "roll_dumbbell"            "pitch_dumbbell"          
 [83] "yaw_dumbbell"             "kurtosis_roll_dumbbell"  
 [85] "kurtosis_picth_dumbbell"  "skewness_roll_dumbbell"  
 [87] "skewness_pitch_dumbbell"  "max_roll_dumbbell"       
 [89] "max_picth_dumbbell"       "max_yaw_dumbbell"        
 [91] "min_roll_dumbbell"        "min_pitch_dumbbell"      
 [93] "min_yaw_dumbbell"         "amplitude_roll_dumbbell" 
 [95] "amplitude_pitch_dumbbell" "amplitude_yaw_dumbbell"  
 [97] "total_accel_dumbbell"     "var_accel_dumbbell"      
 [99] "avg_roll_dumbbell"        "stddev_roll_dumbbell"    
[101] "var_roll_dumbbell"        "avg_pitch_dumbbell"      
[103] "stddev_pitch_dumbbell"    "var_pitch_dumbbell"      
[105] "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"     
[107] "var_yaw_dumbbell"         "gyros_dumbbell_x"        
[109] "gyros_dumbbell_y"         "gyros_dumbbell_z"        
[111] "accel_dumbbell_x"         "accel_dumbbell_y"        
[113] "accel_dumbbell_z"         "magnet_dumbbell_x"       
[115] "magnet_dumbbell_y"        "magnet_dumbbell_z"       
[117] "roll_forearm"             "pitch_forearm"           
[119] "yaw_forearm"              "kurtosis_roll_forearm"   
[121] "kurtosis_picth_forearm"   "skewness_roll_forearm"   
[123] "skewness_pitch_forearm"   "max_roll_forearm"        
[125] "max_picth_forearm"        "max_yaw_forearm"         
[127] "min_roll_forearm"         "min_pitch_forearm"       
[129] "min_yaw_forearm"          "amplitude_roll_forearm"  
[131] "amplitude_pitch_forearm"  "amplitude_yaw_forearm"   
[133] "total_accel_forearm"      "var_accel_forearm"       
[135] "avg_roll_forearm"         "stddev_roll_forearm"     
[137] "var_roll_forearm"         "avg_pitch_forearm"       
[139] "stddev_pitch_forearm"     "var_pitch_forearm"       
[141] "avg_yaw_forearm"          "stddev_yaw_forearm"      
[143] "var_yaw_forearm"          "gyros_forearm_x"         
[145] "gyros_forearm_y"          "gyros_forearm_z"         
[147] "accel_forearm_x"          "accel_forearm_y"         
[149] "accel_forearm_z"          "magnet_forearm_x"        
[151] "magnet_forearm_y"         "magnet_forearm_z"        
> dim(data)
[1] 4024  152
> pairs(data[1:10])
> # enable multi-core processing
> library(doParallel)
Loading required package: foreach
Loading required package: iterators
Loading required package: parallel
> #cl <- makeCluster(detectCores())

registerDoParallel()
> set.seed(12345)
> dataTrain<-data[1:3020,]
> dataTest<-data[3021:4024,]
> head(dataTrain)
  user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
1    eurico           1322489729                34670 28/11/2011 14:15
2    eurico           1322489729                62641 28/11/2011 14:15
3    eurico           1322489729                70653 28/11/2011 14:15
4    eurico           1322489729                82654 28/11/2011 14:15
5    eurico           1322489729                90637 28/11/2011 14:15
6    eurico           1322489729               170626 28/11/2011 14:15
  new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
1         no          1      3.70       41.6    -82.8                3
2         no          1      3.66       42.8    -82.5                2
3         no          1      3.58       43.7    -82.3                1
4         no          1      3.56       44.4    -82.1                1
5         no          1      3.57       45.1    -81.9                1
6         no          1      3.45       45.6    -81.9                1
  kurtosis_roll_belt kurtosis_picth_belt skewness_roll_belt
1           -1.03566            -0.39133           0.005406
2           -1.03566            -0.39133           0.005406
3           -1.03566            -0.39133           0.005406
4           -1.03566            -0.39133           0.005406
5           -1.03566            -0.39133           0.005406
6           -1.03566            -0.39133           0.005406
  skewness_roll_belt.1 max_roll_belt max_picth_belt max_yaw_belt
1             0.045115          -4.1             20           -1
2             0.045115          -4.1             20           -1
3             0.045115          -4.1             20           -1
4             0.045115          -4.1             20           -1
5             0.045115          -4.1             20           -1
6             0.045115          -4.1             20           -1
  min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt
1         -7.25             18           -1               1.345
2         -7.25             18           -1               1.345
3         -7.25             18           -1               1.345
4         -7.25             18           -1               1.345
5         -7.25             18           -1               1.345
6         -7.25             18           -1               1.345
  amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt
1                    2                  0                  0.3
2                    2                  0                  0.3
3                    2                  0                  0.3
4                    2                  0                  0.3
5                    2                  0                  0.3
6                    2                  0                  0.3
  avg_roll_belt stddev_roll_belt var_roll_belt avg_pitch_belt
1         121.9              0.6          0.35          25.75
2         121.9              0.6          0.35          25.75
3         121.9              0.6          0.35          25.75
4         121.9              0.6          0.35          25.75
5         121.9              0.6          0.35          25.75
6         121.9              0.6          0.35          25.75
  stddev_pitch_belt var_pitch_belt avg_yaw_belt stddev_yaw_belt
1              0.35            0.1        -4.95             0.4
2              0.35            0.1        -4.95             0.4
3              0.35            0.1        -4.95             0.4
4              0.35            0.1        -4.95             0.4
5              0.35            0.1        -4.95             0.4
6              0.35            0.1        -4.95             0.4
  var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z accel_belt_x
1         0.17         2.02         0.18         0.02           -3
2         0.17         1.96         0.14         0.05           -2
3         0.17         1.88         0.08         0.05           -2
4         0.17         1.80         0.03         0.08           -6
5         0.17         1.77         0.00         0.13           -4
6         0.17         1.75        -0.03         0.16            1
  accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y magnet_belt_z
1          -18           22           387           525          -267
2          -13           16           405           512          -254
3           -6            8           409           511          -244
4           -5            7           422           513          -221
5           -9            0           418           508          -208
6           -9           -5           432           510          -189
  roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm avg_roll_arm
1      132     -43.7   -53.6              38       65.0977     76.22175
2      129     -45.3   -49.0              38       65.0977     76.22175
3      125     -46.8   -43.7              35       65.0977     76.22175
4      120     -48.1   -38.1              35       65.0977     76.22175
5      115     -49.1   -31.7              34       65.0977     76.22175
6      110     -49.6   -25.8              33       65.0977     76.22175
  stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm var_pitch_arm
1         16.1039     259.3599      -10.1695         10.66725      113.7978
2         16.1039     259.3599      -10.1695         10.66725      113.7978
3         16.1039     259.3599      -10.1695         10.66725      113.7978
4         16.1039     259.3599      -10.1695         10.66725      113.7978
5         16.1039     259.3599      -10.1695         10.66725      113.7978
6         16.1039     259.3599      -10.1695         10.66725      113.7978
  avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x gyros_arm_y
1     19.0615        35.8809    1287.463        2.65       -0.61
2     19.0615        35.8809    1287.463        2.79       -0.64
3     19.0615        35.8809    1287.463        2.91       -0.69
4     19.0615        35.8809    1287.463        3.08       -0.72
5     19.0615        35.8809    1287.463        3.20       -0.77
6     19.0615        35.8809    1287.463        3.31       -0.83
  gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y
1       -0.02         143          30        -346          556         -205
2       -0.11         146          35        -339          599         -206
3       -0.15         156          44        -307          613         -198
4       -0.23         158          52        -305          646         -186
5       -0.25         163          55        -288          670         -175
6       -0.30         160          59        -274          696         -174
  magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm kurtosis_yaw_arm
1         -374          -1.18224           -0.96912         -0.86977
2         -335          -1.18224           -0.96912         -0.86977
3         -319          -1.18224           -0.96912         -0.86977
4         -268          -1.18224           -0.96912         -0.86977
5         -241          -1.18224           -0.96912         -0.86977
6         -193          -1.18224           -0.96912         -0.86977
  skewness_roll_arm skewness_pitch_arm skewness_yaw_arm max_roll_arm
1           0.12353           -0.10319         0.059765         8.45
2           0.12353           -0.10319         0.059765         8.45
3           0.12353           -0.10319         0.059765         8.45
4           0.12353           -0.10319         0.059765         8.45
5           0.12353           -0.10319         0.059765         8.45
6           0.12353           -0.10319         0.059765         8.45
  max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm min_yaw_arm
1         77.25          38        -33.6         -58.6          10
2         77.25          38        -33.6         -58.6          10
3         77.25          38        -33.6         -58.6          10
4         77.25          38        -33.6         -58.6          10
5         77.25          38        -33.6         -58.6          10
6         77.25          38        -33.6         -58.6          10
  amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell
1             36.945               121.5                27      51.23554
2             36.945               121.5                27      55.82442
3             36.945               121.5                27      55.46983
4             36.945               121.5                27      55.94486
5             36.945               121.5                27      55.21174
6             36.945               121.5                27      54.24731
  pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
1      11.698847    104.26473               -0.09595
2       9.645819    100.22805               -0.09595
3       6.875244    101.08411               -0.09595
4      11.079297     99.78456               -0.09595
5      11.426833    100.42258               -0.09595
6      14.126636    100.61574               -0.09595
  kurtosis_picth_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
1                 -0.4422                 0.0819                  -0.216
2                 -0.4422                 0.0819                  -0.216
3                 -0.4422                 0.0819                  -0.216
4                 -0.4422                 0.0819                  -0.216
5                 -0.4422                 0.0819                  -0.216
6                 -0.4422                 0.0819                  -0.216
  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
1             41.85                133             -0.1            -26.75
2             41.85                133             -0.1            -26.75
3             41.85                133             -0.1            -26.75
4             41.85                133             -0.1            -26.75
5             41.85                133             -0.1            -26.75
6             41.85                133             -0.1            -26.75
  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
1               20.2             -0.1                   55.71
2               20.2             -0.1                   55.71
3               20.2             -0.1                   55.71
4               20.2             -0.1                   55.71
5               20.2             -0.1                   55.71
6               20.2             -0.1                   55.71
  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
1                    54.74                      0                    4
2                    54.74                      0                    4
3                    54.74                      0                    4
4                    54.74                      0                    5
5                    54.74                      0                    4
6                    54.74                      0                    4
  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
1            2.41635          -5.11805               17.058
2            2.41635          -5.11805               17.058
3            2.41635          -5.11805               17.058
4            2.41635          -5.11805               17.058
5            2.41635          -5.11805               17.058
6            2.41635          -5.11805               17.058
  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
1           291.001            13.9312               14.1062
2           291.001            13.9312               14.1062
3           291.001            13.9312               14.1062
4           291.001            13.9312               14.1062
5           291.001            13.9312               14.1062
6           291.001            13.9312               14.1062
  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
1           199.0775          64.7063             13.5747         184.5578
2           199.0775          64.7063             13.5747         184.5578
3           199.0775          64.7063             13.5747         184.5578
4           199.0775          64.7063             13.5747         184.5578
5           199.0775          64.7063             13.5747         184.5578
6           199.0775          64.7063             13.5747         184.5578
  gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
1            -0.31             0.16             0.08                5
2            -0.31             0.14             0.07                4
3            -0.31             0.16             0.05                3
4            -0.31             0.16             0.07                5
5            -0.31             0.14             0.07                5
6            -0.31             0.14             0.07                6
  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
1               21               37              -471               191
2               22               35              -472               184
3               23               37              -468               190
4               24               38              -469               184
5               23               37              -468               189
6               22               36              -473               188
  magnet_dumbbell_z roll_forearm pitch_forearm yaw_forearm
1               277         -111          26.5         138
2               281         -112          26.2         138
3               275         -114          26.0         137
4               285         -115          25.8         137
5               292         -117          25.5         137
6               278         -118          25.1         137
  kurtosis_roll_forearm kurtosis_picth_forearm skewness_roll_forearm
1              -1.09475               -0.97525              -0.05065
2              -1.09475               -0.97525              -0.05065
3              -1.09475               -0.97525              -0.05065
4              -1.09475               -0.97525              -0.05065
5              -1.09475               -0.97525              -0.05065
6              -1.09475               -0.97525              -0.05065
  skewness_pitch_forearm max_roll_forearm max_picth_forearm max_yaw_forearm
1                0.17285             49.6               168            -1.1
2                0.17285             49.6               168            -1.1
3                0.17285             49.6               168            -1.1
4                0.17285             49.6               168            -1.1
5                0.17285             49.6               168            -1.1
6                0.17285             49.6               168            -1.1
  min_roll_forearm min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
1             4.65            -168.5            -1.1                   32.2
2             4.65            -168.5            -1.1                   32.2
3             4.65            -168.5            -1.1                   32.2
4             4.65            -168.5            -1.1                   32.2
5             4.65            -168.5            -1.1                   32.2
6             4.65            -168.5            -1.1                   32.2
  amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
1                   341.5                     0                  30
2                   341.5                     0                  31
3                   341.5                     0                  32
4                   341.5                     0                  33
5                   341.5                     0                  34
6                   341.5                     0                  36
  var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
1           14.0772         27.85936            45.16342         2749.163
2           14.0772         27.85936            45.16342         2749.163
3           14.0772         27.85936            45.16342         2749.163
4           14.0772         27.85936            45.16342         2749.163
5           14.0772         27.85936            45.16342         2749.163
6           14.0772         27.85936            45.16342         2749.163
  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
1          25.35597             8.906695          79.33451        17.09505
2          25.35597             8.906695          79.33451        17.09505
3          25.35597             8.906695          79.33451        17.09505
4          25.35597             8.906695          79.33451        17.09505
5          25.35597             8.906695          79.33451        17.09505
6          25.35597             8.906695          79.33451        17.09505
  stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
1           74.27584        5541.956           -0.05           -0.37
2           74.27584        5541.956           -0.06           -0.37
3           74.27584        5541.956           -0.05           -0.27
4           74.27584        5541.956            0.02           -0.24
5           74.27584        5541.956            0.08           -0.27
6           74.27584        5541.956            0.14           -0.29
  gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
1           -0.43            -170             155             184
2           -0.59            -178             164             182
3           -0.72            -182             172             185
4           -0.79            -185             182             188
5           -0.82            -188             195             188
6           -0.82            -208             207             190
  magnet_forearm_x magnet_forearm_y magnet_forearm_z
1            -1160             1400             -876
2            -1150             1410             -871
3            -1130             1400             -863
4            -1120             1400             -855
5            -1100             1400             -843
6            -1090             1400             -838
> head(dataTest)
     user_name raw_timestamp_part_1 raw_timestamp_part_2  cvtd_timestamp
3021     pedro           1323094996               656284 5/12/2011 14:23
3022     pedro           1323094996               664357 5/12/2011 14:23
3023     pedro           1323094996               672361 5/12/2011 14:23
3024     pedro           1323094996               692335 5/12/2011 14:23
3025     pedro           1323094996               700442 5/12/2011 14:23
3026     pedro           1323094996               712340 5/12/2011 14:23
     new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
3021         no         69       125       25.9    -3.24               20
3022         no         69       125       25.8    -3.13               20
3023         no         69       125       25.8    -3.07               20
3024         no         69       125       25.7    -3.01               20
3025         no         69       125       25.7    -2.96               20
3026         no         69       125       25.6    -2.80               20
     kurtosis_roll_belt kurtosis_picth_belt skewness_roll_belt
3021           -1.03566            -0.39133           0.005406
3022           -1.03566            -0.39133           0.005406
3023           -1.03566            -0.39133           0.005406
3024           -1.03566            -0.39133           0.005406
3025           -1.03566            -0.39133           0.005406
3026           -1.03566            -0.39133           0.005406
     skewness_roll_belt.1 max_roll_belt max_picth_belt max_yaw_belt
3021             0.045115          -4.1             20           -1
3022             0.045115          -4.1             20           -1
3023             0.045115          -4.1             20           -1
3024             0.045115          -4.1             20           -1
3025             0.045115          -4.1             20           -1
3026             0.045115          -4.1             20           -1
     min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt
3021         -7.25             18           -1               1.345
3022         -7.25             18           -1               1.345
3023         -7.25             18           -1               1.345
3024         -7.25             18           -1               1.345
3025         -7.25             18           -1               1.345
3026         -7.25             18           -1               1.345
     amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt
3021                    2                  0                  0.3
3022                    2                  0                  0.3
3023                    2                  0                  0.3
3024                    2                  0                  0.3
3025                    2                  0                  0.3
3026                    2                  0                  0.3
     avg_roll_belt stddev_roll_belt var_roll_belt avg_pitch_belt
3021         121.9              0.6          0.35          25.75
3022         121.9              0.6          0.35          25.75
3023         121.9              0.6          0.35          25.75
3024         121.9              0.6          0.35          25.75
3025         121.9              0.6          0.35          25.75
3026         121.9              0.6          0.35          25.75
     stddev_pitch_belt var_pitch_belt avg_yaw_belt stddev_yaw_belt
3021              0.35            0.1        -4.95             0.4
3022              0.35            0.1        -4.95             0.4
3023              0.35            0.1        -4.95             0.4
3024              0.35            0.1        -4.95             0.4
3025              0.35            0.1        -4.95             0.4
3026              0.35            0.1        -4.95             0.4
     var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z accel_belt_x
3021         0.17        -0.37        -0.02        -0.43          -40
3022         0.17        -0.37        -0.02        -0.43          -42
3023         0.17        -0.39        -0.02        -0.44          -43
3024         0.17        -0.39        -0.02        -0.44          -42
3025         0.17        -0.39        -0.02        -0.44          -40
3026         0.17        -0.39        -0.02        -0.43          -42
     accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y magnet_belt_z
3021           71         -175             2           579          -375
3022           72         -178            -2           581          -390
3023           71         -179             5           579          -387
3024           69         -178             4           575          -389
3025           68         -177             5           584          -368
3026           70         -177             7           575          -389
     roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm avg_roll_arm
3021    -24.7      6.42    -115               6       65.0977     76.22175
3022    -25.0      6.61    -115               5       65.0977     76.22175
3023    -25.2      6.71    -116               5       65.0977     76.22175
3024    -25.5      6.70    -116               4       65.0977     76.22175
3025    -25.6      6.61    -116               4       65.0977     76.22175
3026    -25.5      6.43    -117               3       65.0977     76.22175
     stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
3021         16.1039     259.3599      -10.1695         10.66725
3022         16.1039     259.3599      -10.1695         10.66725
3023         16.1039     259.3599      -10.1695         10.66725
3024         16.1039     259.3599      -10.1695         10.66725
3025         16.1039     259.3599      -10.1695         10.66725
3026         16.1039     259.3599      -10.1695         10.66725
     var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
3021      113.7978     19.0615        35.8809    1287.463        0.47
3022      113.7978     19.0615        35.8809    1287.463        0.34
3023      113.7978     19.0615        35.8809    1287.463        0.22
3024      113.7978     19.0615        35.8809    1287.463        0.14
3025      113.7978     19.0615        35.8809    1287.463        0.10
3026      113.7978     19.0615        35.8809    1287.463        0.03
     gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
3021       -0.37       -0.13          -4          14          56
3022       -0.27       -0.31          -2          15          49
3023       -0.19       -0.44           6           6          49
3024       -0.13       -0.49           7           3          42
3025       -0.10       -0.44           7           4          37
3026       -0.10       -0.39           5           3          33
     magnet_arm_x magnet_arm_y magnet_arm_z kurtosis_roll_arm
3021         -378          385          448          -1.18224
3022         -389          377          454          -1.18224
3023         -392          372          461          -1.18224
3024         -391          369          463          -1.18224
3025         -395          369          453          -1.18224
3026         -399          360          461          -1.18224
     kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm
3021           -0.96912         -0.86977           0.12353
3022           -0.96912         -0.86977           0.12353
3023           -0.96912         -0.86977           0.12353
3024           -0.96912         -0.86977           0.12353
3025           -0.96912         -0.86977           0.12353
3026           -0.96912         -0.86977           0.12353
     skewness_pitch_arm skewness_yaw_arm max_roll_arm max_picth_arm
3021           -0.10319         0.059765         8.45         77.25
3022           -0.10319         0.059765         8.45         77.25
3023           -0.10319         0.059765         8.45         77.25
3024           -0.10319         0.059765         8.45         77.25
3025           -0.10319         0.059765         8.45         77.25
3026           -0.10319         0.059765         8.45         77.25
     max_yaw_arm min_roll_arm min_pitch_arm min_yaw_arm amplitude_roll_arm
3021          38        -33.6         -58.6          10             36.945
3022          38        -33.6         -58.6          10             36.945
3023          38        -33.6         -58.6          10             36.945
3024          38        -33.6         -58.6          10             36.945
3025          38        -33.6         -58.6          10             36.945
3026          38        -33.6         -58.6          10             36.945
     amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell pitch_dumbbell
3021               121.5                27      2.239617       27.20759
3022               121.5                27     -2.321713       27.02507
3023               121.5                27      1.132156       27.51712
3024               121.5                27     -6.717451       30.68508
3025               121.5                27    -11.203061       29.49277
3026               121.5                27     -4.579686       31.41632
     yaw_dumbbell kurtosis_roll_dumbbell kurtosis_picth_dumbbell
3021     129.7753               -0.09595                 -0.4422
3022     129.9500               -0.09595                 -0.4422
3023     129.5380               -0.09595                 -0.4422
3024     125.6207               -0.09595                 -0.4422
3025     125.4100               -0.09595                 -0.4422
3026     125.3085               -0.09595                 -0.4422
     skewness_roll_dumbbell skewness_pitch_dumbbell max_roll_dumbbell
3021                 0.0819                  -0.216             41.85
3022                 0.0819                  -0.216             41.85
3023                 0.0819                  -0.216             41.85
3024                 0.0819                  -0.216             41.85
3025                 0.0819                  -0.216             41.85
3026                 0.0819                  -0.216             41.85
     max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
3021                133             -0.1            -26.75
3022                133             -0.1            -26.75
3023                133             -0.1            -26.75
3024                133             -0.1            -26.75
3025                133             -0.1            -26.75
3026                133             -0.1            -26.75
     min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
3021               20.2             -0.1                   55.71
3022               20.2             -0.1                   55.71
3023               20.2             -0.1                   55.71
3024               20.2             -0.1                   55.71
3025               20.2             -0.1                   55.71
3026               20.2             -0.1                   55.71
     amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
3021                    54.74                      0                    9
3022                    54.74                      0                    9
3023                    54.74                      0                    9
3024                    54.74                      0                    9
3025                    54.74                      0                    9
3026                    54.74                      0                    9
     var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
3021            2.41635          -5.11805               17.058
3022            2.41635          -5.11805               17.058
3023            2.41635          -5.11805               17.058
3024            2.41635          -5.11805               17.058
3025            2.41635          -5.11805               17.058
3026            2.41635          -5.11805               17.058
     var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
3021           291.001            13.9312               14.1062
3022           291.001            13.9312               14.1062
3023           291.001            13.9312               14.1062
3024           291.001            13.9312               14.1062
3025           291.001            13.9312               14.1062
3026           291.001            13.9312               14.1062
     var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell
3021           199.0775          64.7063             13.5747
3022           199.0775          64.7063             13.5747
3023           199.0775          64.7063             13.5747
3024           199.0775          64.7063             13.5747
3025           199.0775          64.7063             13.5747
3026           199.0775          64.7063             13.5747
     var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y gyros_dumbbell_z
3021         184.5578             0.47            -0.19            -0.13
3022         184.5578             0.47            -0.19            -0.15
3023         184.5578             0.47            -0.22            -0.16
3024         184.5578             0.50            -0.21            -0.18
3025         184.5578             0.50            -0.18            -0.18
3026         184.5578             0.48            -0.14            -0.16
     accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x
3021               24                2               86               529
3022               23               -2               83               540
3023               24                1               85               533
3024               27               -6               85               534
3025               26              -10               85               537
3026               27               -4               83               542
     magnet_dumbbell_y magnet_dumbbell_z roll_forearm pitch_forearm
3021              -521               -91          115          16.3
3022              -516               -99          115          17.2
3023              -518               -92          116          18.1
3024              -506               -90          116          19.2
3025              -512               -88          117          20.5
3026              -517               -94          117          21.9
     yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
3021        87.7              -1.09475               -0.97525
3022        89.0              -1.09475               -0.97525
3023        90.4              -1.09475               -0.97525
3024        91.9              -1.09475               -0.97525
3025        93.4              -1.09475               -0.97525
3026        94.8              -1.09475               -0.97525
     skewness_roll_forearm skewness_pitch_forearm max_roll_forearm
3021              -0.05065                0.17285             49.6
3022              -0.05065                0.17285             49.6
3023              -0.05065                0.17285             49.6
3024              -0.05065                0.17285             49.6
3025              -0.05065                0.17285             49.6
3026              -0.05065                0.17285             49.6
     max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
3021               168            -1.1             4.65            -168.5
3022               168            -1.1             4.65            -168.5
3023               168            -1.1             4.65            -168.5
3024               168            -1.1             4.65            -168.5
3025               168            -1.1             4.65            -168.5
3026               168            -1.1             4.65            -168.5
     min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
3021            -1.1                   32.2                   341.5
3022            -1.1                   32.2                   341.5
3023            -1.1                   32.2                   341.5
3024            -1.1                   32.2                   341.5
3025            -1.1                   32.2                   341.5
3026            -1.1                   32.2                   341.5
     amplitude_yaw_forearm total_accel_forearm var_accel_forearm
3021                     0                  36           14.0772
3022                     0                  36           14.0772
3023                     0                  37           14.0772
3024                     0                  37           14.0772
3025                     0                  37           14.0772
3026                     0                  38           14.0772
     avg_roll_forearm stddev_roll_forearm var_roll_forearm
3021         27.85936            45.16342         2749.163
3022         27.85936            45.16342         2749.163
3023         27.85936            45.16342         2749.163
3024         27.85936            45.16342         2749.163
3025         27.85936            45.16342         2749.163
3026         27.85936            45.16342         2749.163
     avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm
3021          25.35597             8.906695          79.33451
3022          25.35597             8.906695          79.33451
3023          25.35597             8.906695          79.33451
3024          25.35597             8.906695          79.33451
3025          25.35597             8.906695          79.33451
3026          25.35597             8.906695          79.33451
     avg_yaw_forearm stddev_yaw_forearm var_yaw_forearm gyros_forearm_x
3021        17.09505           74.27584        5541.956            0.42
3022        17.09505           74.27584        5541.956            0.50
3023        17.09505           74.27584        5541.956            0.55
3024        17.09505           74.27584        5541.956            0.58
3025        17.09505           74.27584        5541.956            0.51
3026        17.09505           74.27584        5541.956            0.40
     gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
3021           -0.87           -0.02             105             293
3022           -0.90            0.07              80             300
3023           -1.01            0.08              79             316
3024           -1.24            0.03              60             317
3025           -1.32           -0.10              42             317
3026           -1.30           -0.23              40             325
     accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
3021            -164             -275              791              694
3022            -161             -290              792              678
3023            -164             -300              794              682
3024            -169             -322              794              672
3025            -171             -341              787              676
3026            -170             -343              785              681
> train_control<- trainControl(method="cv", number=10)
> 
> model<- train( pitch_belt ~., data=dataTrain,trControl=train_control, method="rf")

Random Forest 

3020 samples
 151 predictor

No pre-processing
Resampling: Cross-Validated (10 fold) 
Summary of sample sizes: 2718, 2718, 2716, 2719, 2718, 2720, ... 
Resampling results across tuning parameters:

  mtry  RMSE       Rsquared   MAE      
    2   3.7719395  0.9858292  2.0072729
   80   0.4378226  0.9994475  0.1698078
  159   0.5446360  0.9991612  0.1890224

RMSE was used to select the optimal model using the smallest value.
The final value used for the model was mtry = 80.

> summary(data)
       X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
 Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
 1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
 Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
 Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
 3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
 Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
          cvtd_timestamp  new_window    num_window      roll_belt     
 28/11/2011 14:14: 1498   no :19216   Min.   :  1.0   Min.   :-28.90  
 05/12/2011 11:24: 1497   yes:  406   1st Qu.:222.0   1st Qu.:  1.10  
 30/11/2011 17:11: 1440               Median :424.0   Median :113.00  
 05/12/2011 11:25: 1425               Mean   :430.6   Mean   : 64.41  
 02/12/2011 14:57: 1380               3rd Qu.:644.0   3rd Qu.:123.00  
 02/12/2011 13:34: 1375               Max.   :864.0   Max.   :162.00  
   pitch_belt          yaw_belt       total_accel_belt kurtosis_roll_belt
 Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00    Min.   :-2.121    
 1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00    1st Qu.:-1.329    
 Median :  5.2800   Median : -13.00   Median :17.00    Median :-0.899    
 Mean   :  0.3053   Mean   : -11.21   Mean   :11.31    Mean   :-0.220    
 3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00    3rd Qu.:-0.219    
 Max.   : 60.3000   Max.   : 179.00   Max.   :29.00    Max.   :33.000    
 kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
 Min.   :-2.190      Mode:logical      Min.   :-5.745    
 1st Qu.:-1.107      NA's:19622        1st Qu.:-0.444    
 Median :-0.151                        Median : 0.000    
 Mean   : 4.334                        Mean   :-0.026    
 3rd Qu.: 3.178                        3rd Qu.: 0.417    
 Max.   :58.000                        Max.   : 3.595    
 skewness_roll_belt.1 skewness_yaw_belt max_roll_belt     max_picth_belt 
 Min.   :-7.616       Mode:logical      Min.   :-94.300   Min.   : 3.00  
 1st Qu.:-1.114       NA's:19622        1st Qu.:-88.000   1st Qu.: 5.00  
 Median :-0.068                         Median : -5.100   Median :18.00  
 Mean   :-0.296                         Mean   : -6.667   Mean   :12.92  
 3rd Qu.: 0.661                         3rd Qu.: 18.500   3rd Qu.:19.00  
 Max.   : 7.348                         Max.   :180.000   Max.   :30.00  
  max_yaw_belt   min_roll_belt     min_pitch_belt   min_yaw_belt  
 Min.   :-2.10   Min.   :-180.00   Min.   : 0.00   Min.   :-2.10  
 1st Qu.:-1.30   1st Qu.: -88.40   1st Qu.: 3.00   1st Qu.:-1.30  
 Median :-0.90   Median :  -7.85   Median :16.00   Median :-0.90  
 Mean   :-0.22   Mean   : -10.44   Mean   :10.76   Mean   :-0.22  
 3rd Qu.:-0.20   3rd Qu.:   9.05   3rd Qu.:17.00   3rd Qu.:-0.20  
 Max.   :33.00   Max.   : 173.00   Max.   :23.00   Max.   :33.00  
 amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
 Min.   :  0.000     Min.   : 0.000       Min.   :0         
 1st Qu.:  0.300     1st Qu.: 1.000       1st Qu.:0         
 Median :  1.000     Median : 1.000       Median :0         
 Mean   :  3.769     Mean   : 2.167       Mean   :0         
 3rd Qu.:  2.083     3rd Qu.: 2.000       3rd Qu.:0         
 Max.   :360.000     Max.   :12.000       Max.   :0         
 var_total_accel_belt avg_roll_belt    stddev_roll_belt var_roll_belt    
 Min.   : 0.000       Min.   :-27.40   Min.   : 0.000   Min.   :  0.000  
 1st Qu.: 0.100       1st Qu.:  1.10   1st Qu.: 0.200   1st Qu.:  0.000  
 Median : 0.200       Median :116.35   Median : 0.400   Median :  0.100  
 Mean   : 0.926       Mean   : 68.06   Mean   : 1.337   Mean   :  7.699  
 3rd Qu.: 0.300       3rd Qu.:123.38   3rd Qu.: 0.700   3rd Qu.:  0.500  
 Max.   :16.500       Max.   :157.40   Max.   :14.200   Max.   :200.700  
 avg_pitch_belt    stddev_pitch_belt var_pitch_belt    avg_yaw_belt     
 Min.   :-51.400   Min.   :0.000     Min.   : 0.000   Min.   :-138.300  
 1st Qu.:  2.025   1st Qu.:0.200     1st Qu.: 0.000   1st Qu.: -88.175  
 Median :  5.200   Median :0.400     Median : 0.100   Median :  -6.550  
 Mean   :  0.520   Mean   :0.603     Mean   : 0.766   Mean   :  -8.831  
 3rd Qu.: 15.775   3rd Qu.:0.700     3rd Qu.: 0.500   3rd Qu.:  14.125  
 Max.   : 59.700   Max.   :4.000     Max.   :16.200   Max.   : 173.500  
 stddev_yaw_belt    var_yaw_belt        gyros_belt_x      
 Min.   :  0.000   Min.   :    0.000   Min.   :-1.040000  
 1st Qu.:  0.100   1st Qu.:    0.010   1st Qu.:-0.030000  
 Median :  0.300   Median :    0.090   Median : 0.030000  
 Mean   :  1.341   Mean   :  107.487   Mean   :-0.005592  
 3rd Qu.:  0.700   3rd Qu.:    0.475   3rd Qu.: 0.110000  
 Max.   :176.600   Max.   :31183.240   Max.   : 2.220000  
  gyros_belt_y       gyros_belt_z      accel_belt_x       accel_belt_y   
 Min.   :-0.64000   Min.   :-1.4600   Min.   :-120.000   Min.   :-69.00  
 1st Qu.: 0.00000   1st Qu.:-0.2000   1st Qu.: -21.000   1st Qu.:  3.00  
 Median : 0.02000   Median :-0.1000   Median : -15.000   Median : 35.00  
 Mean   : 0.03959   Mean   :-0.1305   Mean   :  -5.595   Mean   : 30.15  
 3rd Qu.: 0.11000   3rd Qu.:-0.0200   3rd Qu.:  -5.000   3rd Qu.: 61.00  
 Max.   : 0.64000   Max.   : 1.6200   Max.   :  85.000   Max.   :164.00  
  accel_belt_z     magnet_belt_x   magnet_belt_y   magnet_belt_z   
 Min.   :-275.00   Min.   :-52.0   Min.   :354.0   Min.   :-623.0  
 1st Qu.:-162.00   1st Qu.:  9.0   1st Qu.:581.0   1st Qu.:-375.0  
 Median :-152.00   Median : 35.0   Median :601.0   Median :-320.0  
 Mean   : -72.59   Mean   : 55.6   Mean   :593.7   Mean   :-345.5  
 3rd Qu.:  27.00   3rd Qu.: 59.0   3rd Qu.:610.0   3rd Qu.:-306.0  
 Max.   : 105.00   Max.   :485.0   Max.   :673.0   Max.   : 293.0  
    roll_arm         pitch_arm          yaw_arm          total_accel_arm
 Min.   :-180.00   Min.   :-88.800   Min.   :-180.0000   Min.   : 1.00  
 1st Qu.: -31.77   1st Qu.:-25.900   1st Qu.: -43.1000   1st Qu.:17.00  
 Median :   0.00   Median :  0.000   Median :   0.0000   Median :27.00  
 Mean   :  17.83   Mean   : -4.612   Mean   :  -0.6188   Mean   :25.51  
 3rd Qu.:  77.30   3rd Qu.: 11.200   3rd Qu.:  45.8750   3rd Qu.:33.00  
 Max.   : 180.00   Max.   : 88.500   Max.   : 180.0000   Max.   :66.00  
 var_accel_arm     avg_roll_arm     stddev_roll_arm    var_roll_arm      
 Min.   :  0.00   Min.   :-166.67   Min.   :  0.000   Min.   :    0.000  
 1st Qu.:  9.03   1st Qu.: -38.37   1st Qu.:  1.376   1st Qu.:    1.898  
 Median : 40.61   Median :   0.00   Median :  5.702   Median :   32.517  
 Mean   : 53.23   Mean   :  12.68   Mean   : 11.201   Mean   :  417.264  
 3rd Qu.: 75.62   3rd Qu.:  76.33   3rd Qu.: 14.921   3rd Qu.:  222.647  
 Max.   :331.70   Max.   : 163.33   Max.   :161.964   Max.   :26232.208  
 avg_pitch_arm     stddev_pitch_arm var_pitch_arm       avg_yaw_arm      
 Min.   :-81.773   Min.   : 0.000   Min.   :   0.000   Min.   :-173.440  
 1st Qu.:-22.770   1st Qu.: 1.642   1st Qu.:   2.697   1st Qu.: -29.198  
 Median :  0.000   Median : 8.133   Median :  66.146   Median :   0.000  
 Mean   : -4.901   Mean   :10.383   Mean   : 195.864   Mean   :   2.359  
 3rd Qu.:  8.277   3rd Qu.:16.327   3rd Qu.: 266.576   3rd Qu.:  38.185  
 Max.   : 75.659   Max.   :43.412   Max.   :1884.565   Max.   : 152.000  
 stddev_yaw_arm     var_yaw_arm         gyros_arm_x        gyros_arm_y     
 Min.   :  0.000   Min.   :    0.000   Min.   :-6.37000   Min.   :-3.4400  
 1st Qu.:  2.577   1st Qu.:    6.642   1st Qu.:-1.33000   1st Qu.:-0.8000  
 Median : 16.682   Median :  278.309   Median : 0.08000   Median :-0.2400  
 Mean   : 22.270   Mean   : 1055.933   Mean   : 0.04277   Mean   :-0.2571  
 3rd Qu.: 35.984   3rd Qu.: 1294.850   3rd Qu.: 1.57000   3rd Qu.: 0.1400  
 Max.   :177.044   Max.   :31344.568   Max.   : 4.87000   Max.   : 2.8400  
  gyros_arm_z       accel_arm_x       accel_arm_y      accel_arm_z     
 Min.   :-2.3300   Min.   :-404.00   Min.   :-318.0   Min.   :-636.00  
 1st Qu.:-0.0700   1st Qu.:-242.00   1st Qu.: -54.0   1st Qu.:-143.00  
 Median : 0.2300   Median : -44.00   Median :  14.0   Median : -47.00  
 Mean   : 0.2695   Mean   : -60.24   Mean   :  32.6   Mean   : -71.25  
 3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.0   3rd Qu.:  23.00  
 Max.   : 3.0200   Max.   : 437.00   Max.   : 308.0   Max.   : 292.00  
  magnet_arm_x     magnet_arm_y     magnet_arm_z    kurtosis_roll_arm
 Min.   :-584.0   Min.   :-392.0   Min.   :-597.0   Min.   :-1.809   
 1st Qu.:-300.0   1st Qu.:  -9.0   1st Qu.: 131.2   1st Qu.:-1.345   
 Median : 289.0   Median : 202.0   Median : 444.0   Median :-0.894   
 Mean   : 191.7   Mean   : 156.6   Mean   : 306.5   Mean   :-0.366   
 3rd Qu.: 637.0   3rd Qu.: 323.0   3rd Qu.: 545.0   3rd Qu.:-0.038   
 Max.   : 782.0   Max.   : 583.0   Max.   : 694.0   Max.   :21.456   
 kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
 Min.   :-2.084     Min.   :-2.103   Min.   :-2.541    Min.   :-4.565    
 1st Qu.:-1.280     1st Qu.:-1.220   1st Qu.:-0.561    1st Qu.:-0.618    
 Median :-1.010     Median :-0.733   Median : 0.040    Median :-0.035    
 Mean   :-0.542     Mean   : 0.406   Mean   : 0.068    Mean   :-0.065    
 3rd Qu.:-0.379     3rd Qu.: 0.115   3rd Qu.: 0.671    3rd Qu.: 0.454    
 Max.   :19.751     Max.   :56.000   Max.   : 4.394    Max.   : 3.043    
 skewness_yaw_arm  max_roll_arm     max_picth_arm       max_yaw_arm   
 Min.   :-6.708   Min.   :-73.100   Min.   :-173.000   Min.   : 4.00  
 1st Qu.:-0.743   1st Qu.: -0.175   1st Qu.:  -1.975   1st Qu.:29.00  
 Median :-0.133   Median :  4.950   Median :  23.250   Median :34.00  
 Mean   :-0.229   Mean   : 11.236   Mean   :  35.751   Mean   :35.46  
 3rd Qu.: 0.344   3rd Qu.: 26.775   3rd Qu.:  95.975   3rd Qu.:41.00  
 Max.   : 7.483   Max.   : 85.500   Max.   : 180.000   Max.   :65.00  
  min_roll_arm    min_pitch_arm      min_yaw_arm    amplitude_roll_arm
 Min.   :-89.10   Min.   :-180.00   Min.   : 1.00   Min.   :  0.000   
 1st Qu.:-41.98   1st Qu.: -72.62   1st Qu.: 8.00   1st Qu.:  5.425   
 Median :-22.45   Median : -33.85   Median :13.00   Median : 28.450   
 Mean   :-21.22   Mean   : -33.92   Mean   :14.66   Mean   : 32.452   
 3rd Qu.:  0.00   3rd Qu.:   0.00   3rd Qu.:19.00   3rd Qu.: 50.960   
 Max.   : 66.40   Max.   : 152.00   Max.   :38.00   Max.   :119.500   
 amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell     pitch_dumbbell   
 Min.   :  0.000     Min.   : 0.00     Min.   :-153.71   Min.   :-149.59  
 1st Qu.:  9.925     1st Qu.:13.00     1st Qu.: -18.49   1st Qu.: -40.89  
 Median : 54.900     Median :22.00     Median :  48.17   Median : -20.96  
 Mean   : 69.677     Mean   :20.79     Mean   :  23.84   Mean   : -10.78  
 3rd Qu.:115.175     3rd Qu.:28.75     3rd Qu.:  67.61   3rd Qu.:  17.50  
 Max.   :360.000     Max.   :52.00     Max.   : 153.55   Max.   : 149.40  
  yaw_dumbbell      kurtosis_roll_dumbbell kurtosis_picth_dumbbell
 Min.   :-150.871   Min.   :-2.174         Min.   :-2.200         
 1st Qu.: -77.644   1st Qu.:-0.682         1st Qu.:-0.721         
 Median :  -3.324   Median :-0.033         Median :-0.133         
 Mean   :   1.674   Mean   : 0.452         Mean   : 0.286         
 3rd Qu.:  79.643   3rd Qu.: 0.940         3rd Qu.: 0.584         
 Max.   : 154.952   Max.   :54.998         Max.   :55.628         
 kurtosis_yaw_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
 Mode:logical          Min.   :-7.384         Min.   :-7.447         
 NA's:19622            1st Qu.:-0.581         1st Qu.:-0.526         
                       Median :-0.076         Median :-0.091         
                       Mean   :-0.115         Mean   :-0.035         
                       3rd Qu.: 0.400         3rd Qu.: 0.505         
                       Max.   : 1.958         Max.   : 3.769         
 skewness_yaw_dumbbell max_roll_dumbbell max_picth_dumbbell
 Mode:logical          Min.   :-70.10    Min.   :-112.90   
 NA's:19622            1st Qu.:-27.15    1st Qu.: -66.70   
                       Median : 14.85    Median :  40.05   
                       Mean   : 13.76    Mean   :  32.75   
                       3rd Qu.: 50.58    3rd Qu.: 133.22   
                       Max.   :137.00    Max.   : 155.00   
 max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell min_yaw_dumbbell
 Min.   :-2.20    Min.   :-149.60   Min.   :-147.00    Min.   :-2.20   
 1st Qu.:-0.70    1st Qu.: -59.67   1st Qu.: -91.80    1st Qu.:-0.70   
 Median : 0.00    Median : -43.55   Median : -66.15    Median : 0.00   
 Mean   : 0.45    Mean   : -41.24   Mean   : -33.18    Mean   : 0.45   
 3rd Qu.: 0.90    3rd Qu.: -25.20   3rd Qu.:  21.20    3rd Qu.: 0.90   
 Max.   :55.00    Max.   :  73.20   Max.   : 120.90    Max.   :55.00   
 amplitude_roll_dumbbell amplitude_pitch_dumbbell amplitude_yaw_dumbbell
 Min.   :  0.00          Min.   :  0.00           Min.   :0             
 1st Qu.: 14.97          1st Qu.: 17.06           1st Qu.:0             
 Median : 35.05          Median : 41.73           Median :0             
 Mean   : 55.00          Mean   : 65.93           Mean   :0             
 3rd Qu.: 81.04          3rd Qu.: 99.55           3rd Qu.:0             
 Max.   :256.48          Max.   :273.59           Max.   :0             
 total_accel_dumbbell var_accel_dumbbell avg_roll_dumbbell
 Min.   : 0.00        Min.   :  0.000    Min.   :-128.96  
 1st Qu.: 4.00        1st Qu.:  0.378    1st Qu.: -12.33  
 Median :10.00        Median :  1.000    Median :  48.23  
 Mean   :13.72        Mean   :  4.388    Mean   :  23.86  
 3rd Qu.:19.00        3rd Qu.:  3.434    3rd Qu.:  64.37  
 Max.   :58.00        Max.   :230.428    Max.   : 125.99  
 stddev_roll_dumbbell var_roll_dumbbell  avg_pitch_dumbbell
 Min.   :  0.000      Min.   :    0.00   Min.   :-70.73    
 1st Qu.:  4.639      1st Qu.:   21.52   1st Qu.:-42.00    
 Median : 12.204      Median :  148.95   Median :-19.91    
 Mean   : 20.761      Mean   : 1020.27   Mean   :-12.33    
 3rd Qu.: 26.356      3rd Qu.:  694.65   3rd Qu.: 13.21    
 Max.   :123.778      Max.   :15321.01   Max.   : 94.28    
 stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell  
 Min.   : 0.000        Min.   :   0.00    Min.   :-117.950  
 1st Qu.: 3.482        1st Qu.:  12.12    1st Qu.: -76.696  
 Median : 8.089        Median :  65.44    Median :  -4.505  
 Mean   :13.147        Mean   : 350.31    Mean   :   0.202  
 3rd Qu.:19.238        3rd Qu.: 370.11    3rd Qu.:  71.234  
 Max.   :82.680        Max.   :6836.02    Max.   : 134.905  
 stddev_yaw_dumbbell var_yaw_dumbbell   gyros_dumbbell_x   
 Min.   :  0.000     Min.   :    0.00   Min.   :-204.0000  
 1st Qu.:  3.885     1st Qu.:   15.09   1st Qu.:  -0.0300  
 Median : 10.264     Median :  105.35   Median :   0.1300  
 Mean   : 16.647     Mean   :  589.84   Mean   :   0.1611  
 3rd Qu.: 24.674     3rd Qu.:  608.79   3rd Qu.:   0.3500  
 Max.   :107.088     Max.   :11467.91   Max.   :   2.2200  
 gyros_dumbbell_y   gyros_dumbbell_z  accel_dumbbell_x  accel_dumbbell_y 
 Min.   :-2.10000   Min.   : -2.380   Min.   :-419.00   Min.   :-189.00  
 1st Qu.:-0.14000   1st Qu.: -0.310   1st Qu.: -50.00   1st Qu.:  -8.00  
 Median : 0.03000   Median : -0.130   Median :  -8.00   Median :  41.50  
 Mean   : 0.04606   Mean   : -0.129   Mean   : -28.62   Mean   :  52.63  
 3rd Qu.: 0.21000   3rd Qu.:  0.030   3rd Qu.:  11.00   3rd Qu.: 111.00  
 Max.   :52.00000   Max.   :317.000   Max.   : 235.00   Max.   : 315.00  
 accel_dumbbell_z  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
 Min.   :-334.00   Min.   :-643.0    Min.   :-3600     Min.   :-262.00  
 1st Qu.:-142.00   1st Qu.:-535.0    1st Qu.:  231     1st Qu.: -45.00  
 Median :  -1.00   Median :-479.0    Median :  311     Median :  13.00  
 Mean   : -38.32   Mean   :-328.5    Mean   :  221     Mean   :  46.05  
 3rd Qu.:  38.00   3rd Qu.:-304.0    3rd Qu.:  390     3rd Qu.:  95.00  
 Max.   : 318.00   Max.   : 592.0    Max.   :  633     Max.   : 452.00  
  roll_forearm       pitch_forearm     yaw_forearm     
 Min.   :-180.0000   Min.   :-72.50   Min.   :-180.00  
 1st Qu.:  -0.7375   1st Qu.:  0.00   1st Qu.: -68.60  
 Median :  21.7000   Median :  9.24   Median :   0.00  
 Mean   :  33.8265   Mean   : 10.71   Mean   :  19.21  
 3rd Qu.: 140.0000   3rd Qu.: 28.40   3rd Qu.: 110.00  
 Max.   : 180.0000   Max.   : 89.80   Max.   : 180.00  
 kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
 Min.   :-1.879        Min.   :-2.098         Mode:logical        
 1st Qu.:-1.398        1st Qu.:-1.376         NA's:19622          
 Median :-1.119        Median :-0.890                             
 Mean   :-0.689        Mean   : 0.419                             
 3rd Qu.:-0.618        3rd Qu.: 0.054                             
 Max.   :40.060        Max.   :33.626                             
 skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
 Min.   :-2.297        Min.   :-5.241         Mode:logical        
 1st Qu.:-0.402        1st Qu.:-0.881         NA's:19622          
 Median : 0.003        Median :-0.156                             
 Mean   :-0.009        Mean   :-0.223                             
 3rd Qu.: 0.370        3rd Qu.: 0.514                             
 Max.   : 5.856        Max.   : 4.464                             
 max_roll_forearm max_picth_forearm max_yaw_forearm  min_roll_forearm 
 Min.   :-66.60   Min.   :-151.00   Min.   :-1.900   Min.   :-72.500  
 1st Qu.:  0.00   1st Qu.:   0.00   1st Qu.:-1.400   1st Qu.: -6.075  
 Median : 26.80   Median : 113.00   Median :-1.100   Median :  0.000  
 Mean   : 24.49   Mean   :  81.49   Mean   :-0.689   Mean   : -0.167  
 3rd Qu.: 45.95   3rd Qu.: 174.75   3rd Qu.:-0.600   3rd Qu.: 12.075  
 Max.   : 89.80   Max.   : 180.00   Max.   :40.100   Max.   : 62.100  
 min_pitch_forearm min_yaw_forearm  amplitude_roll_forearm
 Min.   :-180.00   Min.   :-1.900   Min.   :  0.000       
 1st Qu.:-175.00   1st Qu.:-1.400   1st Qu.:  1.125       
 Median : -61.00   Median :-1.100   Median : 17.770       
 Mean   : -57.57   Mean   :-0.689   Mean   : 24.653       
 3rd Qu.:   0.00   3rd Qu.:-0.600   3rd Qu.: 39.875       
 Max.   : 167.00   Max.   :40.100   Max.   :126.000       
 amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
 Min.   :  0.0           Min.   :0             Min.   :  0.00     
 1st Qu.:  2.0           1st Qu.:0             1st Qu.: 29.00     
 Median : 83.7           Median :0             Median : 36.00     
 Mean   :139.1           Mean   :0             Mean   : 34.72     
 3rd Qu.:350.0           3rd Qu.:0             3rd Qu.: 41.00     
 Max.   :360.0           Max.   :0             Max.   :108.00     
 var_accel_forearm avg_roll_forearm   stddev_roll_forearm
 Min.   :  0.000   Min.   :-177.234   Min.   :  0.000    
 1st Qu.:  6.759   1st Qu.:  -0.909   1st Qu.:  0.428    
 Median : 21.165   Median :  11.172   Median :  8.030    
 Mean   : 33.502   Mean   :  33.165   Mean   : 41.986    
 3rd Qu.: 51.240   3rd Qu.: 107.132   3rd Qu.: 85.373    
 Max.   :172.606   Max.   : 177.256   Max.   :179.171    
 var_roll_forearm   avg_pitch_forearm stddev_pitch_forearm
 Min.   :    0.00   Min.   :-68.17    Min.   : 0.000      
 1st Qu.:    0.18   1st Qu.:  0.00    1st Qu.: 0.336      
 Median :   64.48   Median : 12.02    Median : 5.516      
 Mean   : 5274.10   Mean   : 11.79    Mean   : 7.977      
 3rd Qu.: 7289.08   3rd Qu.: 28.48    3rd Qu.:12.866      
 Max.   :32102.24   Max.   : 72.09    Max.   :47.745      
 var_pitch_forearm  avg_yaw_forearm   stddev_yaw_forearm var_yaw_forearm   
 Min.   :   0.000   Min.   :-155.06   Min.   :  0.000    Min.   :    0.00  
 1st Qu.:   0.113   1st Qu.: -26.26   1st Qu.:  0.524    1st Qu.:    0.27  
 Median :  30.425   Median :   0.00   Median : 24.743    Median :  612.21  
 Mean   : 139.593   Mean   :  18.00   Mean   : 44.854    Mean   : 4639.85  
 3rd Qu.: 165.532   3rd Qu.:  85.79   3rd Qu.: 85.817    3rd Qu.: 7368.41  
 Max.   :2279.617   Max.   : 169.24   Max.   :197.508    Max.   :39009.33  
 gyros_forearm_x   gyros_forearm_y     gyros_forearm_z    accel_forearm_x  
 Min.   :-22.000   Min.   : -7.02000   Min.   : -8.0900   Min.   :-498.00  
 1st Qu.: -0.220   1st Qu.: -1.46000   1st Qu.: -0.1800   1st Qu.:-178.00  
 Median :  0.050   Median :  0.03000   Median :  0.0800   Median : -57.00  
 Mean   :  0.158   Mean   :  0.07517   Mean   :  0.1512   Mean   : -61.65  
 3rd Qu.:  0.560   3rd Qu.:  1.62000   3rd Qu.:  0.4900   3rd Qu.:  76.00  
 Max.   :  3.970   Max.   :311.00000   Max.   :231.0000   Max.   : 477.00  
 accel_forearm_y  accel_forearm_z   magnet_forearm_x  magnet_forearm_y
 Min.   :-632.0   Min.   :-446.00   Min.   :-1280.0   Min.   :-896.0  
 1st Qu.:  57.0   1st Qu.:-182.00   1st Qu.: -616.0   1st Qu.:   2.0  
 Median : 201.0   Median : -39.00   Median : -378.0   Median : 591.0  
 Mean   : 163.7   Mean   : -55.29   Mean   : -312.6   Mean   : 380.1  
 3rd Qu.: 312.0   3rd Qu.:  26.00   3rd Qu.:  -73.0   3rd Qu.: 737.0  
 Max.   : 923.0   Max.   : 291.00   Max.   :  672.0   Max.   :1480.0  
 magnet_forearm_z classe  
 Min.   :-973.0   A:5580  
 1st Qu.: 191.0   B:3797  
 Median : 511.0   C:3422  
 Mean   : 393.6   D:3216  
 3rd Qu.: 653.0   E:3607  
 Max.   :1090.0           
 [ reached getOption("max.print") -- omitted 1 row ]
> summary(validation)
       X            user_name raw_timestamp_part_1 raw_timestamp_part_2
 Min.   : 1.00   adelmo  :1   Min.   :1.322e+09    Min.   : 36553      
 1st Qu.: 5.75   carlitos:3   1st Qu.:1.323e+09    1st Qu.:268655      
 Median :10.50   charles :1   Median :1.323e+09    Median :530706      
 Mean   :10.50   eurico  :4   Mean   :1.323e+09    Mean   :512167      
 3rd Qu.:15.25   jeremy  :8   3rd Qu.:1.323e+09    3rd Qu.:787738      
 Max.   :20.00   pedro   :3   Max.   :1.323e+09    Max.   :920315      
          cvtd_timestamp new_window   num_window      roll_belt       
 30/11/2011 17:11:4      no:20      Min.   : 48.0   Min.   : -5.9200  
 05/12/2011 11:24:3                 1st Qu.:250.0   1st Qu.:  0.9075  
 30/11/2011 17:12:3                 Median :384.5   Median :  1.1100  
 05/12/2011 14:23:2                 Mean   :379.6   Mean   : 31.3055  
 28/11/2011 14:14:2                 3rd Qu.:467.0   3rd Qu.: 32.5050  
 02/12/2011 13:33:1                 Max.   :859.0   Max.   :129.0000  
   pitch_belt         yaw_belt      total_accel_belt kurtosis_roll_belt
 Min.   :-41.600   Min.   :-93.70   Min.   : 2.00    Mode:logical      
 1st Qu.:  3.013   1st Qu.:-88.62   1st Qu.: 3.00    NA's:20           
 Median :  4.655   Median :-87.85   Median : 4.00                      
 Mean   :  5.824   Mean   :-59.30   Mean   : 7.55                      
 3rd Qu.:  6.135   3rd Qu.:-63.50   3rd Qu.: 8.00                      
 Max.   : 27.800   Max.   :162.00   Max.   :21.00                      
 kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
 Mode:logical        Mode:logical      Mode:logical      
 NA's:20             NA's:20           NA's:20           
                                                         
                                                         
                                                         
                                                         
 skewness_roll_belt.1 skewness_yaw_belt max_roll_belt  max_picth_belt
 Mode:logical         Mode:logical      Mode:logical   Mode:logical  
 NA's:20              NA's:20           NA's:20        NA's:20       
                                                                     
                                                                     
                                                                     
                                                                     
 max_yaw_belt   min_roll_belt  min_pitch_belt min_yaw_belt  
 Mode:logical   Mode:logical   Mode:logical   Mode:logical  
 NA's:20        NA's:20        NA's:20        NA's:20       
                                                            
                                                            
                                                            
                                                            
 amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
 Mode:logical        Mode:logical         Mode:logical      
 NA's:20             NA's:20              NA's:20           
                                                            
                                                            
                                                            
                                                            
 var_total_accel_belt avg_roll_belt  stddev_roll_belt var_roll_belt 
 Mode:logical         Mode:logical   Mode:logical     Mode:logical  
 NA's:20              NA's:20        NA's:20          NA's:20       
                                                                    
                                                                    
                                                                    
                                                                    
 avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt  
 Mode:logical   Mode:logical      Mode:logical   Mode:logical  
 NA's:20        NA's:20           NA's:20        NA's:20       
                                                               
                                                               
                                                               
                                                               
 stddev_yaw_belt var_yaw_belt    gyros_belt_x     gyros_belt_y   
 Mode:logical    Mode:logical   Min.   :-0.500   Min.   :-0.050  
 NA's:20         NA's:20        1st Qu.:-0.070   1st Qu.:-0.005  
                                Median : 0.020   Median : 0.000  
                                Mean   :-0.045   Mean   : 0.010  
                                3rd Qu.: 0.070   3rd Qu.: 0.020  
                                Max.   : 0.240   Max.   : 0.110  
  gyros_belt_z      accel_belt_x     accel_belt_y     accel_belt_z    
 Min.   :-0.4800   Min.   :-48.00   Min.   :-16.00   Min.   :-187.00  
 1st Qu.:-0.1375   1st Qu.:-19.00   1st Qu.:  2.00   1st Qu.: -24.00  
 Median :-0.0250   Median :-13.00   Median :  4.50   Median :  27.00  
 Mean   :-0.1005   Mean   :-13.50   Mean   : 18.35   Mean   : -17.60  
 3rd Qu.: 0.0000   3rd Qu.: -8.75   3rd Qu.: 25.50   3rd Qu.:  38.25  
 Max.   : 0.0500   Max.   : 46.00   Max.   : 72.00   Max.   :  49.00  
 magnet_belt_x    magnet_belt_y   magnet_belt_z       roll_arm      
 Min.   :-13.00   Min.   :566.0   Min.   :-426.0   Min.   :-137.00  
 1st Qu.:  5.50   1st Qu.:578.5   1st Qu.:-398.5   1st Qu.:   0.00  
 Median : 33.50   Median :600.5   Median :-313.5   Median :   0.00  
 Mean   : 35.15   Mean   :601.5   Mean   :-346.9   Mean   :  16.42  
 3rd Qu.: 46.25   3rd Qu.:631.2   3rd Qu.:-305.0   3rd Qu.:  71.53  
 Max.   :169.00   Max.   :638.0   Max.   :-291.0   Max.   : 152.00  
   pitch_arm          yaw_arm        total_accel_arm var_accel_arm 
 Min.   :-63.800   Min.   :-167.00   Min.   : 3.00   Mode:logical  
 1st Qu.: -9.188   1st Qu.: -60.15   1st Qu.:20.25   NA's:20       
 Median :  0.000   Median :   0.00   Median :29.50                 
 Mean   : -3.950   Mean   :  -2.80   Mean   :26.40                 
 3rd Qu.:  3.465   3rd Qu.:  25.50   3rd Qu.:33.25                 
 Max.   : 55.000   Max.   : 178.00   Max.   :44.00                 
 avg_roll_arm   stddev_roll_arm var_roll_arm   avg_pitch_arm 
 Mode:logical   Mode:logical    Mode:logical   Mode:logical  
 NA's:20        NA's:20         NA's:20        NA's:20       
                                                             
                                                             
                                                             
                                                             
 stddev_pitch_arm var_pitch_arm  avg_yaw_arm    stddev_yaw_arm
 Mode:logical     Mode:logical   Mode:logical   Mode:logical  
 NA's:20          NA's:20        NA's:20        NA's:20       
                                                              
                                                              
                                                              
                                                              
 var_yaw_arm     gyros_arm_x      gyros_arm_y       gyros_arm_z     
 Mode:logical   Min.   :-3.710   Min.   :-2.0900   Min.   :-0.6900  
 NA's:20        1st Qu.:-0.645   1st Qu.:-0.6350   1st Qu.:-0.1800  
                Median : 0.020   Median :-0.0400   Median :-0.0250  
                Mean   : 0.077   Mean   :-0.1595   Mean   : 0.1205  
                3rd Qu.: 1.248   3rd Qu.: 0.2175   3rd Qu.: 0.5650  
                Max.   : 3.660   Max.   : 1.8500   Max.   : 1.1300  
  accel_arm_x      accel_arm_y      accel_arm_z       magnet_arm_x    
 Min.   :-341.0   Min.   :-65.00   Min.   :-404.00   Min.   :-428.00  
 1st Qu.:-277.0   1st Qu.: 52.25   1st Qu.:-128.50   1st Qu.:-373.75  
 Median :-194.5   Median :112.00   Median : -83.50   Median :-265.00  
 Mean   :-134.6   Mean   :103.10   Mean   : -87.85   Mean   : -38.95  
 3rd Qu.:   5.5   3rd Qu.:168.25   3rd Qu.: -27.25   3rd Qu.: 250.50  
 Max.   : 106.0   Max.   :245.00   Max.   :  93.00   Max.   : 750.00  
  magnet_arm_y     magnet_arm_z    kurtosis_roll_arm kurtosis_picth_arm
 Min.   :-307.0   Min.   :-499.0   Mode:logical      Mode:logical      
 1st Qu.: 205.2   1st Qu.: 403.0   NA's:20           NA's:20           
 Median : 291.0   Median : 476.5                                       
 Mean   : 239.4   Mean   : 369.8                                       
 3rd Qu.: 358.8   3rd Qu.: 517.0                                       
 Max.   : 474.0   Max.   : 633.0                                       
 kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
 Mode:logical     Mode:logical      Mode:logical       Mode:logical    
 NA's:20          NA's:20           NA's:20            NA's:20         
                                                                       
                                                                       
                                                                       
                                                                       
 max_roll_arm   max_picth_arm  max_yaw_arm    min_roll_arm   min_pitch_arm 
 Mode:logical   Mode:logical   Mode:logical   Mode:logical   Mode:logical  
 NA's:20        NA's:20        NA's:20        NA's:20        NA's:20       
                                                                           
                                                                           
                                                                           
                                                                           
 min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
 Mode:logical   Mode:logical       Mode:logical        Mode:logical     
 NA's:20        NA's:20            NA's:20             NA's:20          
                                                                        
                                                                        
                                                                        
                                                                        
 roll_dumbbell      pitch_dumbbell    yaw_dumbbell      
 Min.   :-111.118   Min.   :-54.97   Min.   :-103.3200  
 1st Qu.:   7.494   1st Qu.:-51.89   1st Qu.: -75.2809  
 Median :  50.403   Median :-40.81   Median :  -8.2863  
 Mean   :  33.760   Mean   :-19.47   Mean   :  -0.9385  
 3rd Qu.:  58.129   3rd Qu.: 16.12   3rd Qu.:  55.8335  
 Max.   : 123.984   Max.   : 96.87   Max.   : 132.2337  
 kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
 Mode:logical           Mode:logical            Mode:logical         
 NA's:20                NA's:20                 NA's:20              
                                                                     
                                                                     
                                                                     
                                                                     
 skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
 Mode:logical           Mode:logical            Mode:logical         
 NA's:20                NA's:20                 NA's:20              
                                                                     
                                                                     
                                                                     
                                                                     
 max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
 Mode:logical      Mode:logical       Mode:logical     Mode:logical     
 NA's:20           NA's:20            NA's:20          NA's:20          
                                                                        
                                                                        
                                                                        
                                                                        
 min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
 Mode:logical       Mode:logical     Mode:logical           
 NA's:20            NA's:20          NA's:20                
                                                            
                                                            
                                                            
                                                            
 amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
 Mode:logical             Mode:logical           Min.   : 1.0        
 NA's:20                  NA's:20                1st Qu.: 7.0        
                                                 Median :15.5        
                                                 Mean   :17.2        
                                                 3rd Qu.:29.0        
                                                 Max.   :31.0        
 var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
 Mode:logical       Mode:logical      Mode:logical        
 NA's:20            NA's:20           NA's:20             
                                                          
                                                          
                                                          
                                                          
 var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
 Mode:logical      Mode:logical       Mode:logical         
 NA's:20           NA's:20            NA's:20              
                                                           
                                                           
                                                           
                                                           
 var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
 Mode:logical       Mode:logical     Mode:logical        Mode:logical    
 NA's:20            NA's:20          NA's:20             NA's:20         
                                                                         
                                                                         
                                                                         
                                                                         
 gyros_dumbbell_x  gyros_dumbbell_y  gyros_dumbbell_z accel_dumbbell_x 
 Min.   :-1.0300   Min.   :-1.1100   Min.   :-1.180   Min.   :-159.00  
 1st Qu.: 0.1600   1st Qu.:-0.2100   1st Qu.:-0.485   1st Qu.:-140.25  
 Median : 0.3600   Median : 0.0150   Median :-0.280   Median : -19.00  
 Mean   : 0.2690   Mean   : 0.0605   Mean   :-0.266   Mean   : -47.60  
 3rd Qu.: 0.4625   3rd Qu.: 0.1450   3rd Qu.:-0.165   3rd Qu.:  15.75  
 Max.   : 1.0600   Max.   : 1.9100   Max.   : 1.100   Max.   : 185.00  
 accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
 Min.   :-30.00   Min.   :-221.0   Min.   :-576.0    Min.   :-558.0   
 1st Qu.:  5.75   1st Qu.:-192.2   1st Qu.:-528.0    1st Qu.: 259.5   
 Median : 71.50   Median :  -3.0   Median :-508.5    Median : 316.0   
 Mean   : 70.55   Mean   : -60.0   Mean   :-304.2    Mean   : 189.3   
 3rd Qu.:151.25   3rd Qu.:  76.5   3rd Qu.:-317.0    3rd Qu.: 348.2   
 Max.   :166.00   Max.   : 100.0   Max.   : 523.0    Max.   : 403.0   
 magnet_dumbbell_z  roll_forearm     pitch_forearm      yaw_forearm      
 Min.   :-164.00   Min.   :-176.00   Min.   :-63.500   Min.   :-168.000  
 1st Qu.: -33.00   1st Qu.: -40.25   1st Qu.:-11.457   1st Qu.: -93.375  
 Median :  49.50   Median :  94.20   Median :  8.830   Median : -19.250  
 Mean   :  71.40   Mean   :  38.66   Mean   :  7.099   Mean   :   2.195  
 3rd Qu.:  96.25   3rd Qu.: 143.25   3rd Qu.: 28.500   3rd Qu.: 104.500  
 Max.   : 368.00   Max.   : 176.00   Max.   : 59.300   Max.   : 159.000  
 kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
 Mode:logical          Mode:logical           Mode:logical        
 NA's:20               NA's:20                NA's:20             
                                                                  
                                                                  
                                                                  
                                                                  
 skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
 Mode:logical          Mode:logical           Mode:logical        
 NA's:20               NA's:20                NA's:20             
                                                                  
                                                                  
                                                                  
                                                                  
 max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm
 Mode:logical     Mode:logical      Mode:logical    Mode:logical    
 NA's:20          NA's:20           NA's:20         NA's:20         
                                                                    
                                                                    
                                                                    
                                                                    
 min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
 Mode:logical      Mode:logical    Mode:logical          
 NA's:20           NA's:20         NA's:20               
                                                         
                                                         
                                                         
                                                         
 amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
 Mode:logical            Mode:logical          Min.   :21.00      
 NA's:20                 NA's:20               1st Qu.:24.00      
                                               Median :32.50      
                                               Mean   :32.05      
                                               3rd Qu.:36.75      
                                               Max.   :47.00      
 var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
 Mode:logical      Mode:logical     Mode:logical        Mode:logical    
 NA's:20           NA's:20          NA's:20             NA's:20         
                                                                        
                                                                        
                                                                        
                                                                        
 avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
 Mode:logical      Mode:logical         Mode:logical      Mode:logical   
 NA's:20           NA's:20              NA's:20           NA's:20        
                                                                         
                                                                         
                                                                         
                                                                         
 stddev_yaw_forearm var_yaw_forearm gyros_forearm_x   gyros_forearm_y  
 Mode:logical       Mode:logical    Min.   :-1.0600   Min.   :-5.9700  
 NA's:20            NA's:20         1st Qu.:-0.5850   1st Qu.:-1.2875  
                                    Median : 0.0200   Median : 0.0350  
                                    Mean   :-0.0200   Mean   :-0.0415  
                                    3rd Qu.: 0.2925   3rd Qu.: 2.0475  
                                    Max.   : 1.3800   Max.   : 4.2600  
 gyros_forearm_z   accel_forearm_x  accel_forearm_y  accel_forearm_z 
 Min.   :-1.2600   Min.   :-212.0   Min.   :-331.0   Min.   :-282.0  
 1st Qu.:-0.0975   1st Qu.:-114.8   1st Qu.:   8.5   1st Qu.:-199.0  
 Median : 0.2300   Median :  86.0   Median : 138.0   Median :-148.5  
 Mean   : 0.2610   Mean   :  38.8   Mean   : 125.3   Mean   : -93.7  
 3rd Qu.: 0.7625   3rd Qu.: 166.2   3rd Qu.: 268.0   3rd Qu.: -31.0  
 Max.   : 1.8000   Max.   : 232.0   Max.   : 406.0   Max.   : 179.0  
 magnet_forearm_x magnet_forearm_y magnet_forearm_z   problem_id   
 Min.   :-714.0   Min.   :-787.0   Min.   :-32.0    Min.   : 1.00  
 1st Qu.:-427.2   1st Qu.:-328.8   1st Qu.:275.2    1st Qu.: 5.75  
 Median :-189.5   Median : 487.0   Median :491.5    Median :10.50  
 Mean   :-159.2   Mean   : 191.8   Mean   :460.2    Mean   :10.50  
 3rd Qu.:  41.5   3rd Qu.: 720.8   3rd Qu.:661.5    3rd Qu.:15.25  
 Max.   : 532.0   Max.   : 800.0   Max.   :884.0    Max.   :20.00  
 [ reached getOption("max.print") -- omitted 1 row ]
> dim(data)
[1] 19622   160
> dim(validation)
[1]  20 160
> #Remove unnecessary columns
> # first 7 columns don't contain useful info
> data <- data[,-seq(1:7)]
> validation <- validation[,-seq(1:7)]
> 
> #Remove columns with NAs This reduces de amount of predictors to 53
> # select columns that don't have NAs
> indexNA <- as.vector(sapply(data[,1:152],function(x) {length(which(is.na(x)))!=0}))
> data <- data[,!indexNA]
> validation <- validation[,!indexNA]
> # set last (classe) and prior (- classe) column index
> #last <- as.numeric(ncol(data))
> #prior <- last - 1
> 
> # set variables to numerics for correlation check, except the "classe"
> for (i in 1:prior) {
+   data[,i] <- as.numeric(data[,i])
+   validation[,i] <- as.numeric(validation[,i])
+ }
> 
> #check the correlations
> cor.check <- cor(data[, -c(last)])
> diag(cor.check) <- 0 
> plot( levelplot(cor.check, main ="Correlation matrix for all WLE features in training set",
+                 scales=list(x=list(rot=90), cex=1.0)))
> # find the highly correlated variables
> highly.cor <- findCorrelation(cor(data[, -c(last)]), cutoff=0.9)
> # pre process variables
> preObj <-preProcess(data[,1:prior],method=c('knnImpute', 'center', 'scale'))
> dataPrep <- predict(preObj, data[,1:prior])
> dataPrep$classe <- data$classe
> 
> valPrep <-predict(preObj,validation[,1:prior])
> valPrep$problem_id <- validation$problem_id
> # test near zero variance
> myDataNZV <- nearZeroVar(dataPrep, saveMetrics=TRUE)
> if (any(myDataNZV$nzv)) nzv else message("No variables with near zero variance")
No variables with near zero variance
> dataPrep <- dataPrep[,myDataNZV$nzv==FALSE]
> valPrep <- valPrep[,myDataNZV$nzv==FALSE]
> # split dataset into training and test set
> inTrain <- createDataPartition(y=dataPrep$classe, p=0.7, list=FALSE )
> training <- dataPrep[inTrain,]
> testing <- dataPrep[-inTrain,]
> # set seed for reproducibility
> set.seed(12345)
> 
> # get the best mtry
> bestmtry <- tuneRF(training[-last],training$classe, ntreeTry=100, 
+                    stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
mtry = 7  OOB error = 0.65% 
Searching left ...
mtry = 5 	OOB error = 0.72% 
-0.1123596 0.01 
Searching right ...
mtry = 10 	OOB error = 0.55% 
0.1573034 0.01 
mtry = 15 	OOB error = 0.57% 
-0.04 0.01 
> 
> mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]
> 
> # Model 1: RandomForest
> wle.rf <-randomForest(classe~.,data=training, mtry=mtry, ntree=501, 
+                       keep.forest=TRUE, proximity=TRUE, 
+                       importance=TRUE,test=testing)
> # plot the Out of bag error estimates
> layout(matrix(c(1,2),nrow=1), width=c(4,1)) 
> par(mar=c(5,4,4,0)) #No margin on the right side
> plot(wle.rf, log="y", main ="Out-of-bag (OOB) error estimate per Number of Trees")
> par(mar=c(5,0,4,2)) #No margin on the left side
> plot(c(0,1),type="n", axes=F, xlab="", ylab="")
> legend("top", colnames(wle.rf$err.rate),col=1:6,cex=0.8,fill=1:6)
> # plot the accuracy and Gini
> varImpPlot(wle.rf, main="Mean Decrease of Accuracy and Gini per variable")
> # MDSplot (we couldn't execute this due to lack of memory)
> MDSplot(wle.rf, training$classe)
Error: cannot allocate vector of size 1.4 Gb
> # results with training set
> predict1 <- predict(wle.rf, newdata=training)
> confusionMatrix(predict1,training$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 3906    0    0    0    0
         B    0 2658    0    0    0
         C    0    0 2396    0    0
         D    0    0    0 2252    0
         E    0    0    0    0 2525

Overall Statistics
                                     
               Accuracy : 1          
                 95% CI : (0.9997, 1)
    No Information Rate : 0.2843     
    P-Value [Acc > NIR] : < 2.2e-16  
                                     
                  Kappa : 1          
 Mcnemar's Test P-Value : NA         

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
> #Confusion Matrix and Statistics
> # results with test set
> predict2 <- predict(wle.rf, newdata=testing)
> confusionMatrix(predict2,testing$classe)
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673   11    0    0    0
         B    1 1125    8    0    0
         C    0    3 1016    8    1
         D    0    0    2  955    2
         E    0    0    0    1 1079

Overall Statistics
                                          
               Accuracy : 0.9937          
                 95% CI : (0.9913, 0.9956)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.992           
 Mcnemar's Test P-Value : NA              


Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   0.9877   0.9903   0.9907   0.9972
Specificity            0.9974   0.9981   0.9975   0.9992   0.9998
Pos Pred Value         0.9935   0.9921   0.9883   0.9958   0.9991
Neg Pred Value         0.9998   0.9971   0.9979   0.9982   0.9994
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1912   0.1726   0.1623   0.1833
Detection Prevalence   0.2862   0.1927   0.1747   0.1630   0.1835
Balanced Accuracy      0.9984   0.9929   0.9939   0.9949   0.9985
> 
> # Confusion Matrix and Statistics
> #Train Model 2: Decision Tree
> # Model 2: Decision Tree
> dt <- rpart(classe ~ ., data=training, method="class")
> 
> # fancyRpartPlot works for small trees, but not for ours
> fancyRpartPlot(dt)
Warning message:
labs do not fit even at cex 0.15, there may be some overplotting












