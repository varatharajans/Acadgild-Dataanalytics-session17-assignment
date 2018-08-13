# Acadgild-Dataanalytics-session17-assignment
DATA ANALYTICS WITH R, EXCEL AND TABLEAU SESSION 17ASSIGNMENT 



                                          Session 17 Assignment
                                         Weight Lifting Exercise-2


This human activity recognition research has traditionally focused on discriminating between different activities, i.e. to predict "which" activity was performed at a specific point in time (like with the Daily Living Activities dataset above). The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer. The "how (well)" investigation has only received little attention so far, even though it potentially provides useful information for a large variety of applications, such as sports training.

2. Perform the below given activities: 
a. Create classification model using logistic regression model 
b. verify model goodness of fit 
c. Report the accuracy measures 
d. Report the variable importance 
e. Report the unimportant variables 
f. Interpret the results 
g. Visualize the results


setwd("C:/Users/Seshan/Desktop/sv R related/acadgild/assignments/session17")
library(readr)
Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1 <- read.csv("Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1.csv",header=T,na.strings=c(""))
View(Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1)
View(Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1)
data<-Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1
#Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1 <- read.csv("Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1.csv",header=T,na.strings=c(""))
#data<-s <- read.csv("Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations1.csv",header=T,na.strings=c(""))
View(data)

# load libraries
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lattice)
library(rattle)
summary(data)
library(C50)
#install.package('devtools') # Only needed if you dont have this installed.
library(devtools)
install_github('adam-m-mcelhinney/helpRFunctions')
library(helpRFunctions)
names(data)
dim(data)
library(caret)
library(zoo)
library(plyr) 
is.na(data)
which(is.na(data))
sum(is.na(data))
colSums(is.na(data))
data[is.na(data)] <- mean(data, na.rm = TRUE)
str(data)
summary(data)
pairs(data[8:15])


# set last (classe) and prior (- classe) column index
last <- as.numeric(ncol(data))
prior <- last - 1
# set variables to numerics for correlation check, except the "classe"
for (i in 1:prior) {
  data[,i] <- as.numeric(data[,i])}

# enable multi-core processing
library(doParallel)
#cl <- makeCluster(detectCores())
registerDoParallel()
set.seed(12345)
dataTrain<-data[1:4004,]
dataTest<-data[4005:4024,]
cor.check <- cor(dataTrain[, -c(last)])
diag(cor.check) <- 0 
plot( levelplot(cor.check,main ="Correlation matrix for all WLE features in training set",
                scales=list(x=list(rot=90), cex=1.0) ))

# logistic regression model:
fit <- glm(classe~.,data = dataTrain,family = binomial(link='logit'))
summary(fit)

library(MASS)
step_fit <- stepAIC(fit,method='backward')
summary(step_fit)
confint(step_fit)
#ANOVA on base model
anova(fit,test = 'Chisq')

#ANOVA from reduced model after applying the Step AIC
anova(step_fit,test = 'Chisq')

#plot the fitted model
plot(fit$fitted.values)

pred_link <- predict(fit,newdata = dataTest,type = 'link')

#check for multicollinearity
library(car)
vif(fit)
vif(step_fit)

library(caret)
#with default prob cut 0.50
dataTest$pred_classe <- ifelse(pred<0.7,'yes','no')

table(dataTest$pred_classe,dataTest$classe)

#training split of churn classes
round(table(dataTrain$classe)/nrow(dataTrain),2)*100
# test split of churn classes
round(table(dataTest$classe)/nrow(dataTest),2)*100
#predicted split of churn classes 
round(table(dataTest$pred_classe)/nrow(dataTest),2)*100

#create confusion matrix 
confusionMatrix(dataTest$classe,dataTest$classe)

#how do we create a cross validation scheme
control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3)
seed <-7
metric <- 'Accuracy'
set.seed(seed)
fit_default <- train(classe~., 
                     data = dataTrain,
                     method = 'glm',
                     metric = 0,
                     trControl = control)
print(fit_default)

library(caret)
varImp(step_fit)
varImp(fit_default)
library(devtools)

install_github("riv","tomasgreif")

install_github("woe","tomasgreif")

library(woe)

library(riv)

iv_df <- iv.mult(dataTrain, y="classe", summary=TRUE, verbose=TRUE)
iv_df

iv <- iv.mult(dataTrain, y="classe", summary=FALSE, verbose=TRUE)
# Plot information value summary

iv.plot.summary(iv_df)


