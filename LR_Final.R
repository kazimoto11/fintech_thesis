
rm(list=ls())
###################################################################################################

#Set working directory
setwd("/Users/derickkazimoto/Desktop/Master's thesis/R - Logistic regression")
data <- read.csv("data.csv") 

#Checking the types of data
str(data)

#Changing status to a factor
data$Status = as.factor(data$Status)

#Checking the target value distribution
table(data$Status)

# Scaling the dataset
# Excluding the account_id and status variables
data[,-c(1,10)] = scale(data[,-c(1,10)])

#Splitting the data into train (70%) and test sets (30%).

#70% of the sample size
smp_size <- floor(0.7 * nrow(data))
#set the seed to 1 to make the partition reproducible
set.seed(1)

train_ind <- sample(seq_len(nrow(data)), size = smp_size) #row numbers for training set
train <- data[train_ind,]#training data (75%)
test <- data[-train_ind,]#testing data (25%)

table(train$Status)
table(test$Status)

###################################################################################################

#Logistic Regression Model 1 - No of delinquent loans

#Fitting a logistic regression model

glm.fits=glm(Status ~No_of_delinquent_loans, 
             data=train ,family =binomial )

#Summary of the logistic regression model
summary (glm.fits)

#Making train predictions
glm.probs = predict(glm.fits, train,type='response')
train_preds = rep(0,nrow(train))
train_preds[glm.probs>0.5] = 1
train_preds

#Making test predictions
glm.probs = predict(glm.fits, test,type='response')
preds = rep(0,nrow(test))
preds[glm.probs>0.5] = 1
preds

#Evaluating the test set
library(caret)
confusionMatrix(train$Status,as.factor(train_preds) )
confusionMatrix(test$Status, as.factor(preds))

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(glm.fits, test,type='response')

pred <- prediction(predictions, test$Status)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(test$Status, predictions)
gini = (2*auc(test$Status, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------------------------------

#LOOCV 

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_cv <- dat[-i,]
  test_cv <- dat[i,]
  
  # Fitting
  model <- glm(Status ~ No_of_delinquent_loans, 
               data=train_cv,family=binomial)
  
  # Predict results
  results_prob <- predict(model,test_cv,type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test_cv$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

# Histogram of the model accuracy
hist(acc,xlab='Accuracy',ylab='Freq',main='Accuracy LOOCV',
     col='cyan',border='blue',density=30)

###################################################################################################

#Logistic Regression Model 2 - MNP_rate

#Fitting a logistic regression model

glm.fits=glm(Status ~MNP_rate, 
             data=train ,family =binomial )

#Summary of the logistic regression model
summary (glm.fits)

#Making train predictions
glm.probs = predict(glm.fits, train,type='response')
train_preds = rep(0,nrow(train))
train_preds[glm.probs>0.5] = 1
train_preds

#Making test predictions
glm.probs = predict(glm.fits, test,type='response')
preds = rep(0,nrow(test))
preds[glm.probs>0.5] = 1
preds

#Evaluating the test set
library(caret)
confusionMatrix(train$Status,as.factor(train_preds) )
confusionMatrix(test$Status, as.factor(preds))


#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(glm.fits, test,type='response')

pred <- prediction(predictions, test$Status)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(test$Status, predictions)
gini = (2*auc(test$Status, predictions)) -1
gini
#plot(PRROC_obj)

predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1
Gini(y_pred = predictions, y_true = labels)


#--------------------------------------------------------------------------------------------------------

#LOOCV 

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_cv <- dat[-i,]
  test_cv <- dat[i,]
  
  # Fitting
  model <- glm(Status ~ MNP_rate, 
               data=train_cv,family=binomial)
  
  # Predict results
  results_prob <- predict(model,test_cv,type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test_cv$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

# Histogram of the model accuracy
hist(acc,xlab='Accuracy',ylab='Freq',main='Accuracy LOOCV',
     col='cyan',border='blue',density=30)

###################################################################################################

#Logistic Regression Model 3 - MNP_rate, Avg_amount_per_transaction

#Fitting a logistic regression model

glm.fits=glm(Status ~MNP_rate+Avg_amount_per_transaction, 
             data=train ,family =binomial )

#Summary of the logistic regression model
summary (glm.fits)

#Making train predictions
glm.probs = predict(glm.fits, train,type='response')
train_preds = rep(0,nrow(train))
train_preds[glm.probs>0.5] = 1
train_preds

#Making test predictions
glm.probs = predict(glm.fits, test,type='response')
preds = rep(0,nrow(test))
preds[glm.probs>0.5] = 1
preds

#Evaluating the test set
library(caret)
confusionMatrix(train$Status,as.factor(train_preds) )
confusionMatrix(test$Status, as.factor(preds))

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(glm.fits, test,type='response')

pred <- prediction(predictions, test$Status)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(test$Status, predictions)
gini = (2*auc(test$Status, predictions)) -1
gini
#plot(PRROC_obj)

predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1
Gini(y_pred = predictions, y_true = labels)

#--------------------------------------------------------------------------------------------------------

#LOOCV 

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_cv <- dat[-i,]
  test_cv <- dat[i,]
  
  # Fitting
  model <- glm(Status ~ MNP_rate+Avg_amount_per_transaction, 
               data=train_cv,family=binomial)
  
  # Predict results
  results_prob <- predict(model,test_cv,type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test_cv$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

# Histogram of the model accuracy
hist(acc,xlab='Accuracy',ylab='Freq',main='Accuracy LOOCV',
     col='cyan',border='blue',density=30)

###################################################################################################

#Logistic Regression Model 4 - MNP_rate,  No_of_delinquent_loans

#Fitting a logistic regression model

glm.fits=glm(Status ~MNP_rate+No_of_delinquent_loans, 
             data=train ,family =binomial )

#Summary of the logistic regression model
summary (glm.fits)

#Making train predictions
glm.probs = predict(glm.fits, train,type='response')
train_preds = rep(0,nrow(train))
train_preds[glm.probs>0.5] = 1
train_preds

#Making test predictions
glm.probs = predict(glm.fits, test,type='response')
preds = rep(0,nrow(test))
preds[glm.probs>0.5] = 1
preds

#Evaluating the test set
library(caret)
confusionMatrix(train$Status,as.factor(train_preds) )
confusionMatrix(test$Status, as.factor(preds))

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(glm.fits, test,type='response')

pred <- prediction(predictions, test$Status)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(test$Status, predictions)
gini = (2*auc(test$Status, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------------------------------

#LOOCV 

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_cv <- dat[-i,]
  test_cv <- dat[i,]
  
  # Fitting
  model <- glm(Status ~ MNP_rate+No_of_delinquent_loans, 
               data=train_cv,family=binomial)
  
  # Predict results
  results_prob <- predict(model,test_cv,type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test_cv$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

# Histogram of the model accuracy
hist(acc,xlab='Accuracy',ylab='Freq',main='Accuracy LOOCV',
     col='cyan',border='blue',density=30)

###################################################################################################

#Logistic Regression Model 5 - No_of_paid_loans+No_of_delinquent_loans

#Fitting a logistic regression model

glm.fits=glm(Status ~No_of_paid_loans+No_of_delinquent_loans, 
             data=train ,family =binomial )

#Summary of the logistic regression model
summary (glm.fits)

#Making train predictions
glm.probs = predict(glm.fits, train,type='response')
train_preds = rep(0,nrow(train))
train_preds[glm.probs>0.5] = 1
train_preds

#Making test predictions
glm.probs = predict(glm.fits, test,type='response')
preds = rep(0,nrow(test))
preds[glm.probs>0.5] = 1
preds

#Evaluating the test set
library(caret)
confusionMatrix(train$Status,as.factor(train_preds) )
confusionMatrix(test$Status, as.factor(preds))

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(glm.fits, test,type='response')

pred <- prediction(predictions, test$Status)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(test$Status, predictions)
gini = (2*auc(test$Status, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------------------------------

#LOOCV 

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_cv <- dat[-i,]
  test_cv <- dat[i,]
  
  # Fitting
  model <- glm(Status ~ No_of_paid_loans+No_of_delinquent_loans, 
               data=train_cv,family=binomial)
  
  # Predict results
  results_prob <- predict(model,test_cv,type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test_cv$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

# Histogram of the model accuracy
hist(acc,xlab='Accuracy',ylab='Freq',main='Accuracy LOOCV',
     col='cyan',border='blue',density=30)

###################################################################################################

#Logistic Regression Model 6 - No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
#No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate,

#Fitting a logistic regression model

glm.fits=glm(Status ~No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
               No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate,, 
             data=train ,family =binomial )

#Summary of the logistic regression model
summary (glm.fits)

#Making train predictions
glm.probs = predict(glm.fits, train,type='response')
train_preds = rep(0,nrow(train))
train_preds[glm.probs>0.5] = 1
train_preds

#Making test predictions
glm.probs = predict(glm.fits, test,type='response')
preds = rep(0,nrow(test))
preds[glm.probs>0.5] = 1
preds

#Evaluating the test set
library(caret)
confusionMatrix(train$Status,as.factor(train_preds) )
confusionMatrix(test$Status, as.factor(preds))

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(glm.fits, test,type='response')

pred <- prediction(predictions, test$Status)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(test$Status, predictions)
gini = (2*auc(test$Status, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------------------------------

#LOOCV 

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_cv <- dat[-i,]
  test_cv <- dat[i,]
  
  # Fitting
  model <- glm(Status ~ No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                 No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate,, 
               data=train_cv,family=binomial)
  
  # Predict results
  results_prob <- predict(model,test_cv,type='response')
  
  # If prob > 0.5 then 1, else 0
  results <- ifelse(results_prob > 0.5,1,0)
  
  # Actual answers
  answers <- test_cv$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

# Histogram of the model accuracy
hist(acc,xlab='Accuracy',ylab='Freq',main='Accuracy LOOCV',
     col='cyan',border='blue',density=30)

###################################################################################################

# Credit Score Model Analysis

probs <- predict(glm.fits, data,type='response')
probs <- round(probs*100)

# select variables account_id , Status
myvars <- c("account_id", "Status")
newdata <- data[myvars]

newdata$Default_Probability <- probs

#Export to excel
library("writexl")
write_xlsx(newdata,"/Users/derickkazimoto/Desktop/Master's thesis/Data/analysis.xlsx")

c.analysis <-  data.frame(probs, data$Status)
c.analysis





