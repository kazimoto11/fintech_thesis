
rm(list=ls())

#Set working directory
setwd("/Users/derickkazimoto/Desktop/Master's thesis/R - Logistic regression")

data <- read.csv("data.csv") 

#Checking the types of data
str(data)

#Changing status to a factor
data$Status = as.factor(data$Status)


#Checking the target value distribution
table(data$Status)

# Excluding the account_id and status variables
data[,-c(1,10)] = scale(data[,-c(1,10)])

#Splitting the data into train (70%) and test sets (30%).

#70% of the sample size
smp_size <- floor(0.70 * nrow(data))
#set the seed to 1 to make the partition reproducible
set.seed(1)

train_ind <- sample(seq_len(nrow(data)), size = smp_size) #row numbers for training set
train <- data[train_ind,-c(1,4,5)]#training data (70%)
test <- data[-train_ind,-c(1,4,5)]#testing data (30%)

table(train$Status)
table(test$Status)


library(ggplot2) # plotting
library(e1071) # svm
library(factoextra) # convex hull plotting


###################################################################################################

#SVM 1A

p = ggplot(data = train, aes(x = No_of_delinquent_loans , y = c(0))) +
  geom_point(aes(color=Status))

p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ No_of_delinquent_loans,
                   data = train,
                   type = 'C-classification',
                   kernel = "linear",
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~No_of_delinquent_loans),
                data = train,
                type = 'C-classification',
                kernel = "linear",
                scale = FALSE,
                cost = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)
library(MLmetrics)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions, labels)


perf <- performance(pred,"tpr","fpr")
#plot(perf,colorize=TRUE)

#library(PRROC)
#library(pROC)

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)



#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
#auc(labels, predictions)
#gini = (2*auc(labels, predictions)) -1
#gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_1a <- dat[-i,]
  test_1a <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ No_of_delinquent_loans),
               data = train_1a,
               type = 'C-classification',
               kernel = "linear",
               scale = FALSE,
               cost = 0.1)
  
  # Predict results
  results <- predict(model,test_1a)
  
  # Actual answers
  answers <- test_1a$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 1B

p = ggplot(data = train, aes(x = No_of_delinquent_loans , y = c(0))) +
  geom_point(aes(color=Status))

p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ No_of_delinquent_loans,
                   data = train,
                   type = 'C-classification',
                   kernel = "radial",
                   gamma =  c(0.1, 1, 10, 100),
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~No_of_delinquent_loans),
                data = train,
                type = 'C-classification',
                kernel = "radial",
                scale = FALSE,
                cost = 1.1, 
                gamma = 10)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)
library(MLmetrics)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions, labels)


perf <- performance(pred,"tpr","fpr")
#plot(perf,colorize=TRUE)

#library(PRROC)
#library(pROC)

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)


predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions,labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels,predictions)
gini = (2*auc(labels,predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_1b <- dat[-i,]
  test_1b <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ No_of_delinquent_loans),
               data = train_1b,
               type = 'C-classification',
               kernel = "radial",
               scale = FALSE,
               cost = 1.1,
               gamma= 10)
  
  # Predict results
  results <- predict(model,test_1b)
  
  # Actual answers
  answers <- test_1b$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 2A

p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +
  geom_point(aes(color=Status))

p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ MNP_rate,
                   data = train,
                   type = 'C-classification',
                   kernel = "linear",
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~MNP_rate),
                data = train,
                type = 'C-classification',
                kernel = "linear",
                scale = FALSE,
                cost = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)
library(MLmetrics)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions, labels)


perf <- performance(pred,"tpr","fpr")
#plot(perf,colorize=TRUE)

#library(PRROC)
#library(pROC)

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions, labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels, predictions)
gini = (2*auc(labels, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_2a <- dat[-i,]
  test_2a <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ MNP_rate),
               data = train_2a,
               type = 'C-classification',
               kernel = "linear",
               scale = FALSE,
               cost = 0.1)
  
  # Predict results
  results <- predict(model,test_2a)
  
  # Actual answers
  answers <- test_2a$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 2B

p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +
  geom_point(aes(color=Status))

p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ MNP_rate,
                   data = train,
                   type = 'C-classification',
                   kernel = "radial",
                   gamma =  c(0.1, 1, 10, 100),
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~MNP_rate),
                data = train,
                type = 'C-classification',
                kernel = "radial",
                scale = FALSE,
                cost = 27.1, 
                gamma = 1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)
library(MLmetrics)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions, labels)


perf <- performance(pred,"tpr","fpr")
#plot(perf,colorize=TRUE)

#library(PRROC)
#library(pROC)

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

pred <- prediction(predictions,labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels,predictions)
gini = (2*auc(labels,predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_2b <- dat[-i,]
  test_2b <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ MNP_rate),
               data = train_2b,
               type = 'C-classification',
               kernel = "radial",
               scale = FALSE,
               cost = 27.1,
               gamma= 1)
  
  # Predict results
  results <- predict(model,test_2b)
  
  # Actual answers
  answers <- test_2b$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 3A

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ Avg_amount_per_transaction + MNP_rate,
                   data = train,
                   type = 'C-classification',
                   kernel = "linear",
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~Avg_amount_per_transaction + MNP_rate),
                data = train,
                type = 'C-classification',
                kernel = "linear",
                scale = FALSE,
                cost = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1


auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions, labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels, predictions)
gini = (2*auc(labels, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_3a <- dat[-i,]
  test_3a <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ Avg_amount_per_transaction + MNP_rate),
               data = train_3a,
               type = 'C-classification',
               kernel = "linear",
               scale = FALSE,
               cost = 0.1)
  
  # Predict results
  results <- predict(model,test_3a)
  
  # Actual answers
  answers <- test_3a$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 3B

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~Avg_amount_per_transaction + MNP_rate,
                   data = train,
                   type = 'C-classification',
                   kernel = "radial",
                   gamma =  c(0.1, 1, 10, 100),
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~Avg_amount_per_transaction + MNP_rate),
                data = train,
                type = 'C-classification',
                kernel = "radial",
                scale = FALSE,
                cost = 1.1, 
                gamma = 10)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1


auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)


pred <- prediction(predictions,labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels,predictions)
gini = (2*auc(labels,predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_3b <- dat[-i,]
  test_3b <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ Avg_amount_per_transaction + MNP_rate),
               data = train_3b,
               type = 'C-classification',
               kernel = "radial",
               scale = FALSE,
               cost = 1.1,
               gamma= 10)
  
  # Predict results
  results <- predict(model,test_3b)
  
  # Actual answers
  answers <- test_3b$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 4A

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ Avg_amount_per_transaction + No_of_delinquent_loans,
                   data = train,
                   type = 'C-classification',
                   kernel = "linear",
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~Avg_amount_per_transaction + MNP_rate),
                data = train,
                type = 'C-classification',
                kernel = "linear",
                scale = FALSE,
                cost = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions, labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels, predictions)
gini = (2*auc(labels, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_4a <- dat[-i,]
  test_4a <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ Avg_amount_per_transaction + No_of_delinquent_loans),
               data = train_4a,
               type = 'C-classification',
               kernel = "linear",
               scale = FALSE,
               cost = 0.1)
  
  # Predict results
  results <- predict(model,test_4a)
  
  # Actual answers
  answers <- test_4a$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 4B

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~Avg_amount_per_transaction + No_of_delinquent_loans,
                   data = train,
                   type = 'C-classification',
                   kernel = "radial",
                   gamma =  c(0.1, 1, 10, 100),
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~Avg_amount_per_transaction + No_of_delinquent_loans),
                data = train,
                type = 'C-classification',
                kernel = "radial",
                scale = FALSE,
                cost = 17.6, 
                gamma = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions,labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels,predictions)
gini = (2*auc(labels,predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_4b <- dat[-i,]
  test_4b <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ Avg_amount_per_transaction + No_of_delinquent_loans),
               data = train_4b,
               type = 'C-classification',
               kernel = "radial",
               scale = FALSE,
               cost = 19.6,
               gamma= 0.1)
  
  # Predict results
  results <- predict(model,test_4b)
  
  # Actual answers
  answers <- test_4b$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 5A

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ No_of_paid_loans+No_of_delinquent_loans,
                   data = train,
                   type = 'C-classification',
                   kernel = "linear",
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~No_of_paid_loans+No_of_delinquent_loans),
                data = train,
                type = 'C-classification',
                kernel = "linear",
                scale = FALSE,
                cost = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1


auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions, labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels, predictions)
gini = (2*auc(labels, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_5a <- dat[-i,]
  test_5a <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ No_of_paid_loans+No_of_delinquent_loans),
               data = train_5a,
               type = 'C-classification',
               kernel = "linear",
               scale = FALSE,
               cost = 0.1)
  
  # Predict results
  results <- predict(model,test_5a)
  
  # Actual answers
  answers <- test_5a$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 5B

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ No_of_paid_loans+No_of_delinquent_loans,
                   data = train,
                   type = 'C-classification',
                   kernel = "radial",
                   gamma =  c(0.1, 1, 10, 100),
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm(Status ~ No_of_paid_loans+No_of_delinquent_loans,
                data = train,
                type = 'C-classification',
                kernel = "radial",
                scale = FALSE,
                cost = 5.6, 
                gamma = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions,labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels,predictions)
gini = (2*auc(labels,predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_5b <- dat[-i,]
  test_5b <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ No_of_paid_loans+No_of_delinquent_loans),
               data = train_5b,
               type = 'C-classification',
               kernel = "radial",
               scale = FALSE,
               cost = 5.6,
               gamma= 0.1)
  
  # Predict results
  results <- predict(model,test_5b)
  
  # Actual answers
  answers <- test_5b$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 6A

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                     No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate,
                   data = train,
                   type = 'C-classification',
                   kernel = "linear",
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm((Status ~No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                   No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate),
                data = train,
                type = 'C-classification',
                kernel = "linear",
                scale = FALSE,
                cost = 79.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions, labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels, predictions)
gini = (2*auc(labels, predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_6a <- dat[-i,]
  test_6a <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                  No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate),
               data = train_6a,
               type = 'C-classification',
               kernel = "linear",
               scale = FALSE,
               cost = 79.1)
  
  # Predict results
  results <- predict(model,test_6a)
  
  # Actual answers
  answers <- test_6a$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

###################################################################################################

#SVM 6B

#p = ggplot(data = train, aes(x = MNP_rate , y = c(0))) +geom_point(aes(color=Status))

#p

set.seed(123)

#Tune svm
tunesvm = tune.svm(Status ~ No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                     No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate,
                   data = train,
                   type = 'C-classification',
                   kernel = "radial",
                   gamma =  c(0.1, 1, 10, 100),
                   cost = seq(from=0.1 , to=100, by=0.5))

tunesvm
bestmodel = tunesvm$best.model
bestmodel

#names(svm_model)
svm_model = svm(Status ~ No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                  No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate,
                data = train,
                type = 'C-classification',
                kernel = "radial",
                scale = FALSE,
                cost = 2.6, 
                gamma = 0.1)

# Predict Species using the model
svm_pred_train = predict(svm_model, train)
svm_pred = predict(svm_model, test)

# plot the decision boundary using the usual plot() function
# plot(svm_model,data=train, train[,c(5,7)])


#Evaluating the test set
library(caret)
confusionMatrix(train$Status, as.factor(svm_pred_train))
confusionMatrix(test$Status, as.factor(svm_pred))
# plot the decision boundary using the usual plot() function
#plot(svm_model, test)

#--------------------------------------------------------------------------------------------------------
library(ROCR)

predictions = predict(svm_model, test,type='response')

# add a as.numeric and then minus 1
predictions = as.numeric(predictions) -1
labels = as.numeric(test$Status)-1

auc_perf = performance( pred, measure = "auc" )
Gini(y_pred = predictions, y_true = labels)

pred <- prediction(predictions,labels)

perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)

library(PRROC)
library(pROC)

#PRROC_obj <- roc.curve(scores.class0 = predictions, weights.class0=test$Status,curve=TRUE)
auc(labels,predictions)
gini = (2*auc(labels,predictions)) -1
gini
#plot(PRROC_obj)

#--------------------------------------------------------------------------------
#LOOCV

# Select statistically significant variables
dat <- subset(data,select=c(c(2:3),c(6:8),c(10:12)))

acc <- NULL
for(i in 1:nrow(data))
{
  # Train-test splitting
  # 499 samples -> fitting
  # 1 sample -> testing
  train_6b <- dat[-i,]
  test_6b <- dat[i,]
  
  # Fitting
  model <- svm((Status ~ No_of_Airtime_Pins+No_of_loan_disbursements+Amount+No_of_paid_loans+
                  No_of_delinquent_loans+Avg_amount_per_transaction+MNP_rate),
               data = train_6b,
               type = 'C-classification',
               kernel = "radial",
               scale = FALSE,
               cost = 2.6,
               gamma= 0.1)
  
  # Predict results
  results <- predict(model,test_6b)
  
  # Actual answers
  answers <- test_6b$Status
  
  # Calculate accuracy
  misClasificError <- mean(answers != results)
  
  # Collecting results
  acc[i] <- 1-misClasificError
}

# Average accuracy of the model
mean(acc)

