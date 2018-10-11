#-------------------------------------------------------------------------------------------
#Load libraries
#-------------------------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tidyverse)
library(caret)
library(Metrics)
library(pROC)
library(ROCR)
library(randomForest)
library(pdp)
library(corrplot)
library(car)



#-------------------------------------------------------------------------------------------
#Importing Repurchase_Training.csv File
#-------------------------------------------------------------------------------------------

repurchase_raw_data<-read.csv("~/Documents/UTS/02_Courses/36106_DAM/Assignment/Assignment_1/Assignment1B/Raw Data/repurchase_training.csv",header = TRUE)



#-------------------------------------------------------------------------------------------
#Check the data
#-------------------------------------------------------------------------------------------
dim(repurchase_raw_data)

summary(repurchase_raw_data)


#-------------------------------------------------------------------------------------------
#Perform Exploratory Data Analysis on repurchase_training dataset
#-------------------------------------------------------------------------------------------
qplot(as.factor(repurchase_raw_data$Target))
qplot(repurchase_raw_data$age_band)
qplot(repurchase_raw_data$gender)
ggplot(repurchase_raw_data,aes(car_segment))+geom_bar()
repurchase_raw_data%>%group_by(car_model)%>%summarise(Customer_Count=n_distinct(ID))%>%arrange(desc(Customer_Count))
repurchase_raw_data%>%group_by(car_segment)%>%summarise(Customer_Count=n())%>%arrange(desc(Customer_Count))
repurchase_raw_data%>%group_by(age_band)%>%summarise(cnt=n())
repurchase_raw_data%>%group_by(ID)%>%summarise(Duplicates=n())%>%filter(Duplicates >1)


# Distribution of numeric variables(deciles)
qplot(as.factor(repurchase_raw_data$age_of_vehicle_years))
qplot(as.factor(repurchase_raw_data$sched_serv_warr))
qplot(as.factor(repurchase_raw_data$non_sched_serv_warr))
qplot(as.factor(repurchase_raw_data$sched_serv_warr))
qplot(as.factor(repurchase_raw_data$total_paid_services))
qplot(as.factor(repurchase_raw_data$total_services))
qplot(as.factor(repurchase_raw_data$mth_since_last_serv))
qplot(as.factor(repurchase_raw_data$annualised_mileage))
qplot(as.factor(repurchase_raw_data$num_dealers_visited))
qplot(as.factor(repurchase_raw_data$num_serv_dealer_purchased))


#-------------------------------------------------------------------------------------------
#Convert Target Variable to Factor type
#-------------------------------------------------------------------------------------------
repurchase_raw_data$Target=as.factor(repurchase_raw_data$Target)


#-------------------------------------------------------------------------------------------
#Partition the data
#-------------------------------------------------------------------------------------------
set.seed(42)
train_partition <- createDataPartition(repurchase_raw_data$ID, p = 0.75, list = FALSE)
training <- repurchase_raw_data[train_partition, ]
testing <- repurchase_raw_data[-train_partition, ]


#-------------------------------------------------------------------------------------------
#Verify the data in train and test partitions
#-------------------------------------------------------------------------------------------
dim(training)
dim(testing)


#-------------------------------------------------------------------------------------------
#Train the model - Featured selected - all except ID
#-------------------------------------------------------------------------------------------
glm_fit<-glm(formula=Target ~. -ID, family=binomial, data=training)
summary(glm_fit)


#-------------------------------------------------------------------------------------------
#Train the model - Featured selected - all except ID
#-------------------------------------------------------------------------------------------
glm_fit<-glm(formula=Target ~. -ID -car_segment  -non_sched_serv_warr, family=binomial, data=training)
summary(glm_fit)

#-------------------------------------------------------------------------------------------
#Collinearity Check
#-------------------------------------------------------------------------------------------
vif(glm_fit)



#-------------------------------------------------------------------------------------------
#Set training parameters - 5 fold CV
#-------------------------------------------------------------------------------------------
fit_control <- trainControl(## K-fold CV
  method = "cv",
  number = 5,savePredictions = TRUE)


#-------------------------------------------------------------------------------------------
#Run Logistic regression -Train the model
#-------------------------------------------------------------------------------------------

lreg_fit<-train(Target ~. -ID -car_segment -non_sched_serv_warr, data=training, method="glm", family=binomial(), trControl=fit_control)



#-------------------------------------------------------------------------------------------
#Evaluate the Model
#-------------------------------------------------------------------------------------------
summary(lreg_fit)
lreg_fit


#-------------------------------------------------------------------------------------------
#Plot Variable of importance
#-------------------------------------------------------------------------------------------
plot(varImp(lreg_fit),top = 20)


#-------------------------------------------------------------------------------------------
#Fit the model on the test data partition
#-------------------------------------------------------------------------------------------
lreg_prob<-predict(lreg_fit,testing,type = "prob")


#-------------------------------------------------------------------------------------------
#Calculate confusion matrix
#-------------------------------------------------------------------------------------------
threshold <- 0.5
predictor      <- factor( ifelse(lreg_prob[, "1"] > threshold, 1, 0),ordered = TRUE )
#pred      <- relevel(predictor, "yes")   # you may or may not need this; I did
lreg_cfm<-confusionMatrix(predictor,testing$Target)
lreg_cfm


# ---------------------------------------------------------------------------------------------------
# Compute Precision, Recall, F1
# ---------------------------------------------------------------------------------------------------
lreg_cfm$byClass


#-------------------------------------------------------------------------------------------
#Calculate AUC and Plot ROC
#-------------------------------------------------------------------------------------------
lreg_pred <- prediction(lreg_prob[,2], testing$Target)
plot(performance(lreg_pred, measure = "tpr", x.measure = "fpr"),main="ROC Curve for GLM Model",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
lreg_auc <- performance(lreg_pred, measure = "auc")
lreg_auc <- lreg_auc@y.values[[1]]
lreg_auc

# ----------------------------------------------------------------------------------------------------------
# Model the dataset using Tree based Model - Random Forest
# ----------------------------------------------------------------------------------------------------------

rf_fit <- randomForest(Target ~. -ID -car_segment -non_sched_serv_warr, data = training, ntree = 1000, importance = TRUE)


#----------------------------------------------------------------------------------------------------------
#Summarise the model - Random Forest
#----------------------------------------------------------------------------------------------------------
summary(rf_fit)
print(rf_fit)

#----------------------------------------------------------------------------------------------------------
#Variable of Importance - Random Forest
#----------------------------------------------------------------------------------------------------------
varImpPlot(rf_fit)


#----------------------------------------------------------------------------------------------------------
#Fit the model - Random Forest
#----------------------------------------------------------------------------------------------------------
rf_pred<-predict(rf_fit,testing)


#----------------------------------------------------------------------------------------------------------
#Fit the model using probability class - Random Forest
#----------------------------------------------------------------------------------------------------------
rf_pred_prob<-predict(rf_fit,testing,type = "prob")



#----------------------------------------------------------------------------------------------------------
#confusion Matrix - Random Forest
#----------------------------------------------------------------------------------------------------------
rf_cfm<-confusionMatrix(rf_pred,factor(testing$Target,ordered = TRUE))
rf_cfm
rf_auc<-auc(rf_pred,factor(testing$Target,ordered = TRUE))
rf_cfm$byClass
rf_auc

#----------------------------------------------------------------------------------------------------------
#Plot the ROC curve - Random Forest
#----------------------------------------------------------------------------------------------------------

rf_perf = prediction(rf_pred_prob[,2],testing$Target)

# 1. True Positive and Negative Rate
tpr_fpr = performance(rf_perf, "tpr","fpr")

# 2. Plot the ROC curve
plot(tpr_fpr,main="ROC Curve for Random Forest Model",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")


#----------------------------------------------------------------------------------------------------------
#Partial Dependency Plots - Random Forest Model
#----------------------------------------------------------------------------------------------------------
#partial(x=rf_fit, pred.data=training, x.var=annualised_mileage)
autoplot(partial(rf_fit, pred.var=c("annualised_mileage"), chull = TRUE))
autoplot(partial(rf_fit, pred.var=c("mth_since_last_serv"), chull = TRUE))
autoplot(partial(rf_fit, pred.var=c("gender"), chull = TRUE))
autoplot(partial(rf_fit, pred.var=c("num_serv_dealer_purchased"), chull = TRUE))
autoplot(partial(rf_fit, pred.var=c("age_of_vehicle_years"), chull = TRUE))


#----------------------------------------------------------------------------------------------------------
#Partial Dependency Plots - Model Compare
#----------------------------------------------------------------------------------------------------------
model_compare_metrics<-cbind(data.frame(rf_cfm$byClass),data.frame(lreg_cfm$byClass))
model_compare_metrics_auc<-cbind(rf_auc,lreg_auc)
model_compare_metrics
model_compare_metrics_auc


#----------------------------------------------------------------------------------------------------------
#ROC Curve - Random Forest and Logistic Regression
#----------------------------------------------------------------------------------------------------------
plot(tpr_fpr,main="ROC Curve for Random Forest and Logistic Regression model",col=3,lwd=2)
#abline(a=0,b=1,lwd=2,lty=2,col="gray")
plot(performance(lreg_pred, measure = "tpr", x.measure = "fpr"),main="ROC Curve for GLM Model",col=2,lwd=2,add=TRUE)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
legend("bottomright", legend=c("Random Forest", "Logistic Regression"),
       col=c("green", "red"), lty=1:1, cex=0.8)


#----------------------------------------------------------------------------------------------------------
#Import the Repurchase Validation data set
#----------------------------------------------------------------------------------------------------------

repurchase_validation_raw_data<-read.csv("~/Documents/UTS/02_Courses/36106_DAM/Assignment/Assignment_1/Assignment1B/Raw Data/repurchase_validation.csv",header = TRUE)



#----------------------------------------------------------------------------------------------------------
#Check the data
#----------------------------------------------------------------------------------------------------------
glimpse(repurchase_validation_raw_data)


#----------------------------------------------------------------------------------------------------------
#Create ID column and target column in the data set
#----------------------------------------------------------------------------------------------------------
repurchase_validation_raw_data$ID <- seq.int(nrow(repurchase_validation_raw_data))
lvl<-c("0","1")
repurchase_validation_raw_data$Target <- factor(lvl,levels=c("0","1"))


#----------------------------------------------------------------------------------------------------------
#Explore the data
#----------------------------------------------------------------------------------------------------------
ggplot(repurchase_validation_raw_data,aes(age_band))+geom_bar()
ggplot(repurchase_validation_raw_data,aes(gender))+geom_bar()
#qplot(repurchase_validation_raw_data$age_band)

#qplot(repurchase_validation_raw_data$gender)
repurchase_validation_raw_data%>%group_by(car_model)%>%summarise(Customer_Count=n_distinct(ID))%>%arrange(desc(Customer_Count))
repurchase_validation_raw_data%>%group_by(car_segment)%>%summarise(Customer_Count=n())%>%arrange(desc(Customer_Count))
ggplot(repurchase_validation_raw_data,aes(car_segment))+geom_bar()
qplot(as.factor(repurchase_validation_raw_data$age_of_vehicle_years))
qplot(as.factor(repurchase_validation_raw_data$sched_serv_warr))
qplot(as.factor(repurchase_validation_raw_data$non_sched_serv_warr))
qplot(as.factor(repurchase_validation_raw_data$sched_serv_warr))
qplot(as.factor(repurchase_validation_raw_data$total_paid_services))
qplot(as.factor(repurchase_validation_raw_data$total_services))
qplot(as.factor(repurchase_validation_raw_data$mth_since_last_serv))
qplot(as.factor(repurchase_validation_raw_data$annualised_mileage))
qplot(as.factor(repurchase_validation_raw_data$num_dealers_visited))
qplot(as.factor(repurchase_validation_raw_data$num_serv_dealer_purchased))


#----------------------------------------------------------------------------------------------------------
#Fit the RF model and Calculate Target Class
#----------------------------------------------------------------------------------------------------------
repurchase_validation_raw_data <- rbind(repurchase_raw_data[1, ] , repurchase_validation_raw_data)
repurchase_validation_raw_data<-repurchase_validation_raw_data[-1,]
rf_validation_pred<-predict(rf_fit,repurchase_validation_raw_data)
repurchase_validation_raw_data$target_class<-rf_validation_pred


#----------------------------------------------------------------------------------------------------------
#calculate Target Probability
#----------------------------------------------------------------------------------------------------------
rf_validation_pred_prob<-predict(rf_fit,repurchase_validation_raw_data,type="prob")
repurchase_validation_raw_data$target_probability<-rf_validation_pred_prob[,2]


#----------------------------------------------------------------------------------------------------------
#Write output to CSV
#----------------------------------------------------------------------------------------------------------
write.csv((repurchase_validation_raw_data%>%select(ID=ID,target_probability=target_probability,target_class=target_class)),file="~/Documents/UTS/02_Courses/36106_DAM/Assignment/Assignment_1/Assignment1B/Raw Data/repurchase_validation_13293283.csv",row.names = FALSE)

