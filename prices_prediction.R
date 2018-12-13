rm(list = ls())
library(tidyverse)
library(caret)
library(keras)
library(Matrix)
library(mice)
library(xgboost)
library(rBayesianOptimization)
`%nin%` = negate(`%in%`)

percent_nulls = function(x){
  (sum((is.na(x)) | (x=="") | (x=="NA") | (x=="na") | (sum(is.nan(x))) | sum(is.null(x)))/length(x) * 100)
}

train_data = read.csv("train.csv")


percent_na_columns = sapply(train_data, percent_nulls)

cols_to_remove = names(percent_na_columns[which(percent_na_columns > 20)])

train_data = train_data[,colnames(train_data) %nin% cols_to_remove]
    
str(train_data)
train_data = train_data %>%
  mutate_if(is.factor, as.numeric)

str(train_data)

mice_train_model = mice(train_data)

imputed_train_data = complete(mice_train_model)
train_data = train_data %>%
                na.omit()


index = createDataPartition(train_data$SalePrice, p = .70, list = FALSE)

train_data_sample = train_data[index, ]
validation_data_sample = train_data[-index, ]


nrow(train_data_sample)
nrow(validation_data_sample)


train_features = train_data_sample[,colnames(train_data_sample) %nin% c("SalePrice")]
train_target = train_data_sample[,c("SalePrice")]

validation_features = validation_data_sample[,colnames(train_data_sample) %nin% c("SalePrice")]
validation_target = validation_data_sample[,c("SalePrice")]


# pp = preProcess(train_features, "range")
# 
# train_features = predict(pp, train_features)
# validation_features = predict(pp, validation_features)


# str(train_features)
# 
# train_features
# 
encoded_train_features = model.matrix(~.+0,data = train_features[, !names(train_features) %in% c("SalePrice")])
encoded_validation_features = model.matrix(~.+0,data = validation_features[, !names(validation_features) %in% c("SalePrice")])


library(keras)
model = keras_model_sequential() %>%   
  layer_dense(units = 100, activation = "relu", input_shape = ncol(train_features)) %>%
  # layer_dropout(0.3) %>%
  layer_dense(units = 50, activation = "relu") %>%
  #  layer_dropout(0.3) %>%
  #  layer_dense(units = 50, activation = "relu") %>%
  # layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = "linear")


adam = optimizer_adam(lr = 0.01)

compile(model, loss = "mse", optimizer = adam)

history = fit(model,  encoded_train_features, train_data_sample$SalePrice, epochs = 20, batch_size = 10, validation_split = 0.1)
# summary(model)


pred = predict(model, encoded_validation_features)


mean(abs(validation_data_sample$SalePrice - pred) / validation_data_sample$SalePrice)


## predicting on test data

cols_to_subset = colnames(train_data)[colnames(train_data) %nin% c("SalePrice")]

test_data = read.csv("test.csv")

test_data = test_data[,cols_to_subset]


test_data = test_data %>%
  mutate_if(is.factor, as.numeric)


str(test_data)


sapply(test_data, percent_nulls)

mice_model_test = mice(test_data)

test_data_imputed = complete(mice_model_test)

sapply(test_data_imputed, percent_nulls)
nrow(test_data_imputed)

test_data_imputed = test_data_imputed[,colnames(train_data) %nin% c("SalePrice")]

previous_na_action = options('na.action')
options(na.action='na.pass')

encoded_test_features = model.matrix(~.+0,data = test_data_imputed)


pred_test = predict(model, encoded_test_features)

length(pred_test)

options(na.action=previous_na_action$na.action)

nn_submission = data.frame(Id = test_data_imputed$Id, SalePrice = pred_test)

nn_submission[which(is.na(nn_submission)),]

write.csv(nn_submission, "nn_submission.csv", row.names = F)


