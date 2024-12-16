# Load necessary libraries
library(ggplot2)#For data visualization
library(lattice)
library(caret)
library(dplyr)
library(stringr)#For string operations

# Load the dataset
library(readr)
data <- read.csv("D:/MSc-2 Sem-3 Reserach  Project/Car details v3.csv")
View(data)
head(data,5)
#View the first few rows of the dataset
head(data)
#Inspect the structure of the dataset
str(data)
summary(data)
# Data Preprocessing 
## Convert categorical variables to factors
data$fuel <- as.factor(data$fuel)
data$seller_type <- as.factor(data$seller_type)
data$transmission <- as.factor(data$transmission)
data$owner <- as.factor(data$owner)

## Clean numerical columns (e.g., remove units from mileage, engine, max_power)
data$mileage <- as.numeric(sub(" kmpl", "", data$mileage))
data$engine <- as.numeric(sub(" CC", "", data$engine))
data$max_power <- as.numeric(sub(" bhp", "", data$max_power))
data$torque<-str_extract(data$torque,"\\d+\\.?\\d*")
data$torque<-as.numeric(data$torque)

#Handle missing values using median imputation
colSums(is.na(data))
preProcValues<-preProcess(data,method="medianImpute")
data<-predict(preProcValues,data)
colSums(is.na(data)) #Verify that missing values are handled

View(data)

# Select relevant columns for modeling
model_data <- dplyr::select(data,selling_price, year, km_driven, mileage, engine, max_power, seats, 
         fuel, seller_type, transmission, owner)

# Split the dataset into training and test sets
set.seed(123)
train_index <- createDataPartition(model_data$selling_price, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]


# Define the control method for training
train_control <- trainControl(method = "cv", number = 10)

# Fit models using caret

pre_process <- preProcess(train_data, method = c("center", "scale"))
train_data <- predict(pre_process, train_data)
test_data <- predict(pre_process, test_data)
## Linear Regression (lm)
lm_model <- train(selling_price ~ ., data = train_data, method = "lm", 
                  trControl = train_control)

## Support Vector Machine (svmLinear)
svm_model <- train(selling_price ~ ., data = train_data, method = "svmLinear", 
                   trControl = train_control)


# Normalize numeric columns
pre_process <- preProcess(train_data, method = c("center", "scale"))
train_data <- predict(pre_process, train_data)
test_data <- predict(pre_process, test_data)

## Decision Tree (rpart)
rpart_model <- train(selling_price ~ ., data = train_data, method = "rpart", 
                     trControl = train_control)

## K-Nearest Neighbors (knn)
knn_model <- train(selling_price ~ ., data = train_data, method = "knn", 
                   trControl = train_control)


# Evaluate models on the test set
lm_pred <- predict(lm_model, test_data)
svm_pred <- predict(svm_model, test_data)
rpart_pred <- predict(rpart_model, test_data)
knn_pred <- predict(knn_model, test_data)

# Calculate RMSE for each model
lm_rmse <- RMSE(lm_pred, test_data$selling_price)
svm_rmse <- RMSE(svm_pred, test_data$selling_price)
rpart_rmse <- RMSE(rpart_pred, test_data$selling_price)
knn_rmse <- RMSE(knn_pred, test_data$selling_price)

# Print RMSEs
cat("RMSEs for the models:\n")
cat("Linear Regression:", lm_rmse, "\n")
cat("SVM:", svm_rmse, "\n")
cat("Decision Tree:", rpart_rmse, "\n")
cat("KNN:", knn_rmse, "\n")

#Define function to calculate performance metrics
evaluate_model<-function(model,test_data,model_name)
{
  predictions<-predict(model,newdata=test_data)
  rmse<-RMSE(predictions,test_data$selling_price)
  r2<-R2(predictions,test_data$selling_price)
  cat(model_name,"RMSE:",rmse,"\n")
  cat(model_name,"R-squared:",r2,"\n\n")
  return(c(RMSE=rmse,R2=r2))
}

#Evaluate models
lm_metrics<-evaluate_model(lm_model,test_data,
                           "LinearRegression")
knn_metrics<-evaluate_model(knn_model,test_data,"K-NearestNeighbors")
rpart_metrics<-evaluate_model(rpart_model,test_data,"RegressionTree")
svm_metrics<-evaluate_model(svm_model,test_data,"SupportVectorMachine")

#Compile all metrics into a data frame
model_performance<-data.frame(
  Model=c("LinearRegression","KNN",
          "RegressionTree","SVM"),
  RMSE=c(lm_metrics["RMSE"],knn_metrics["RMSE"],rpart_metrics["RMSE"],svm_metrics
         ["RMSE"]),
  R_Squared=c(lm_metrics["R2"],knn_metrics["R2"],rpart_metrics["R2"],svm_metrics["R2"]))

#Display model performance
print(model_performance)


#Interpretation 

#Linear Regression:
# RMSE:  6.92 (very high).R-Squared: 0.4790
# Linear regression captures about 47.9% of the variance in selling_price,
# but the high RMSE indicates poor predictive accuracy. 
# The linear model struggles due to the dataset's potential nonlinear relationships or outliers.

#K-Nearest Neighbors (KNN):

# RMSE: 0.216(extremely low), R-Squared: 0.947
#KNN performs exceptionally well, capturing 94.7% of the variance with very low RMSE.
# This suggests that KNN is highly accurate for this dataset. However,
# its success may depend on proper data scaling and the choice of k.

# #Regression Tree (Decision Tree):
# RMSE: 0.517(moderate),R-Squared:0.695
#       The decision tree captures 69.5% of the variance with moderate predictive accuracy.
# Its performance suggests it can model some nonlinear relationships but is likely less robust than
# KNN for this dataset.

# Support Vector Machine (SVM):
# RMSE:6.82 (very high),R-Squared:0.537
#   SVM captures 53.7% of the variance but suffers from a very high RMSE.
#   The high error indicates that the SVM model might not be tuned appropriately
#   (e.g., kernel choice, parameter tuning) or that the dataset has challenges 
#   (e.g., high-dimensional or unscaled features).

  
#-----------------
##General Insights:
#------------------
# Best Model: KNN shows the best performance (lowest RMSE, highest R²),
# indicating its suitability for this dataset.
# Poor Models: Linear regression and SVM perform poorly, 
# likely due to assumptions of linearity or in appropriate parameter tuning.
# Decision Tree: Performs moderately well but is less accurate than KNN.


library(ggplot2)
library(corrplot)

# 1. Distribution of the selling price
ggplot(model_data, aes(x = selling_price)) +
  geom_histogram(binwidth = 50000, fill = "blue", color = "white") +
  labs(title = "Distribution of Selling Price", x = "Selling Price", y = "Frequency",)
      


#Interpretation 
##The histogram illustrates the distribution of the selling price in the dataset. Heres the interpretation:
##Right-Skewed Distribution:
# The selling price has a highly skewed distribution, with most of the data concentrated towards the lower end.
# This indicates that a majority of the cars are sold at lower prices (likely below ₹1,000,000).

## Frequency Peaks:
# The first few bins (representing lower price ranges) have the highest frequencies, showing that inexpensive cars dominate the dataset.

## Outliers in High Price Range:
# A small number of cars have significantly higher selling prices, as seen in the sparse bins extending toward ₹10,000,000.
# These could represent luxury or premium vehicles.



# 2. Scatterplot: Selling Price vs Km Driven
ggplot(model_data, aes(x = km_driven, y = selling_price)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  labs(title = "Selling Price vs Km Driven", x = "Km Driven", y = "Selling Price")
# The scatterplot shows the relationship between Selling Price and Km Driven for 
#the cars in the dataset. Here is the interpretation:
 
# 1. Negative Relationship
# There is a general downward trend, indicating a negative correlation between km_driven and selling_price.
# Cars with higher kilometers driven tend to have lower selling prices, reflecting depreciation due to usage.
# 2. Dense Cluster at Lower Values
# The majority of the cars are concentrated in the lower range of km_driven (below 200,000 km) and selling_price (below ₹2,500,000).
# This suggests that the dataset is dominated by moderately used, lower-priced vehicles.
# 3. Outliers
# A few cars have exceptionally high km_driven values (e.g., over 1,000,000 km)
#but still show low selling prices. 
#These could represent very old or heavily used vehicles.
# Some outliers in the selling price (above ₹7,500,000) exist despite relatively low mileage,
#  likely representing luxury cars or special models.


# 3. Boxplot: Selling Price by Fuel Type
ggplot(model_data, aes(x = fuel, y = selling_price)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Selling Price by Fuel Type", x = "Fuel Type", y = "Selling Price")

# 4. Correlation Matrix (numerical variables)
numeric_vars <- model_data %>% select(selling_price, year, km_driven, mileage, engine, max_power, seats)
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", title = "Correlation Matrix", tl.cex = 0.8)

# 5. Model Performance Comparison
rmse_data <- data.frame(
  Model = c("Linear Regression", "SVM", "Decision Tree", "KNN"),
  RMSE = c(lm_metrics["RMSE"], svm_metrics["RMSE"],rpart_metrics["RMSE"], knn_metrics["RMSE"]))
ggplot(rmse_data, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Performance (RMSE)", x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
