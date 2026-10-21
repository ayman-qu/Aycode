import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 

df = pd.read_csv("C:\\Users\\Dell\\Downloads\\Dataset files\\car_price_dataset.csv")

numeric_col = df.select_dtypes(include = ["Int64", "float64"])
correlation_matrix = numeric_col.corr() # corr() works for only numeric columns


# HeatMap 
# sns.heatmap(correlation_matrix, cmap = "coolwarm", linewidth = .5, annot = True)
# plt.title("Correlation Heatmap")
# plt.show()

# Histogram 
# df["Price"].plot.hist(bins = 20, color = "teal")
# plt.title("Distributions of Car Prices")
# plt.show()

# Scatterplot from Matplotlib for mileage and prices
# plt.scatter(df["Mileage"], df["Price"], alpha = 0.5)
# plt.xlabel("Mileage")
# plt.ylabel("Price")
# plt.title("Scatterplot of Prices and Year")
# plt.show()

# Bar Plot of brands and prices
# mean_Price = df.groupby("Brand")["Price"].mean().sort_values(ascending=False).tail()
# mean_Price.plot(kind = "bar", fontsize  = 6.8)
# plt.xlabel("Brand")
# plt.ylabel("Average Price")
# plt.title("Mean Prices by Brand")
# plt.show()

# Box Plot
# sns.boxplot(x = "Price", data = df, y = "Brand", hue = "Year")
# plt.title("Boxplot of Car prices")
# plt.show()

# Bar plot for most common car models 
# car_models = df["Model"].value_counts().head(10)
# car_models.plot(kind = "bar", fontsize = 6.8)
# plt.title("Car models")
# plt.xlabel("car models")
# plt.ylabel("Frequncy")
# plt.show()

# Scatterplot for Engine_size and price
# plt.scatter(df["Engine_Size"], df["Price"], alpha = 0.2)
# plt.title("Scatterplot of Engine Size and Price")
# plt.xlabel("Engine Size")
# plt.ylabel("Price")
# plt.show()

# Histogram for low - Mileage cars
# df[df["Mileage"] < 100000]["Mileage"].plot.hist(bins = 10, color = "darkgreen", rwidth = 0.8)
# plt.title("Low- Mileage Cars", color = "teal")
# plt.xlabel("Mileage")
# plt.ylabel("Frequency")
# plt.show()

# Bar plot of Car Count in each city 
# count_model = df["Model"].value_counts().sort_values(ascending = True).head(10)
# count_model.plot(kind = "bar", fontsize = 6.8)
# plt.title("Model count of Cars")
# plt.xlabel("Model")
# plt.ylabel("Frequency")
# plt.show()

# Grouping by the fuel type and prices 
# fuel_price = df.groupby("Fuel_Type")["Price"].mean() 
# print(fuel_price.sort_values(ascending = False))

# Grouping by Year and Mileage
# median_mileage = df.groupby("Year")["Mileage"].median()
# print(median_mileage.sort_values(ascending = False).head(5))

# Groupby by Brand and Fuel and Engine_size
# max_engine_size = df.groupby(["Brand","Fuel_Type"])["Engine_Size"].max()
# print(max_engine_size)

# Groupby by Brand and Doors and transmission
# door_count = df.groupby(["Brand", "Model", "Transmission"])["Doors"].count().sort_values(ascending = True)
# print(door_count)

# Groupby by Brand and model and mean price
# avg_price = df.groupby(["Brand", "Model"])["Price"].mean()
# print(avg_price)

# Car count by year groupby
# car_count = df.groupby("Year")["Brand"].count()
# print((car_count).sort_values(ascending = False))

# Avg mileage by brand and fuel type
# avg_mileage = df.groupby(["Brand", "Fuel_Type"])["Mileage"].mean()
# print(avg_mileage)

# Groupby by min_price
# min_price = df.groupby(["Mileage", "Fuel_Type"])["Price"].mean().
# print(min_price)

# Bar plot for brand and price 
# df.groupby("Brand")["Price"].mean().head(10).sort_values(ascending = False).plot(kind = "bar")
# plt.xlabel("Car brand") 
# plt.ylabel("Avearge price")
# plt.title("Average Price by car Brand")
# plt.show()

# Pie plot 
# df.groupby("Fuel_Type")["Price"].max().plot(kind = "pie", autopct = "%1.1f%%")
# plt.title("Maximum Price by Fuelt type")
# plt.show()

# Feature engineering
X = df.iloc[:,:-1] # X contains all columns except for the last column
Y = df.iloc[:, -1] # Y contains last column "price"
# drop outs 
X.drop(columns = ["Model", "Doors", "Owner_Count", "Transmission"], inplace = True)

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()     # Label Encoding 

X["Brand"] = label_encoder.fit_transform(X["Brand"])
X["Fuel_Type"] = label_encoder.fit_transform(X["Fuel_Type"])

# NOW, time for training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train) 
scaled_X_test = scaler.transform(X_test) # fit_tranform doesnt apply to testing data

# Modeling 
from sklearn.metrics import mean_squared_error, mean_absolute_error #(MAE, MSE)
import numpy as np

def modelresults(predictions):
    print("Mean Absolute Error on model is {}".format(mean_absolute_error(Y_test, predictions)))
    print("Root Mean Squared Error on model is {}".format(np.sqrt(mean_squared_error(Y_test, predictions))))

# Machine Learning ( Which model gives best performance? )
# Linear regression 
from sklearn.linear_model import LinearRegression  

LR = LinearRegression()
LR.fit(scaled_X_train, Y_train)

pred = LR.predict(scaled_X_test)
        
modelresults(pred) # the result came and will be compared to other models 

# HyperParameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

svrmodel = SVR() # Support Vector Regression

param_gridsvr = {"C": [0.001, 0.01, 0.1, 0.5], "kernel": ["linear", "rbf", "poly"], "gamma" : ["scale", "auto"], "degree" : [2, 3, 4, 5]}

grid_svr = GridSearchCV(svrmodel, param_gridsvr) # taking too long for the computer ( skipped) 

svrmodel.fit(scaled_X_train, Y_train)

predssvr = svrmodel.predict(scaled_X_test)

modelresults(predssvr) # the result came and will be compared

from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(oob_score=True)
rfrmodel.fit(scaled_X_train, Y_train)
print("Score =", rfrmodel.oob_score_)

rfrpredicts = rfrmodel.predict(scaled_X_test)

modelresults(rfrpredicts) # the result came and this is the best result compared to others (14k mse, 30k rmse)

# so From this project we learned that Random Forest regressor turns out to be the Best Model, in terms of 
# MAE AND RMSE 
#  Linear Regression however gave a larger error and so did SVR model!
#  everything was new to me so i will be re doing it again.

# fix prompts and headings to error print () mae ,rmse for all models 