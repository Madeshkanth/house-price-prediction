import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('/content/housing.csv')
data.head()
data.info()
data.dropna(inplace=True)
data.info()
data.describe()
from sklearn.model_selection import train_test_split
x=data.drop(['median_house_value'],axis=1)#'median_house_value' is a target value
y=data['median_house_value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
train_data=x_train.join(y_train)
train_data=x_train.join(y_train)
train_data.hist(figsize=(15,10))
print(train_data)
train_data_numeric = train_data.select_dtypes(include=['number'])
train_data_numeric.corr()
plt.figure(figsize=(15,8))
sns.heatmap(train_data_numeric.corr(),annot=True,cmap='YlGnBu')
train_data['total_rooms']=np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)
train_data['population']=np.log(train_data['population']+1)
train_data['households']=np.log(train_data['households']+1)
train_data.hist(figsize=(15,10))
train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
plt.figure(figsize=(15,10))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')
plt.figure(figsize=(15,8))
sns.scatterplot(data=train_data,x="latitude",y='longitude',hue="median_house_value",palette="coolwarm")
train_data['bedroom_ratio']=train_data['total_bedrooms']/train_data['total_rooms']
train_data['household_rooms']=train_data['households']/train_data['total_rooms']
plt.figure(figsize=(15,10))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')
from sklearn.linear_model import LinearRegression
x_train,y_train=train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']
reg=LinearRegression()
reg.fit(x_train,y_train)
import numpy as np
import pandas as pd

# Assuming x_test and y_test are already defined

# Join x_test and y_test into test_data
test_data = x_test.join(y_test)

# Apply log transformation to the specified columns
test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

# Perform one-hot encoding for the 'ocean_proximity' column
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)

# Create new feature columns
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households']

# Convert boolean columns to integers (1 and 0)
test_data[['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']] = test_data[['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']].astype(int)

# Separate features (x_test) and target (y_test)
x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

# Display the result
print(x_test)
from sklearn.ensemble import RandomForestRegressor
x_train,y_train=train_data.drop(['median_house_value'],axis=1),train_data['median_house_value']
reg=RandomForestRegressor()
reg.fit(x_train,y_train)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Make predictions on the test set
y_pred = reg.predict(x_test)

# Calculate R² score (regression accuracy score)
r2 = r2_score(y_test, y_pred)

# Display R² score
print(f"R² Score: {r2}")

# Optionally, display RMSE if you want
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Display the first few predictions and actual values
print(f"Predictions: {y_pred[:5]}")
print(f"Actual Values: {y_test[:5].values}")
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 4],  # Corrected to 'min_samples_split'
    "max_depth": [None, 4, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(x_train, y_train)
