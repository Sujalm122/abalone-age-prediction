#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries for data manipulation, visualization, and machine learning.
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtb
import seaborn as sns


# In[2]:


# Upload the 'abalone.csv' file to the Colab environment.
from google.colab import files
uploaded = files.upload()


# In[3]:


# Read the uploaded CSV file into a pandas DataFrame.
abalone=pd.read_csv("abalone.csv")


# In[4]:


# Display the first 5 rows of the DataFrame to get an initial look at the data.
abalone.head()


# In[5]:


# Display the last 5 rows of the DataFrame.
abalone.tail()


# In[6]:


# Get a concise summary of the DataFrame, including data types and non-null values.
abalone.info()


# In[7]:


# Generate descriptive statistics for numerical columns in the DataFrame.
abalone.describe()


# In[8]:


# Check for null (missing) values in each column of the DataFrame.
abalone.isnull()


# In[9]:


# Visualize missing values using a heatmap to quickly identify any gaps in the data.
sns.heatmap(abalone.isnull(),cmap="magma")
mtb.title("Missing values")


# In[10]:


# Display the column names of the DataFrame.
abalone.columns


# In[11]:


# Create a bar plot to visualize the distribution of the 'Sex' column.
abalone['Sex'].value_counts().plot(kind='bar', color=['orange','red','green'])
mtb.title('Distribution of sex')
mtb.xlabel('Category')
mtb.ylabel('count')
mtb.show()


# In[12]:


# Encode the 'Sex' categorical column into numerical values (M:0, F:1, I:2) and visualize the encoded distribution.
abalone['Sex'] = abalone['Sex'].map({ 'M': 0, 'F': 1, 'I': 2})
sns.countplot(x='Sex',data=abalone)
mtb.title('Encoded Sex Feature')
mtb.show()


# In[13]:


# Compute and visualize the correlation matrix between the DataFrame's numerical features using a heatmap.
corr = abalone.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
mtb.title("Correlation Matrix")
mtb.show()


# In[14]:


# Display the distribution of the 'Rings' (target variable) using a histogram with a Kernel Density Estimate.
sns.histplot(abalone['Rings'], bins=20, kde=True, color='yellow')
mtb.title('Distribution of Rings (Target Variable)')
mtb.show()


# In[15]:


# Create a scatter plot to show the relationship between 'Length' and 'Rings'.
sns.scatterplot(x='Length', y='Rings', data=abalone)
mtb.title('Length vs Rings')
mtb.show()


# In[16]:


# Split the data into training and testing sets (80% train, 20% test) for model evaluation.
from sklearn.model_selection import train_test_split
x=abalone.drop('Rings',axis=1)
y=abalone['Rings']
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)


# In[17]:


# Scale the numerical features using StandardScaler to normalize the data, which can improve model performance.
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[18]:


# Train and evaluate multiple regression models (Linear Regression, Ridge, Decision Tree, Random Forest).
# Print their Mean Squared Error (MSE) and R-squared (R2) scores for comparison.
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('Random Forest', RandomForestRegressor(random_state=42))
]

results = {}
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f'{name}: MSE = {mse:.2f}, R2 = {r2:.2f}')


# In[19]:


# Visualize the R2 scores of the different regression models using a bar plot for easy comparison of their performance.
mtb.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
mtb.title('Comparison of Regression Models (R2 Score)')
mtb.ylabel('R2 Score')
mtb.xticks(rotation=30)
mtb.show()


# In[20]:


# Initialize, train, and evaluate a Decision Tree Regressor.
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
y_pred_dtr = dtr.predict(x_test)
print("Decision Tree MSE:", mean_squared_error(y_test, y_pred))
print("Decision Tree R2:", r2_score(y_test, y_pred))


# In[21]:


# Initialize, train, and evaluate a Random Forest Regressor.
dtr = RandomForestRegressor()
dtr.fit(x_train, y_train)
y_pred_dtr = dtr.predict(x_test)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred))
print("Random Forest R2:", r2_score(y_test, y_pred))


# In[22]:


# Define a function `prediction_age` that takes abalone features as input and predicts the number of rings using the trained model.
def prediction_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight):
    features = np.array([[Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight]])
    pred = dtr.predict(features).reshape(1, -1)
    return pred[0]


# In[23]:


# Demonstrate the `prediction_age` function with a sample input and print the predicted rings.
Sex = 2
Length = 0.6
Diameter = 0.45
Height = 0.15
Whole_weight = 1.2
Shucked_weight = 0.6
Viscera_weight = 0.3
Shell_weight = 0.4

prediction = prediction_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight)
print("Predicted Rings (Age):", prediction)


# In[24]:


# Demonstrate the `prediction_age` function with another sample input and print the predicted rings.
Sex = 3
Length = 0.4
Diameter = 0.59
Height = 0.25
Whole_weight = 1.8
Shucked_weight = 0.9
Viscera_weight = 0.1
Shell_weight = 0.7

prediction = prediction_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight)
print("Predicted Rings (Age):", prediction)

