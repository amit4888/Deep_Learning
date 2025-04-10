#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Library

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler     ## Standardization Technique


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Import Warning Library
import warnings
warnings.filterwarnings("ignore")


# # Read The Dataset

# In[3]:


df = pd.read_csv("Prodigy University Dataset.csv")


# In[4]:


#  Display 5 rows of the dataset

df.head()


# In[5]:


# Shape of the dataset
df.shape


# In[6]:


#  Create scatter plot
plt.scatter(df['sat_sum'], df['fy_gpa'], color='blue')

plt.title("SAT_SCORE Vs Final Year GPA")
plt.xlabel("SAT_SCORE")
plt.ylabel("FY_GPA")

plt.show()


# In[7]:


# Conclusion : 
# There is a positive linear relationship – as SAT scores increase, the final year GPA tends to increase as well.


# In[8]:


plt.scatter(df['hs_gpa'], df['fy_gpa'], color='red')

plt.title("High School GPA Vs Final Year GPA")
plt.xlabel("HS GPA")
plt.ylabel("FY_GPA")

plt.show()


# In[9]:


# Conclusion :
# # This also shows a positive linear relationship, but the data seems less spread out compared to the SAT plot


# # Feature Engineering

# # Data Cleaning

# In[10]:


# Check Missing Values
df.isnull().sum()


# In[11]:


# Check Dtype & Non-Null Values
df.info()


# In[12]:


# Descriptive Statistics of the dataset
df.describe()


# # EDA (Exploratory Data Analysis)

# In[13]:


sns.boxplot(x=df['sat_sum'], color='orange')

plt.title('Boxplot of SAT SCORE')
plt.xlabel('SAT SCORE')

plt.show()


# In[14]:


sns.boxplot(x=df['hs_gpa'], color='orange')

plt.title('Boxplot of High School GPA')
plt.xlabel('HS_GPA')

plt.show()


# In[15]:


plt.figure(figsize = (16,5))
plt.subplot(1,2,1)
sns.distplot(df['sat_sum'])

plt.subplot(1,2,2)
sns.distplot(df['hs_gpa'])

plt.show()


# In[16]:


# For SAT_SCORE  -->  graph looks close to normal, so Z-score is valid here.

# For HS_GPA --> Since the distribution is bimodal and skewed, Z-score may not be very effective or reliable.


# # Outlier Removal / Detection

# In[17]:


# An outlier is a data point that significantly differs from other observations in the dataset, 
# potentially indicating variability, error, or a rare event.


# # Z - Score Technique

# In[18]:


print(f"Mean Value of SAT_Score {df['sat_sum'].mean()}")
print(f"Std Value of SAT_Score {df['sat_sum'].std()}")
print(f"Min Value of SAT_Score {df['sat_sum'].min()}")
print(f"Max Value of SAT_Score {df['sat_sum'].max()}")


# In[19]:


# Finding the boundary value

print("Upper boundary",df['sat_sum'].mean() + 3*df['sat_sum'].std())
print("Lower boundary",df['sat_sum'].mean() - 3*df['sat_sum'].std())


# In[20]:


# Finding the outliers

df[(df['sat_sum'] > 584.76) | (df['sat_sum'] < 241.86)]


# # # Treat Outlier

# In[21]:


# Trimming

new_df = df[(df['sat_sum'] < 584.76) & (df['sat_sum'] > 241.86)]
new_df.shape


# # Data Preprocessing

# In[22]:


from sklearn.preprocessing import MinMaxScaler

# Create scaler objects
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Apply and overwrite same column
new_df['sat_sum'] = std_scaler.fit_transform(new_df[['sat_sum']])
new_df['hs_gpa'] = minmax_scaler.fit_transform(new_df[['hs_gpa']])


# In[23]:


# Here, we can see Our data has converted into fixed range.
round(new_df.describe(),2)


# In[24]:


# Seperate Dependent and Independent Variable

X = new_df.drop(columns=["fy_gpa"])
y = new_df["fy_gpa"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Build the ANN Model

# In[64]:


from tensorflow.keras.optimizers import Adam


# In[65]:


model = Sequential([
    Dense(1, input_dim=2, activation='linear')
    ])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Predict
predictions = model.predict(X_test)
print("First 5 Predicted GPA values:", predictions[:5].flatten())


# In[66]:


model.summary()


# In[67]:


from sklearn.metrics import r2_score, mean_squared_error
# Get predictions
y_preds = model.predict(X_test)

# R² Score
r2 = r2_score(y_test, y_preds)
print("R2 Score (Accuracy):", r2)

# Mean Squared Error
mse = mean_squared_error(y_test, preds)
print("Mean Squared Error:", mse)


# # Optimizers

# In[43]:


from keras.optimizers import SGD, RMSprop

# Simple ANN model
model = Sequential([
    Dense(16, input_dim=2, activation='relu'),
    Dense(1, activation='linear')
])


# In[44]:


# SGD Optimizer
sgd_optimizer = SGD(learning_rate=0.01)

model.compile(optimizer=sgd_optimizer, loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=30)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)


# In[45]:


# RMSProp Optimizer
rms_optimizer = RMSprop(learning_rate=0.001)

model.compile(optimizer=rms_optimizer, loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=30)


# In[48]:


from sklearn.metrics import r2_score

preds = model.predict(X_test)
r2 = r2_score(y_test, y_preds)
print("R2 Score with this optimizer:", r2)


# In[ ]:




