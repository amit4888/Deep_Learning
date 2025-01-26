#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Student Mental health.csv")


# In[3]:


df


# In[4]:


# shape of the dataset

df.shape


# In[5]:


# check null values and dtypes
df.info()


# In[6]:


# statistics of df

df.describe()


# In[15]:


# 1.Finding out the column number whose age is unknown to us


unknown_age = df[df['Age'].isnull()]

# Print rows with all columns
print("Rows where 'Age' is null:")
unknown_age


# In[20]:


# 2 Find row number from 40 to 45 

# Select rows from 40 to 45 (inclusive)
rows = df.iloc[40:46] 
rows


# In[21]:


# 3 Counting Courses which are opted by most of the student ?

df['What is your course?'].value_counts()


# In[11]:


# 4. #Counting number of Female and Male
    
df['Choose your gender'].value_counts()


# In[29]:


# Finding Mean Age of Female and Male

# Filter data for Female and Male
female_ages = df[df["Choose your gender"] == "Female"]["Age"]
male_ages = df[df["Choose your gender"] == "Male"]["Age"]

# Calculate mean using the formula
mean_female = female_ages.sum() / female_ages.count()
mean_male = male_ages.sum() / male_ages.count()

print("Mean Age of Female:", mean_female)
print("Mean Age of Male:", mean_male)


# In[15]:


# Finding Number of Students who are married and not married respectively

df['Marital status'].value_counts()


# In[56]:


# Percentage of students facing depression ?

total_students = df["Do you have Depression?"].count()
depress_students = df[df["Do you have Depression?"] == "Yes"]["Do you have Depression?"].count()

per_Dep = (depress_students / total_students) * 100

print(f"Percentage of students facing depression: {per_Dep}%")


# In[57]:


# My Unique Question

dep_students = df[df["Do you have Depression?"] == "Yes"]["Age"].count()


# In[58]:


#Percentage of students facing anxiety

total_students = df["Do you have Anxiety?"].count()
students_with_anxiety = df[df["Do you have Anxiety?"] == "Yes"]["Do you have Anxiety?"].count()

percentage_depression = (students_with_anxiety / total_students) * 100

print(f"Percentage of students facing depression: {percentage_depression}%")


# In[59]:


# #percentage of student having panic attacks


total_students = df["Do you have Panic attack?"].count()
students_with_panic_attack = df[df["Do you have Panic attack?"] == "Yes"]["Do you have Panic attack?"].count()

percentage_depression = (students_with_panic_attack / total_students) * 100

print(f"Percentage of students facing depression: {percentage_depression}%")


# In[ ]:





# In[ ]:




