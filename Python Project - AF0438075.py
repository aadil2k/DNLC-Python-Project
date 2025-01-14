#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Dataset

# ## Data Manipulation Part

# In[2]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#importing the data
df = pd.read_csv('D:/Data Sets/Python/Heart.csv')
df


# In[4]:


#getting top 5 records
df.head()


# In[5]:


#getting last 5 records
df.tail()


# In[6]:


#getting the shape of data
df.shape


# In[7]:


#getting the size of data
df.size


# In[8]:


#information abut the data
df.info()


# In[9]:


#Checking the data types of each column
df.dtypes


# In[10]:


#checking the null values
df.isnull()


# In[11]:


#counting the null values in every column
df.isna().sum()


# In[12]:


#getting the statistical discription of the data
df.describe()


# In[13]:


#Checking the dupicate rows in the data set
df.duplicated().any()


# In[14]:


#Removing the duplicate rows
df = df.drop_duplicates()
df.head()


# #### Get minimum and maximum age along with mean, meadian, mode and Standard Deviation
# 

# In[15]:


#Calculate Mean
mean = df['age'].mean()
#Calculate Median
median = df['age'].median()
#Calculate Mode
mode = df['age'].mode().iloc[0]
#Calculate standard deviation
std = df['age'].std()
#Calculate Minimum values
minimum = df['age'].min()
#Calculate Maximum values
maximum = df.age.max()
print(f" Mean of Age : {mean}")
print(f" Median of Age : {median}")
print(f" Mode of Age : {mode}")
print(f" Standard deviation of Age : {std:.2f}")
print(f" Maximum of Age : {maximum}")
print(f" Minimum of Age : {minimum}")


# #### Check how many males and females are in your dataset
# 

# In[16]:


print(df['sex'].value_counts())


# ## Data Visualization Part

# #### Exploring Relationships: Heatmaps with Python for Data Visualization

# In[18]:


#Get co-relationship of your data
df.corr()


# #### Create a heatmap to show the corelationship of data
# 

# In[19]:


plt.figure(figsize=(16,6))
sns.heatmap(df.corr(),annot = True)


# In[20]:


#Get the Numbers affacted and not affected by heart disease
# Count the number of people affected and not affected by heart disease
affected_count = (df['target'] == 1).sum()
not_affected_count = (df['target'] == 0).sum()

# Print results
print(f"Number of people affected by heart disease: {affected_count}")
print(f"Number of people not affected by heart disease: {not_affected_count}")


# #### Create bar chart to show the Numbers affacted and not affected by heart disease.

# In[21]:


df['target'].value_counts().plot(kind = 'bar')
plt.show


# In[22]:


#Count the occurance of each chest Pain Type
chest_pain_counts = df['cp'].value_counts()
chest_pain_counts


# #### Show the Gender Distribution in dataset using charts

# In[23]:


# Count the occurrences of each gender
gender_counts = df['sex'].value_counts()
gender_counts


# In[24]:


# Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution - Bar Chart')
plt.show()

# Pie Chart
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'pink'])
plt.title('Gender Distribution - Pie Chart')
plt.show()


# #### Show the Distribution of heart dieasese among males and females

# In[25]:


# Group by gender and target, then count occurrences
gender_target_counts = df.groupby(['sex', 'target']).size().unstack()
gender_target_counts


# In[26]:


# Plotting
bar_width = 0.35
index = np.arange(len(gender_target_counts))

plt.figure(figsize=(6, 4))

# Bars for each target category
plt.bar(index, gender_target_counts[0], bar_width, label='No Heart Disease', color='skyblue')
plt.bar(index + bar_width, gender_target_counts[1], bar_width, label='Heart Disease', color='orange')

# Adding labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender-wise Heart Disease Distribution')
plt.xticks(index + bar_width / 2, gender_target_counts.index)
plt.legend()

plt.tight_layout()
plt.show()


# #### Which age group is most affected by heart desiese?

# In[43]:


# Define age groups
bins = [20, 30, 40, 50, 60, 70, 80]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Aggregate data by age group and count heart disease cases
# Assuming 'HeartDisease' is a binary column where 1 indicates heart disease
age_group_affected = df[df['target'] == 1].groupby('AgeGroup').size()

# Find the age group with the most cases
most_affected_group = age_group_affected.idxmax()
most_affected_count = age_group_affected.max()

print(f"The age group most affected by heart disease is {most_affected_group} with {most_affected_count} cases.")

# Visualization
plt.figure(figsize=(8, 5))
age_group_affected.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Heart Disease Cases by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Number of Cases', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# #### Show Chest Pain distribution in Heart Disease vs Non Heart Disease by using graph

# In[28]:


# Group data by chest pain type and target
chest_pain_counts = df.groupby(['cp', 'target']).size().unstack(fill_value=0)

# Plotting
bar_width = 0.35
x = np.arange(len(chest_pain_counts))

plt.figure(figsize=(10, 6))

# Bars for heart disease and no heart disease
plt.bar(x - bar_width / 2, chest_pain_counts[0], width=bar_width, color='skyblue', label='No Heart Disease')
plt.bar(x + bar_width / 2, chest_pain_counts[1], width=bar_width, color='orange', label='Heart Disease')

# Adding labels, title, and legend
plt.xlabel('Chest Pain Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Chest Pain Distribution in Heart Disease and Non-Heart Disease', fontsize=14)
plt.xticks(x, chest_pain_counts.index, fontsize=10)
plt.legend()

plt.tight_layout()
plt.show()


# #### Percentage of different chest pain type

# In[45]:


chest_pain_col = 'cp'  # Adjust based on your dataset

# Count the occurrences of each chest pain type
chest_pain_counts = df[chest_pain_col].value_counts()
    
 # Calculate percentages
chest_pain_percentages = chest_pain_counts / chest_pain_counts.sum() * 100
    
# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(chest_pain_percentages, labels=chest_pain_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Percentage of Different Chest Pain Types', fontsize=16)
plt.show()


# #### What is the Resting blood pressure (trestbps) Data Distribtion by graph

# In[ ]:


df['trestbps'].hist()


# In[30]:


df.columns


# #### Show the Serum Cholestrol (Chol) Data Distribution by graph

# In[31]:


df['chol'].hist()


# #### What is the distribution of resting blood pressure among patients?

# In[32]:


# Replace 'RestingBP' with the actual column name if different
if 'trestbps' in df.columns:
    # Calculate descriptive statistics
    print("Descriptive Statistics for Resting Blood Pressure:")
    print(df['trestbps'].describe())
    
    # Plot the distribution using a histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df['trestbps'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Resting Blood Pressure', fontsize=16)
    plt.xlabel('Resting Blood Pressure (mmHg)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Plot the distribution using a boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['trestbps'], color='lightgreen')
    plt.title('Boxplot of Resting Blood Pressure', fontsize=16)
    plt.xlabel('Resting Blood Pressure (mmHg)', fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("The column for resting blood pressure (e.g., 'RestingBP') is not found in the dataset.")


# In[40]:


# Scatter plot for Cholesterol vs. Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='chol', hue='target', data=df, palette='coolwarm', alpha=0.7)
plt.title('Cholesterol vs. Age', fontsize=16)
plt.xlabel('age', fontsize=14)
plt.ylabel('chol', fontsize=14)
plt.grid(alpha=0.5)
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()


# # Conclusion of the Project
# In this project, we analyzed a heart-related dataset to understand key health metrics and their relationship to heart disease. By exploring age groups, we identified that middle-aged individuals were the most affected by heart disease. The distribution of resting blood pressure and cholesterol revealed significant variability, with some outliers indicating potential risk factors. A positive correlation between age and cholesterol levels highlighted the growing risk of heart conditions with age. Gender-based analysis showed disparities in heart disease prevalence, emphasizing the need for targeted awareness. Visualizations such as histograms, boxplots, and scatter plots effectively illustrated these trends, providing actionable insights for preventive measures. This project underscores the importance of monitoring key health indicators and adopting healthier lifestyles to mitigate heart disease risks.
# 
# ### Key Insights:
# 
# Demographic Trends: (e.g., The majority of patients with heart conditions were in the age group of 40–60 years. Men were more affected than women in this dataset.)
# 
# Health Metrics: (e.g., High cholesterol levels and blood pressure were observed as common factors among patients with heart conditions.)
# 
# Relationships: (e.g., A positive correlation was found between age and cholesterol levels, indicating an increased risk with age.)
# 
# Risk Factors: (e.g., Diabetes, and obesity were significant contributors to heart disease in the dataset.)

# In[54]:


df.shape


# In[ ]:




