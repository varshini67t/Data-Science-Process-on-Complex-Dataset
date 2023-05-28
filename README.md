# Data-Science-Process-on-Complex-Dataset
## AIM

       To Perform Data Science Process on a complex dataset and save the data to a file. 
## ALGORITHM

       STEP 1: Read the given Data 

       STEP 2: Clean the Data Set using Data Cleaning Process 

       STEP 3: Apply Feature Generation/Feature Selection Techniques on the data set 

       STEP 4: Apply EDA /Data visualization techniques to all the features of the data set
## CODE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

df = pd.read_csv('/content/ds_salaries.csv',encoding='windows-1252')
df.head()


df.isnull().sum()

plt.figure(figsize=(8,8))
plt.title("Data with Outliers")
df.boxplot()
plt.show()


plt.figure(figsize=(8,8))
cols = ['work_year','salary','salary_in_usd','remote_ratio']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

sns.barplot(x=df['work_year'], y=df['salary'], hue=df['work_year'])
plt.legend(loc="center")
plt.title("Highest salary Amount in the year")
plt.show()


sns.boxplot(x=df['employment_type'], y=df['salary'], hue=df['employment_type'])
plt.title("Average salary  given by employee type")


df["tip_percent"] = df["salary"] / df["salary_in_usd"]
sns.scatterplot(x=df['remote_ratio'],y=df['tip_percent'],data=df)
plt.title("salary percentage by salary in US")


sns.boxplot(x=df['experience_level'], y=df['salary'],hue=df['experience_level'])
plt.title("salary based on experience")


sns.scatterplot(x=df['salary_currency'],y=df['salary_in_usd'],hue=df['salary_currency'])
plt.legend(loc="best")
plt.title("Salary currency in usd")


sns.histplot(data=df, x="salary", hue="experience_level", element="step", stat="density")
plt.title("Distribution of salary based on experience level")
plt.show()


sns.barplot(x=df['company_size'],y=df['salary'],hue=df['company_size'])
plt.title("salary based on company size")
plt.show()


sns.boxplot(x="salary_in_usd", y="salary_currency", data=df)
plt.title("usd salary based in different currencies")
plt.show()


sns.violinplot(x="work_year", y="salary", data=df)
plt.title("Tip Amount by Time of Day")
plt.show()


sns.scatterplot(x="salary", y="salary_in_usd", data=df)
plt.title("Correlation between salary and salary in usd")
plt.show()

## OUTPUT
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/b3b1bfdd-61a0-442d-806a-0b43514b42f6)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/aa7fa643-7716-47aa-b392-8f5a043b95e7)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/00af19cb-5775-4be0-8462-fb137bb694a2)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/10bcddac-a861-4c59-b85f-6eef341163ae)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/b633490f-744f-4cca-875a-d93bb5011e7a)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/2fd0834c-e5f0-4414-8e07-744be3aa2346)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/89384cc0-145a-47c3-b7dd-fbeb7b4fb5c8)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/d2a20cbf-931b-49f6-a177-65e1aa801b04)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/0c7e1347-bec4-4f42-8cb0-143bfddc859e)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/8dd25332-8fa1-43fc-9174-a068fd3986d5)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/8f10f3bc-1d18-40cc-8770-11080a7c3a85)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/8c950250-3e31-47a9-b447-129ad43bbda0)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/a11a1e8b-fef7-4952-b4f6-8298fec299ae)
![image](https://github.com/varshini67t/Data-Science-Process-on-Complex-Dataset/assets/107982953/67778889-e0e0-4118-a657-84f1f2679d4b)


## RESULT

    Thus we have performed data visualization operations on complex dataset.
