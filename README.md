# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
 import pandas as pd
 import io
 from sklearn.preprocessing import StandardScaler
 from sklearn.preprocessing import MinMaxScaler
 from sklearn.model_selection import train_test_split
 #read the dataset
 df=pd.read_csv('Churn_Modelling data.csv')
 df
 #drop unwanted columns
 df.drop('RowNumber',axis=1,inplace=True)
 df.drop('CustomerId',axis=1,inplace=True)
 df.drop('Surname',axis=1,inplace=True)
 df.drop('Geography',axis=1,inplace=True)
 df.drop('Age',axis=1,inplace=True)
 df.drop('Gender',axis=1,inplace=True)
 df
#checking for null, duplicates, outliers in lasrt column
df.isnull().sum()

df.duplicated()

df['Exited'].describe()
#normalising data to normal distribution
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['CreditScore','Tenure','Balance',
'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
 df2
 #split dataset
 x=df2.iloc[:,:-1].values #all rows from all except last column
 x
 y=df2.iloc[:,-1].values #all rows from only last column
 y
 ##creating training and test data
 X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
 print(X_train)
 print("Size of X_train: ",len(X_train))
 print(X_test)
 print("Size of X_test: ",len(X_test))

## OUTPUT:
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/8c755d89-a806-4471-aa6d-2b9c0486f4ac)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/a827be7f-b336-42f8-9d77-fcc83f3a78a9)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/abf916df-7a0a-4f51-8669-0c5bb16c3ba5)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/21fc4229-b256-44c3-9f07-1ee8787750d9)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/89cdd1b1-74ff-4315-b964-145394edc673)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/86e8702a-13bd-4f51-b77d-6cd2a463745c)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/2e429950-4009-4219-9696-58feb03f28f5)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/672caf94-4ccd-4bbb-b904-1c392e1d048c)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/72e6ffec-c899-4172-baf9-d4380fbd3796)
![image](https://github.com/DEEPAK22003907/Ex.No.1---Data-Preprocessing/assets/119404520/01122b67-2b07-4424-aa84-ae0c5d8f0064)


## RESULT
Thus, the Data preprocessing is performed over a data set successfully.
