#!/usr/bin/env python
# coding: utf-8

# In[291]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
calendar = pd.read_csv(r"My_files/Project1_data//calendar.csv")
reviews = pd.read_csv(r"My_files/Project1_data//reviews.csv")
listings=pd.read_csv(r"My_files/Project1_data/listings.csv")
calendar.head(10)


# In[292]:


reviews.head(10)


# In[293]:


listings.head(10)


# In[294]:


dates=calendar['date'].nunique()
missing_availability = sum(calendar.groupby('listing_id')['available'].count() != 365)

missing_prices=sum(calendar.groupby('listing_id')['price'].count()!=365)
print("There is data for",dates,",that means this data holds for two years which are 2020 and 2021.")
print("There are",missing_availability,"rows missing availability")
print("There are",missing_prices,"rows missing prices")


# In[295]:


print("In order to make dealing with cacategorical data easier, we should transform availability column to boolean")
calendar["available"]=calendar["available"]=='t'
calendar.head()


# In[296]:


#Calculate the percentage of availability for each day
#availability=calendar[['listing_id','available']].groupby('listing_id').sum().rename(columns={'available':'days_available'})
#availability = calendar[['listing_id', 'available']].groupby('listing_id').sum().rename(columns={"available": "days_available"})
#availability['percentage_availability']=availability.apply(lambda x: round ((x/365)*100,2))
availability = calendar[['listing_id', 'available']].groupby('listing_id').sum().rename(columns={"available": "days_available"})
availability['percentage_available'] = availability.apply(lambda x: round((x/365)* 100, 2))
availability.head()


# In[297]:


calendar.dtypes


# In[298]:


#Now we have to convert price type to numerical-float- 
calendar['price']=calendar['price'].astype(str).str.replace(',','').str.replace('$','')
calendar['price']=calendar['price'].apply(lambda x: float(x))
calendar.dtypes


# In[299]:


#Now caluclating the max,min,average of price and variation also
price=calendar[['listing_id','price']].groupby('listing_id').max().rename(columns={"price":"max_price"})
price["min_price"]=calendar[['listing_id','price']].groupby('listing_id').min()
price["avg_price"]=calendar[["listing_id","price"]].groupby('listing_id').mean().round(1)
price["price_variation"]=price["max_price"]-price["min_price"]
price.head()
#print(price.shape[0],price.shape[1])


# In[300]:


#calculationg the percantage of missing values 
missing=round(np.sum(price['max_price'].isnull())/price.shape[0]*100,2)
print("There is",missing,"values in max_price")


# In[301]:


#Now we have to prapare listings data
print("There is",listings.shape[1],"columns in it")
#For simplicity we will ignore the text filed
listings.dtypes


# In[302]:


listings = listings[['id', 'experiences_offered', 'review_scores_value', 'jurisdiction_names', 'instant_bookable',
                    'cancellation_policy', 'require_guest_profile_picture','require_guest_phone_verification',
                    'calculated_host_listings_count', 'reviews_per_month']]


# In[349]:


#After preparing datasets we are going to merge them
summary = pd.merge(listings, availability, left_on='id', right_on='listing_id', how='left')
summary = pd.merge(summary, price, left_on='id', right_on='listing_id', how='left')
summary.head()


# In[350]:


#Finding rows with all missing values
all_miss=summary.isnull().all().all()
print(all_miss)
print("There is 0 rows with all NAN values that mean no need for removing any")


# In[351]:


#Finding rows that has NAN values
a=np.sum(summary.isnull())
print(a)


# In[352]:


#checking data types
summary.dtypes


# In[353]:


summary.head()


# In[354]:


summary['experiences_offered'] = summary['experiences_offered'] != "none"
summary[['instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification']]  = summary[['instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification']] == "t"
summary[['jurisdiction_names', 'cancellation_policy']] = summary[['jurisdiction_names', 'cancellation_policy']].astype(str)
summary.info()


# In[355]:


summary.head()


# In[356]:


#Checking for columns with all same values, because column that has same values for all rows should be removed
nunique = summary.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
summary=summary.drop(cols_to_drop,axis=1)
summary.head()


# In[381]:


#count of cancelation policy which is categorical data
print(np.sum(summary['cancellation_policy'].isnull()))
#since there is no null data in it. We can creat dummy data 
summary2 = pd.concat([summary.drop('cancellation_policy', axis=1), pd.get_dummies(summary['cancellation_policy'])], axis=1) 
summary2


# In[382]:


summary.dtypes


# In[383]:


#list numerical variables 
num_vars=summary.select_dtypes(['float','int']).columns
print(num_vars)
num_vars = num_vars[num_vars != ('days_available')]
num_vars = num_vars[num_vars != ('percentage_available')]
num_vars = num_vars[num_vars != ('review_scores_value')]
num_vars


# In[384]:


# function for filling missing values with mean values
def fill_mean(df,cols):
    '''
    Input:
    df dataframe with numeric values
    cols columns which will be filled with mean value
    '''
    for col in cols:
        df[col]=df[col].fillna(df[col].mean(),axis=0)
    return df


# In[385]:


#apply filling method for num-vars
filled_summary=fill_mean(summary2,num_vars)
filled_summary.head()


# In[386]:


#Now id column is not needed to be predicted
filled_summary=filled_summary.drop('id',axis=1)


# In[387]:


print(np.sum(filled_summary.isnull()))


# In[388]:


#calculate the percentage of missing values of reviews scores which should be predicted 
per_misval=round(np.sum(filled_summary['review_scores_value'].isnull())/filled_summary.shape[0]*100,2)
print(per_misval,"% is the percentage of missing review scores")


# In[389]:


#drop rows with missing reviews scores
reviews=filled_summary.dropna(subset=['review_scores_value'],how='any')
#drop cancellation_policy
#reviews=reviews.drop("cancellation_policy",axis=1)
print(np.sum(reviews.isnull()))


# In[390]:


#Data modeling
y = reviews['review_scores_value']
X= reviews.drop('review_scores_value', axis = 1)

# Split data into a train and test dataset. The test dataset is 33% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a linear model object
lm_model = LinearRegression()

# Fit model on training data
lm_model.fit(X_train, y_train)


# Use model to predict based on test dataset
predictions = lm_model.predict(X_test)



# In[391]:


# Evaluate prediction power
reviews_r2_score = round(r2_score(y_test, predictions), 3)
print ("The R2 score of the reviews linear model is", reviews_r2_score)


# In[392]:


#create dataset to predict availabilty 
availability_dataset = fill_mean(filled_summary, ['review_scores_value'])
np.sum(availability_dataset.isnull())


# In[393]:


#lINEAR model for prediction of availablity
y2=availability_dataset['percentage_available']
X2=availability_dataset.drop('percentage_available',axis=1)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.33,random_state=84)
lm_model2=LinearRegression()
lm_model2.fit(X2_train,y2_train)
predictions2=lm_model2.predict(X2_test)
availability_r2_score = round(r2_score(y2_test, predictions2), 3)
print ("The R2 score of the availability linear model is", availability_r2_score)


# In[394]:


#Results summery
def summarize_features(coefficients, X_train):
    '''
    Input: coefficients: Are the linear model coefficients
             X_train: Is the trainig dataset
    Outout: Data frame holds model's features as well as coefficients
    Creates dataframe containing model features and their model coefficients (both raw and absolute) to evaluate
    the most influential features in the linear model.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['features'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df


# In[395]:


#Find summary of reviews model
review_summary=summarize_features(lm_model.coef_,X_train)
review_summary


# In[398]:


#For reviews modeling we can see that price coefs are very small, so price variation are not related to review score 
# Plot of review scores vs average price
sns.scatterplot(x = 'avg_price', y = 'review_scores_value', data = reviews)


# In[397]:


#Finding summary of availabilty model
availabilty_summary=summarize_features(lm_model2.coef_,X2_train)
availabilty_summary


# In[399]:


# Find out average number of reviews per month and then plot only for properties with > 2 reviews per month
reviews['reviews_per_month'].describe()
sns.scatterplot(x = 'avg_price', y = 'review_scores_value', data = reviews[reviews['reviews_per_month'] > 2])


# In[400]:


# flexible booking policies tend to have higher review scores and those with strict policies tend to have lower review scores.
#However, coefficients aren't high suggesting a moderate correlation. In conclusion, do properties with booking flexibility 
#have higher reviews, yes but the relationship is marginal


# In[401]:


# Bar plot of review scores by cancellation policy
sns.barplot(x = 'cancellation_policy', y = 'review_scores_value', data = summary, order = ['flexible', 'moderate', 'strict'])


# In[402]:


# Average review score by cancellation policy
summary.groupby('cancellation_policy')['review_scores_value'].mean()


# In[403]:


#The above chart and figures are in line with the conclusion above. Flexible policy propeties on average have a marginally higher review score than moderate and strict policy properties. 
#However, there is very little difference between them due to the vast number of 10 star reviews.



# In[405]:


# Coefficients of availability linear model 
availabilty_summary


# In[406]:


#The following suggests review score is a weak/moderate predictor of availaility and that properties with higher review scores are not more likely to be booked up.

sns.scatterplot(x = 'review_scores_value', y = 'percentage_available', data = reviews)


# In[ ]:


#The above chart shows there is very little relationship between review scores and percentage availabilty. 
#In fact the properties with lowest review scores are marginally more likely to be booked up.

