#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import needed items
from lightfm_dataset_helper.lightfm_dataset_helper import DatasetHelper
import pandas as pd
from lightfm.data import Dataset
from scipy.stats import zscore
import math
import numpy as np

pd.set_option('display.float_format', lambda x: '%.3f' % x)


NUM_EPOCHS = 10
NUM_THREADS= 2


# In[2]:


# Read in the data, ensure cols are stripped for matching
ratings_csv = 'data/ratings.csv'
features_csv = 'data/features.csv'

ratings_df = pd.read_csv(ratings_csv)
features_df = pd.read_csv(features_csv)


features_df.rename(columns=lambda x: x.strip())
ratings_df.rename(columns=lambda x: x.strip())

# Incase there are some missing product_ids, drop them
ratings_df.dropna(axis=0, subset=(['product_id']),inplace=True)
features_df.dropna(axis=0, subset=(['product_id']),inplace=True)


# In[3]:


# We need to do some filtering to ensure that the data is useable
# let's examine the data

print(ratings_df['user_id'].value_counts().describe())
print(features_df['num_ratings'].describe())


# In[4]:


# Seems like the majoity for the users have at least 3 reviews and products at least  1.5m reviews, so let's set the thresolds there
min_reviews = 1500000
min_usr_reviews = 3

# we don't want features if they have an avg rating of 0
before_filter = features_df.shape[0]
features_df= features_df[features_df['avg_ratings'] !=0]
after_filter = features_df.shape[0]

print(f"Average Rating Filter - Before :{before_filter}  After :{after_filter} ")

#filtering based on the data analysis
before_filter = features_df.shape[0]
features_df = features_df[features_df['num_ratings']>=min_reviews]
after_filter = features_df.shape[0]

print(f"Features - Before :{before_filter}  Before :{after_filter} ")

before_filter = ratings_df.shape[0]
filtered_usr=(ratings_df['user_id'].value_counts()>=min_usr_reviews)
filtered_usr= filtered_usr[filtered_usr].index.tolist()
ratings_df = ratings_df[(ratings_df['user_id'].isin(filtered_usr))]
after_filter = ratings_df.shape[0]
print(f"Users - Before :{before_filter}  After :{after_filter} ")


# In[5]:


# normalization needs to occur in a few places: product avg ratings,num_ratings and reducing the user bias

# Z-score normalization for product avg ratings by product id
features_df['prod_z']=features_df.groupby('product_id')['avg_ratings'].transform(lambda x : zscore(x))

# log normalize num_ratings and divide by max
# since lightfm normalized rows, we wish to reduce massive numbers such as num_ratings
features_df['log_norm_ratings']=np.log(features_df['num_ratings'])/ features_df.groupby('product_id')['num_ratings'].transform(np.max)

# log1p seems to be a standard price normalizer based on research, so let's normalize that
features_df['logPrice'] = np.log1p(features_df['price'])

#we normalized the 3 columns, so they can be dropped
features_df = features_df.drop(columns=['avg_ratings','num_ratings','price'])



# In[6]:


# creating the interaction matrix, we need to take into account user bias in the ratings per user.
# there are a few approaches, such as subtracting the average user rating from the particular rating, gaussian distribution, etc.
# subtracting the averager user rating from the rating is quick and dirty in this case.

ratings_df['user_avg'] = ratings_df.groupby(['user_id','product_id'])['rating'].transform('mean')

merged_ratings_items = ratings_df.join(features_df[['product_id','log_norm_ratings','prod_z']].set_index('product_id'), on='product_id',how='inner')

merged_ratings_items['post_bias'] = merged_ratings_items["rating"] - merged_ratings_items["user_avg"]


# In[7]:


# we can construct the user dataframe
user_df = ratings_df[['user_id','rating','user_avg']]
user_df['bias'] = ratings_df['rating'] - ratings_df['user_avg']
user_df['avg_bias'] = user_df.groupby('user_id')['bias'].transform('mean')



# there is no need to keep duplicates
user_df = user_df[['user_id','avg_bias']].drop_duplicates()


# In[8]:


# let's construct this lightfm with the interactions, users, and products
item_features_cols = list(features_df)

items_col = 'product_id'
user_col = 'user_id'
ratings_col = 'post_bias'

# intereraction_df
interaction_df = merged_ratings_items[['user_id','product_id',ratings_col]]


# In[9]:


# using the dataset helper from pypi, which automates the lightfm creation, we make the model.
helper_instance = DatasetHelper(
    users_dataframe = user_df,
    items_dataframe = features_df,
    interactions_dataframe = interaction_df,
    item_id_column=items_col,
    user_features_columns=['avg_bias'],
    items_feature_columns=item_features_cols,
    user_id_column =user_col,
    interaction_column=ratings_col,
    clean_unknown_interactions=True
)



# In[10]:


# we make the model with the wrap loss, and adagrad learning schedule as we have a low number of epochs
from lightfm import LightFM
helper_instance.routine()
model = LightFM(no_components= 80, loss="warp",item_alpha= 1e-7,user_alpha=1e-7,learning_rate = 0.02,learning_schedule='adagrad')


# In[11]:


from lightfm.cross_validation import random_train_test_split
import numpy as np
# we need to split into test and train, so we set the seed and use 20% for teesting

train,test = random_train_test_split(helper_instance.interactions, test_percentage=0.2, random_state=np.random.RandomState(3))


# In[12]:


# we fit the model
cf=model.fit(
    interactions = train,
    item_features =helper_instance.item_features_list,
    verbose=True,
    epochs=NUM_EPOCHS,
    num_threads=NUM_THREADS)


# In[13]:


# Import the evaluation routines
from lightfm.evaluation import auc_score


# Compute and print the AUC score
train_auc = auc_score(cf, train, item_features=helper_instance.item_features_list,num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)
test_auc = auc_score(cf, test, item_features=helper_instance.item_features_list,num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)


# In[14]:


#We make the hybrid model


hybrid_model = model.fit(train,
                item_features=helper_instance.item_features_list,
                epochs=NUM_EPOCHS,
                num_threads=NUM_THREADS,verbose=True)


# In[15]:


#AUC of the hybrid model
train_auc_h = auc_score(hybrid_model,
                      train,
                      item_features=helper_instance.item_features_list,
                      num_threads=NUM_THREADS).mean()

print('Hybrid training set AUC: %s' % train_auc_h)


# In[16]:


test_auc_h = auc_score(hybrid_model,
                      test_interactions=test,
                      item_features=helper_instance.item_features_list,
                      num_threads=NUM_THREADS).mean()
print('Hybrid testing set AUC: %s' % test_auc_h)


# In[ ]:





# In[17]:


#visual score representation
score_dict = {
    "CF":{
        "Train":train_auc,
        "Test":test_auc
    },
    "Hybrid":{
        "Train":train_auc_h,
        "Test":test_auc_h
    }

}

baseline = {
    "CF":{
        "Train":0.887519,
        "Test":0.34728
    },
    "Hybrid":{
        "Train":0.86049,
        "Test":0.703039
    }
}

score_df = pd.DataFrame.from_dict(score_dict).T
base_df = pd.DataFrame.from_dict(baseline).T

print ("Baseline")
print(base_df)
print ('---------')
print ("Improved")
print(score_df)


# In[18]:


# Future improvements

# 1. scikit hyperparameter optimization [sklearn model_selection.RandomizedSearchCV]
# 2. Context of the fields is important for the weighting, such as male/female specific for clothing. Having the fields as just numbers really doesn't help all that much in creating the weights
# 3. Increasing Epochs. my macbook was not happy with me when I tried to do 80 epochs, but the more epochs, the better the fit. 10 seemed to be used quite often in my research, so I picked that.
#4 There are many rating normalizations from my reserach, and it would have been great if I could have tested them all( main one would have been the beta distribution).
#5. Data cleaning could have been much more scientific instead of quantiles with a buffer, but the data seems to be much more improved with these settings

