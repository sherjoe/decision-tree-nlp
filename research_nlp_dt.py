#!/usr/bin/env python
# coding: utf-8


# In[96]:


import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# In[100]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import plot_roc_curve, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ## Part 1: Load the Data Set
# 


# In[4]:


filename = os.path.join(os.getcwd(), "data", "bookReviewsData.csv")
df = pd.read_csv(filename, header=0)

df.head()


# ## Part 2: Exploratory Data Analysis
# # Finding Out if a Gradient Boosted Decision Tree is the Best Decision Tree Model for a Single Label and Feature NLP Problem
# ### I will be utilizing a TF-IDF Vectorizer in order to transform the text into individual vectors.
# I have inspected the data to get a grasp of what qualifies as a positive or negative review. I have chosen the reviews column to be my feature and the positive reviews to be the label I will attempt to predict.
# 

# In[5]:


y = df['Positive Review'] #my label
X = df['Review'] #my feature

X.shape


# ## Part 3: Modeling


# In[6]:


X.head()


# In[7]:


print('A positive review: \n\n', X[1])
print('A negative review: \n\n', X[3])


# I have split the test data into 0.20 for efficiency and to avoid bottleneck executions, in spite of a potential class imbalance. I also made my TF-IDF vectorizer using an ngram_range of (1,2) and min_df of 5.

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1234)


# In[9]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5) #create vectorizer
tfidf_vectorizer.fit(X_train) #fit the vectorizer
X_train_tfidf = tfidf_vectorizer.transform(X_train) #transform training and test data
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[10]:


print(X_test_tfidf)


# On my Gradient Boosted Classifier, I consistently get an AUC score around 90-91, if nothing else in the notebook got significantly changed.

# In[57]:


gbdt_model_c = GradientBoostingClassifier(max_depth= 3, n_estimators = 830) #create model

gbdt_model_c.fit(X_train_tfidf, y_train) # it to datasets

gbdt_pred_c = gbdt_model_c.predict_proba(X_test_tfidf)[:,1] #make predictions

auc = roc_auc_score(y_test, gbdt_pred_c) #calculate AUC score
print('AUC on the test data: {:.4f}'.format(auc))

len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {0}'.format(len_feature_space))


# My grid search in order to find the optimal max_depth and n_estimators hyperparameters of 3 and 830 took 18.3 minutes to complete. However, I think that this was necessary in order to improve the performance of the model by focusing on prior mistakes, bettering the model in the process.

# In[11]:


param_grid_c = {'max_depth': [1,2,3],
                'n_estimators' : [800,830,850]}  #hyperparameter values
gbdt_grid_c = GridSearchCV(gbdt_model_c, param_grid_c, cv = 5, scoring = 'roc_auc', verbose=2)

gbdt_grid_c_search = gbdt_grid_c.fit(X_train_tfidf, y_train) #fit grid search


# In[12]:


gbdt_grid_c_search.best_params_ #best parameter values 


# In[18]:


roc_auc_GBDT_c = gbdt_grid_c_search.best_score_
print("[GBDT] AUC : {:.2f}".format(roc_auc_GBDT_c) ) #close enough best AUC value


# In[19]:


gbdt_grid_c_search.best_estimator_ #model best scores


# I decided to do a random search to see if it would yield the same results as the grid search with less time, but it was the same amount of time.

# In[20]:


param_grid_c = {'max_depth': [1,2,3],
                'n_estimators' : [800,830,850]} #random search hyperparameter values
gbdt_rand_c = RandomizedSearchCV(gbdt_model_c, param_grid_c, cv = 5, scoring = 'roc_auc', verbose=2)

gbdt_rand_c_search = gbdt_rand_c.fit(X_train_tfidf, y_train) #fit random search


# In[21]:


gbdt_rand_c_search.best_params_ #best hyperparameters


# In[40]:


print('Begin ML pipeline...') #pipeline to graph AUC
s = [
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=5)),
        ("gbdt_model", GradientBoostingClassifier(max_depth= 3, n_estimators = 830))
    ]

model_pipeline = Pipeline(steps=s)

model_pipeline.fit(X_train, y_train)

probability_predictions = model_pipeline.predict_proba(X_test)[:,1]

print('End pipeline')


# In[41]:


plot_roc_curve(model_pipeline, X_test, y_test)


# In[24]:


model_pipeline.get_params().keys() #key for vectorizer parameter names


# I used my model pipeline to find out what eventually would be my optimal TD-IDF parameters.

# In[25]:


list1 = [5,10,21]
list2 = [(1,1), (1,2)]   #grid search for best hyperparameters
param_grid = {'vectorizer__min_df': list1, 
              'vectorizer__ngram_range' : list2 }

grid = GridSearchCV(model_pipeline, param_grid, cv = 5, scoring = 'roc_auc', verbose=2)
grid_search = grid.fit(X_train, y_train) #fit grid search


# In[27]:


grid_search.best_params_ #best hyperparameters


# In[63]:


plot_roc_curve(grid_search, X_test, y_test) #plot best results


# In[17]:


print("Predictions for the first 10 examples:")
print("Probability\t\t\tClass")
for i in range(0,10):
    if gbdt_pred_c[i] >= 0.5: #gbdt predictions are true if their number is or more than half
        class_pred = "Good Review"
    else:
        class_pred = "Bad Review"
    print(str(gbdt_pred_c[i]) + "\t\t\t" + str(class_pred))


# It looks like the model performs well enough as both predictions within the 91 AUC score are correct.

# In[18]:


print('Review #1:\n') #prints out first review
print(X_test.to_numpy()[1])

goodReview = True if gbdt_pred_c[1] >= 0.5 else False
    
print('\nPrediction: Is this a good review? {}\n'.format(goodReview))

print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[1]))

print('Review #2:\n') #prints out third review with predicted and actual scores
print(X_test.to_numpy()[3])

goodReview = True if gbdt_pred_c[3] >= 0.5 else False

print('\nPrediction: Is this a good review? {}\n'.format(goodReview)) 

print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[3]))


# I then decided to use a regressor to look more into if the model is overfitting or not. While the regressor can not predict continuous values for this kind of problem, it can give an insight to how the model seperates classes with its AUC.

# In[18]:


gbdt_model_r = GradientBoostingRegressor(max_depth= 3, n_estimators = 800) #create model

gbdt_model_r.fit(X_train_tfidf, y_train) #fit the model

gbdt_pred = gbdt_model_r.predict(X_test_tfidf) # make predictions

auc = roc_auc_score(y_test, gbdt_pred)
print('AUC on the test data: {:.4f}'.format(auc)) # print """AUC"""

len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {0}'.format(len_feature_space))


# In[13]:


param_grid_rsme = {'max_depth': [1,2,3],
                'n_estimators' : [800,830,850]} #best rsme hyperparameters 
gbdt_grid_rsme = GridSearchCV(gbdt_model_r, param_grid_rsme, cv = 5, scoring = 'neg_root_mean_squared_error', verbose=2)

gbdt_grid_rsme_search = gbdt_grid_rsme.fit(X_train_tfidf, y_train)


# In[14]:


rmse_GBDT = -1 * gbdt_grid_rsme_search.best_score_ #multiply by -1 in order to obtain best RMSE
print("[GBDT] RMSE  : {:.2f}".format(rmse_GBDT) )


# In[16]:


gbdt_grid_rsme_search.best_estimator_


# In[17]:


gbdt_grid_rsme_search.best_params_


# In[19]:


gbdt_rmse = mean_squared_error(y_test, gbdt_pred, squared=False)
gbdt_r2 = r2_score(y_test, gbdt_pred)  #prints RSME and R2 scores

print('[GBDT] Root Mean Squared Error: {0}'.format(gbdt_rmse))
print('[GBDT] R2: {0}'.format(gbdt_r2))                 


# While this may not be a completely accurate evaluation metric, it does show that the model is generalizing well, and has a low bias and variance through its high R2 score and low RSME score.

# In[20]:


rg= np.arange(1) #plots r2 and RSME scores
width = 0.35
plt.bar(rg, gbdt_rmse, width, label="RMSE")
plt.bar(rg+width, gbdt_r2, width, label='R2')
plt.xlabel("GBDT Model")
plt.ylabel("RMSE/R2")
plt.ylim([0,1])

plt.title('Model Performance')
plt.legend(loc='upper left', ncol=2)
plt.show()


# I then used a Random Forest Classifier in order to see how well it performs against my GBDT model. The grid search took a lot less long in comparsion, only five minutes, and yielded a similar AUC score to my GBDT model.

# In[29]:


rf_model = RandomForestClassifier(max_depth= 35, n_estimators = 900) #create model

rf_model.fit(X_train_tfidf, y_train) #fit model

rf_pred = rf_model.predict_proba(X_test_tfidf)[:,1] #make predictions

auc = roc_auc_score(y_test, rf_pred) #calculate AUC
print('AUC on the test data: {:.4f}'.format(auc))

len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {0}'.format(len_feature_space))


# In[41]:


param_grid_rf = {'max_depth': [15,25,35],
                'n_estimators' : [800,900,1000]} #find best hyperparameters for AUC
gbdt_grid_rf = GridSearchCV(rf_model, param_grid_rf, cv = 5, scoring = 'roc_auc', verbose=2)

gbdt_grid_rf_search = gbdt_grid_rf.fit(X_train_tfidf, y_train) #fits grid search


# In[42]:


gbdt_grid_rf_search.best_params_


# In[32]:


print('Begin ML pipeline...') #pipeline to plot AUC
s = [
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=5)),
        ("rf_model", RandomForestClassifier(max_depth= 35, n_estimators = 900))
    ]

model_pipeline = Pipeline(steps=s)

model_pipeline.fit(X_train, y_train)

probability_predictions = model_pipeline.predict_proba(X_test)[:,1]

print('End pipeline')


# In[33]:


plot_roc_curve(model_pipeline, X_test, y_test)


# In[34]:


print("Predictions for the first 10 examples:") #prints probability scores
print("Probability\t\t\tClass")
for i in range(0,10):
    if rf_pred[i] >= 0.5 :
        class_pred = "Good Review"
    else:
        class_pred = "Bad Review"
    print(str(rf_pred[i]) + "\t\t\t" + str(class_pred))


# However, for the second review, its True prediction was incorrect. This may be because instead of the model gradually improving on itself, the Random Forest averages the predictions at the end, to where it only samples a portion of the training dataset at the beginning, in order to construct those decision trees. Unfortunately I could not get a Regressor to function without throwing a bottleneck execution, so I can not further analyze it from here.

# In[35]:


print('Review #1:\n') #prints first and second review and their predicted and accurate scores
print(X_test.to_numpy()[1])

goodReview = True if rf_pred[1] >=0.5 else False
    
print('\nPrediction: Is this a good review? {}\n'.format(goodReview))

print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[1]))

print('Review #2:\n')
print(X_test.to_numpy()[3])

goodReview = True if rf_pred[3] >=0.5 else False

print('\nPrediction: Is this a good review? {}\n'.format(goodReview)) 

print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[3]))


# I then used a regular Decision Tree to compare it to my GBDT and RF model. Peforming a grid search was very quick, and I was able to test a lot of different parameter values in one execution, but the end results were disappointing. No matter what the parameters were, it would yield the same AUC score in the 60s.

# In[85]:


dt_model_c = DecisionTreeClassifier(max_depth= 10 ,min_samples_leaf = 100) #create model

dt_model_c.fit(X_train_tfidf, y_train) #fit the model

dt_pred_c = dt_model_c.predict_proba(X_test_tfidf)[:,1] #make predictions 

auc = roc_auc_score(y_test, dt_pred_c)
print('AUC on the test data: {:.4f}'.format(auc)) #calculate AUC

len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {0}'.format(len_feature_space))


# In[78]:


param_grid_dtc = {'max_depth': [1,10,20,30,40,50,60,70,80,90,100],
                'min_samples_leaf' : [100,200,300,400,500,600,700,800,900]}
dt_grid_c = GridSearchCV(dt_model_c, param_grid_dtc, cv = 5, scoring = 'roc_auc', verbose=2)

dt_grid_c_search = dt_grid_c.fit(X_train_tfidf, y_train) #fit gridsearch


# In[79]:


dt_grid_c_search.best_params_ #best hyperparameters 


# In[67]:


print('Begin ML pipeline...') #pipeline to plot AUC
s = [
        ("vectorizer", TfidfVectorizer(ngram_range=(1,1), min_df=5)),
        ("dt_model", DecisionTreeClassifier(max_depth= 30, min_samples_leaf = 100))
    ]

model_pipeline = Pipeline(steps=s)

model_pipeline.fit(X_train, y_train)

probability_predictions = model_pipeline.predict_proba(X_test)[:,1]

print('End pipeline')


# In[68]:


plot_roc_curve(model_pipeline, X_test, y_test)


# In[88]:


dt_model_r = DecisionTreeRegressor(max_depth= 10 ,min_samples_leaf = 100) #create model

dt_model_r.fit(X_train_tfidf, y_train) #fit the model

dt_pred_r = dt_model_r.predict(X_test_tfidf) #make predictions

auc = roc_auc_score(y_test, dt_pred_r) #calculate """AUC"""
print('AUC on the test data: {:.4f}'.format(auc))

len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {0}'.format(len_feature_space))


# In[91]:


param_grid_dtr = {'max_depth': [1,10,20,30,40,50,60,70,80,90,100],
                'min_samples_leaf' : [100,200,300,400,500,600,700,800,900]}
dt_grid_r = GridSearchCV(dt_model_r, param_grid_dtr, cv = 5, scoring = 'neg_root_mean_squared_error', verbose=2)

dt_grid_r_search = dt_grid_r.fit(X_train_tfidf, y_train) #fit gridsearch


# In[92]:


dt_grid_r_search.best_params_ #best hyperparameters


# In[94]:


dt_rmse = mean_squared_error(y_test, dt_pred_r, squared=False) #prints R2 and RSME scores
dt_r2 = r2_score(y_test, dt_pred_r) 

print('[GBDT] Root Mean Squared Error: {0}'.format(dt_rmse))
print('[GBDT] R2: {0}'.format(dt_r2))                 


# The Decision Tree Regressor only confirms my disapponting results by showing how poorly generalized the model is with a siginficantly higher RSME score to the R2 score. Ensemble methods use multiple trees and combines the outputs of them, which this model can not do, which may explain its poor performance.

# In[99]:


rg= np.arange(1) #plots R2 and RSME scores
width = 0.35
plt.bar(rg, dt_rmse, width, label="RMSE")
plt.bar(rg+width, dt_r2, width, label='R2')
plt.xlabel("DT Model")
plt.ylabel("RMSE/R2")
plt.ylim([0,1])

plt.title('Model Performance')
plt.legend(loc='upper left', ncol=2)
plt.show()


# I then implemented a stacking model in order to see if it would perform better than my GBDT and RF models alone. It only got around the same 90 AUC score.

# In[102]:


estimators = [("GBDT", GradientBoostingClassifier(max_depth= 3, n_estimators = 830)),
              ("RF", RandomForestClassifier(max_depth= 35, n_estimators = 900))
             ] #model list


# In[104]:


print('Implement Stacking...') #create the model

stacking_c = StackingClassifier(estimators = estimators, cv = 5, passthrough=False)
stacking_c.fit(X_train_tfidf, y_train) #fit the model

print('End')

stacking_pred_c = stacking_c.predict_proba(X_test_tfidf)[:,1] #make predictions

auc = roc_auc_score(y_test, stacking_pred_c) #calculate AUC score
print('AUC on the test data: {:.4f}'.format(auc))

len_feature_space = len(tfidf_vectorizer.vocabulary_)

print('The size of the feature space: {0}'.format(len_feature_space))


# In[105]:


print('Begin ML pipeline...') #create pipeline to plot AUC
s = [
        ("vectorizer", TfidfVectorizer(ngram_range=(1,2), min_df=5)),
        ("stack_model", StackingClassifier(estimators = estimators, cv = 5, passthrough=False))
    ]

model_pipeline = Pipeline(steps=s)

model_pipeline.fit(X_train, y_train)

probability_predictions = model_pipeline.predict_proba(X_test)[:,1]

print('End pipeline')


# In[106]:


plot_roc_curve(model_pipeline, X_test, y_test)


# It did predict the two reviews correctly, so I would say this model also does generalize and perform well. Though, I think that the GBDT model prediction scores were more accurate to what they were supposed to be than the Stacking model scores.

# In[120]:


print("Predictions for the first 10 examples:") #prints probability scores
print("Probability\t\t\tClass")
for i in range(0,10):
    if stacking_pred_c[i] >= 0.5 :
        class_pred = "Good Review"
    else:
        class_pred = "Bad Review"
    print(str(stacking_pred_c[i]) + "\t\t\t" + str(class_pred))


# In[107]:


print('Review #1:\n') #prints the first two reviews with their predicted and actual values.
print(X_test.to_numpy()[1])

goodReview = True if stacking_pred_c[1] >=0.5 else False
    
print('\nPrediction: Is this a good review? {}\n'.format(goodReview))

print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[1]))

print('Review #2:\n')
print(X_test.to_numpy()[3])

goodReview = True if stacking_pred_c[3] >=0.5 else False

print('\nPrediction: Is this a good review? {}\n'.format(goodReview)) 

print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[3]))


# ## Conclusion :
# 
# ### The Gradient Boosted Decision Tree performed and generalized the best.

# In[ ]:




