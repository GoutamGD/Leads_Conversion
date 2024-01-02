print("Adding all the data processing code related to Leads_Conversion")
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing Pandas and NumPy
import pandas as pd, numpy as np
# Importing all datasets
leads_data = pd.read_csv(r"C:\Users\deygo\Downloads\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv")
leads_data.head()
leads_dictionary = pd.read_csv(r"C:\Users\deygo\Downloads\Leads-Data-Dictionary.csv")
leads_dictionary.head(100)
# Let's check the dimensions of the dataframe
leads_data.shape
# let's look at the statistical aspects of the dataframe
leads_data.describe()
# Let's see the type of each column
leads_data.info()
#removing the null values
leads_data.dropna(inplace=True)
leads_data
leads_data.info()
#Data Preparation
#Converting some binary variables (Yes/No) to 0/1
leads_data=leads_data.replace({'Do Not Email':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Do Not Call':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Search':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Magazine':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Newspaper Article':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'X Education Forums':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Newspaper':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Digital Advertisement':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Through Recommendations':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Receive More Updates About Our Courses':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Update me on Supply Chain Content':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'Get updates on DM Content':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'I agree to pay the amount through cheque':{'Yes':1,'No':0}})
leads_data=leads_data.replace({'A free copy of Mastering The Interview':{'Yes':1,'No':0}})
leads_data
leads_data=leads_data.drop(['Specialization','Country','Lead Origin', 'Lead Source', 'Last Activity', 'How did you hear about X Education','What matters most to you in choosing a course','Tags','City','Asymmetrique Activity Index','Asymmetrique Profile Index','Last Notable Activity','Lead Source', 'Last Activity', 'How did you hear about X Education','What matters most to you in choosing a course','Tags','City','Asymmetrique Activity Index','Asymmetrique Profile Index','Last Notable Activity'],1)
leads_data
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(leads_data[['What is your current occupation','Lead Quality','Lead Profile']], drop_first=True)
#adding the results to the master data frame
leads_data=pd.concat([leads_data,dummy1],axis=1)
leads_data.head()
leads_data.info()
leads_data.dtypes
#leads_data['Prospect ID'] = leads_data['Prospect ID'].apply(pd.to_numeric, errors='coerce')
leads_data
# Checking for outliers in the continuous variables
num_leads_data=leads_data[['Converted','TotalVisits','Total Time Spent on Website','Page Views Per Visit','Asymmetrique Activity Score','Asymmetrique Profile Score']]
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_leads_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])
num_leads_data
#### Checking for Missing Values and Inputing Them
leads_data
# Adding up the missing values (column-wise)
leads_data.isnull().sum()
# Checking the percentage of missing values
round(100*(leads_data.isnull().sum()/len(leads_data.index)), 2)
# Removing NaN Prospect ID rows
#leads_data = leads_data[~np.isnan(leads_data['Prospect ID'])]
leads_data
# Checking percentage of missing values after removing the missing values
round(100*(leads_data.isnull().sum()/len(leads_data.index)), 2)
leads_data.columns
#Dropping the variables which are high data imbalance & donot add value ot the model building 
leads_data.drop(['Do Not Email','Do Not Call','Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Update me on Supply Chain Content','I agree to pay the amount through cheque','Get updates on DM Content'],axis=1,inplace=True)
leads_data.columns
null_data=round(leads_data.isnull().sum()*100/len(leads_data.index),2)
null_data
leads_data
#duplicate check
leads_data.loc[leads_data.duplicated()]
leads_data.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','Lead Quality_Not Sure','Lead Quality_Might be','Lead Profile_Select','Lead Profile_Potential Lead','Lead Quality_Worst'], 1)
### Step 4: Test-Train Split
from sklearn.model_selection import train_test_split
# Putting feature variable to X
X = leads_data.drop(['Converted','Prospect ID'], axis=1)

X.head()
# Putting response variable to y
y = leads_data['Converted']

y.head()
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
### Step 5: Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
### Checking the Conversion Rate
Converted = (sum(leads_data['Converted'])/len(leads_data['Converted'].index))*100
Converted 
#We have almost 54% Conversion rate
#further looking into the variables which are not required and can be dropped 
X_train.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','Lead Quality_Not Sure','Lead Quality_Might be','Lead Profile_Select','Lead Profile_Potential Lead','Lead Quality_Worst'], 1)
X_train.drop(['What is your current occupation','What is your current occupation_Unemployed','What is your current occupation_Working Professional','Lead Quality_Not Sure','Lead Quality_Might be','Lead Profile_Select','Lead Profile_Potential Lead','Lead Quality_Worst'], 1)
X_train.drop(['What is your current occupation','What is your current occupation_Housewife','Lead Quality_Not Sure','Lead Quality_Might be','Lead Profile_Select','Lead Profile_Potential Lead','Lead Quality_Worst'], 1)
X_train.drop(['What is your current occupation','What is your current occupation_Other','Lead Quality_Not Sure','Lead Quality_Might be','Lead Profile_Select','Lead Profile_Potential Lead','Lead Quality_Worst'], 1)
X_train.drop(['What is your current occupation','What is your current occupation_Housewife','Lead Quality_Not Sure','Lead Quality_Might be','Lead Profile_Select','Lead Profile_Potential Lead','Lead Quality_Worst'], 1)
X_train.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','What is your current occupation_Other','What is your current occupation_Housewife'], 1)
X_train.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','What is your current occupation_Other','What is your current occupation_Housewife','Lead Quality','Lead Quality_Not Sure','Lead Quality_Might be','Lead Quality_Low in Relevance','Lead Quality_Worst','Lead Profile','Lead Profile_Lateral Student','Lead Profile_Other Leads','Lead Profile_Potential Lead','Lead Profile_Select','Lead Profile_Student of SomeSchool'], 1)
X_train
X_train.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','What is your current occupation_Other','What is your current occupation_Housewife'],1)
X_train.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','What is your current occupation_Other','What is your current occupation_Housewife','Lead Quality','Lead Quality_Not Sure','Lead Quality_Might be','Lead Quality_Low in Relevance','Lead Quality_Worst'],1)
X_train.drop(['What is your current occupation','What is your current occupation_Student','What is your current occupation_Unemployed','What is your current occupation_Working Professional','What is your current occupation_Other','What is your current occupation_Housewife','Lead Quality','Lead Quality_Not Sure','Lead Quality_Might be','Lead Quality_Low in Relevance','Lead Quality_Worst','Lead Profile','Lead Profile_Lateral Student','Lead Profile_Other Leads','Lead Profile_Potential Lead','Lead Profile_Select','Lead Profile_Student of SomeSchool'],axis=1,inplace=True)
X_train
X_train.isnull().sum()
X_train.drop(['Receive More Updates About Our Courses'],axis=1,inplace=True)
### Step 6: Looking at Correlations
# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X_train.corr(),annot = True)
plt.show()
#### Checking the Correlation Matrix
plt.figure(figsize = (20,10))
sns.heatmap(X_train.corr(),annot = True)
plt.show()
X_train.dtypes
### Step 7: Model Building
#Let's start by splitting our data into a training set and a test set.
#### Running Your First Training Model
import statsmodels.api as sm
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
### Step 8: Feature Selection Using RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg,n_features_to_select=15)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
##### Assessing the model with StatsModels
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
##### Creating a dataframe with the actual Converted flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Lead Number'] = y_train.index
y_train_pred_final.head()
##### Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
# Predicted     not_Converted   Converted
# Actual
# not_Converted       480      160
# Converted           150      570  
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 
#from statsmodels.stats.outliers_influence import variance_inflation_factor
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#There are a few variables with high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex. The variable 'PhoneService' has the highest VIF. So let's start by dropping that.
col = col.drop('Lead Number', 1)
col
# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred[:10]y_train_pred_final['Converted_Prob'] = y_train_pred
y_train_pred_final.head()
# Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
#So overall the accuracy hasn't dropped much.
##### Let's check the VIFs again
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Let's drop TotalCharges since it has a high VIF
col = col.drop('Asymmetrique Profile Score')
col
# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Converted_Prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
##### Let's now check the VIFs again
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Let's take a look at the confusion matrix again 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion
# Actual/Predicted     not_converted    converted
        # not_Converted        464      176
        # converted            159       561 
        # Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
## Metrics beyond simply accuracy
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting converted when customer does not have converted
print(FP/ float(TN+FP))
# positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
### Step 9: Plotting the ROC Curve
#An ROC curve demonstrates several things:

#- It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
#- The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
#- The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
### Step 10: Finding Optimal Cutoff Point
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
#### From the curve above, 0.5 is the optimum point to take it as a cutoff probability.
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
## Precision and Recall
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion
##### Precision
TP / (TP + FP)
confusion[1,1]/(confusion[0,1]+confusion[1,1])
##### Recall
TP / (TP + FN)
confusion[1,1]/(confusion[1,0]+confusion[1,1])
from sklearn.metrics import precision_score, recall_score
?precision_score
precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)
### Precision and recall tradeoff
from sklearn.metrics import precision_recall_curve
y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
### Step 11: Making predictions on the test set
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_test = X_test[col]
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting Lead Number to index
y_test_df['Lead Number'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})
y_pred_final.head()
y_pred_final.set_index(['Lead Number','Converted','Converted_Prob'])
y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.50 else 0)
y_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
