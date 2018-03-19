import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

# Read CSV files, and form combined DataFrame.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine =[train, test]

total = pd.concat(combine)
combine = [train, test, total]

#%% Understanding the data

# Extract Year, Month, Time
for df in combine:
    df['datetime']=pd.to_datetime(df['datetime'])
    df['Month'] = df.datetime.dt.month
    df['Hour'] = df.datetime.dt.hour
    df['Year'] = df.datetime.dt.year
    df['Day']= df.datetime.dt.day
    df['DayoW']=df.datetime.dt.dayofweek
    df['WeekoYear']=df.datetime.dt.weekofyear

for col in ['casual', 'registered', 'count']:
    total['%s_log' % col] = np.log(total[col] + 1)   
    train['%s_log' % col] = np.log(train[col] + 1)   
    
    
#%% Split TRAIN data into training and validation sections. 

def copydata(month,year):
    """This creates a copy of the previous data"""
    index = np.amax(np.where((train.Month==month) & (train.Year==year)))
    data = train.iloc[:(index+1)]
    return data

def split_train(data, cutoff):
    splittrain = data[data['Day'] <= cutoff]
    splittest = data[data['Day'] > cutoff]
    
    return splittrain, splittest
    
#%% Score validation data

# Submission is evaluated using the Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_p, y_a):
    diff = np.log(y_p + 1)-np.log(y_a + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

def set_up_data(featurenames, data):
    """Featurenames should be an array of strings corresponding to column names."""
    X = data[featurenames].as_matrix()
    Y_r = data['registered_log'].as_matrix()
    Y_c = data['casual_log'].as_matrix()
    
    return X, Y_r, Y_c
    
        
def splittrain_validation(model, featurenames):
    """Function iterates over PRIOR ONLY data, for validating model through a split
    training data. """
    months = range(1,13)
    years = [2011,2012]
    score = np.zeros([2,12])
    
    for i in years:
        for j in months:
            data = copydata(j,i)
            splittrain, splittest = split_train(data , 15)
    
            x_train, y_train_r, y_train_c = set_up_data(featurenames, splittrain)
            x_test, y_ar, y_ac = set_up_data(featurenames, splittest)
    
            M_r = model.fit(x_train, y_train_r)
            y_pr = np.exp(M_r.predict(x_test)) - 1
            
            M_c = model.fit(x_train, y_train_c)                              
            y_pc = np.exp(M_c.predict(x_test)) - 1
            
            y_pcomb = np.round(y_pr + y_pc)
            y_pcomb[y_pcomb < 0] = 0
            
            y_acomb = np.exp(y_ar) + np.exp(y_ac) - 2
    
            score[years.index(i),months.index(j)] = rmsle(y_pcomb, y_acomb)
    
    return score


# Using the entire data set as an initial, quick look at score.
def predict(input_cols, model):
    data = train
    
    splittrain, splittest = split_train(data,15)
    
    X_train, y_train_r, y_train_c = set_up_data(input_cols, splittrain)
    X_test, y_test_r, y_test_c = set_up_data(input_cols, splittest)
    
    model_r = model.fit(X_train, y_train_r)
    y_pred_r = np.exp(model_r.predict(X_test)) - 1
    imp_r = model_r.feature_importances_
    
    model_c = model.fit(X_train, y_train_c)
    y_pred_c = np.exp(model_c.predict(X_test)) - 1
    imp_c = model_c.feature_importances_
    
    y_pred_comb = np.round(y_pred_r + y_pred_c)
    y_pred_comb[y_pred_comb < 0] = 0
    
    y_test_comb = np.exp(y_test_r) + np.exp(y_test_c) - 2

    score = rmsle(y_pred_comb, y_test_comb)
    
    return imp_r, imp_c, score

#Predict on test set
def test_prediction(model, featurenames):
    months = range(1,13)
    years = [2011,2012]
    
    for i in years:
        for j in months:
            data = copydata(j,i)
            
            x_tr = data[featurenames].as_matrix()
            y_tr_r = data['casual_log'].as_matrix()
            y_tr_c = data['registered_log'].as_matrix()
            
            x_te = test[featurenames].as_matrix()
            
            Model_c = model.fit(x_tr, y_tr_c)
            y_pr_c = np.exp(Model_c.predict(x_te)) - 1
            
            Model_r = model.fit(x_tr, y_tr_r)
            y_pr_r = np.exp(Model_r.predict(x_te)) - 1
            
            y_pred = np.round(y_pr_r + y_pr_c)
            y_pred[y_pred < 0 ] = 0
            
    return y_pred
