# Importing important sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression

# Packaging classification steps
from sklearn.pipeline import Pipeline

# Training and Testing libraries
from sklearn.model_selection import train_test_split

# import over sampling library
from imblearn.over_sampling import RandomOverSampler

from sklearn.tree import DecisionTreeClassifier

# saving models
import joblib
import pickle
# text transformers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 3.1 Read file as panda dataframe
telecom_data = pd.read_csv(r'C:\Users\Nkanabo\Desktop\Python\clean_telecom_data.csv')

# X.shape should be (N, M) where M >= 1
X = telecom_data[['clean_comment']]
# y.shape should be (N, 1)
y = telecom_data['sentiment']

logModel = LogisticRegression()

# Splitting traing and testing data into 80% and 20% ratio respectively
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, shuffle=True)

# over sampling
ros = RandomOverSampler(random_state=42)
# fit predictor and target variable
x_resampled, y_resampled = ros.fit_resample(X_train, y_train)

x_resampled = x_resampled.squeeze()
y_resampled = y_resampled.squeeze()
y_train = y_train.squeeze()
y_test = y_test.squeeze()
X_train = X_train.squeeze()
X_test = X_test.squeeze()
type(x_resampled)

# Initialize a vectorization and modelling pipeline
lr_pipeline = Pipeline([
    # ('std_slc', std_slc),
    # ('pca', pca),
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=5000)),
])


# Initialize a vectorization and modelling pipeline
# Initialize a vectorization and modelling pipeline
dt_pipeline = Pipeline([
     ('vect', CountVectorizer(ngram_range=(1,2))),
     ('tfidf', TfidfTransformer()),
     ('clf', DecisionTreeClassifier()),
 ])


# Algorithm training X_train, X_test, y_train, y_test
dt_model = dt_pipeline.fit(x_resampled.values.astype('U'), y_resampled.values.astype('U'))
dt_model


lr_model = lr_pipeline.fit(X_train.values.astype('U'), y_train.values.astype('U'))


pickle.dump(lr_model, open('fine_tuned_logistic_regression.pkl', 'wb'))

pickle.dump(dt_model, open('fine_tuned_Decision Tree.pkl', 'wb'))

filename = r'C:\Users\Nkanabo\Desktop\finalized_model.sav'
joblib.dump(lr_model, filename)
print('finshed')