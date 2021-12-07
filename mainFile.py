from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics



def feature_scaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X


def feature_encoder(X):
    lbl = LabelEncoder()
    lbl.fit(list(X.values))
    X = lbl.transform(list(X.values))
    return X




#Read data
data = pd.read_csv("VideoLikesDataset.csv")

#video_id, trending_date, title, channel_title , category_id, publish_time, tags, views, comment_count , comments_disabled , ratings_disabled, video_error_or_removed, Likes (Y)
data["likes"]=data["likes"].replace([np.inf,-np.inf],np.nan)
data["comment_count"]=data["comment_count"].replace([np.inf,-np.inf],np.nan)
data["views"]=data["views"].replace([np.inf,-np.inf],np.nan)
#Drop rows of blank values
data.dropna(how='any', inplace=True)

#Drop removed videos
data = data[data['video_error_or_removed'] == False]

#Drop #NAMES?
data = data[data['video_id'] != "#NAME?"]

#Handle format
data['trending_date'] = pd.to_datetime(data['trending_date'], format='%y.%d.%m')
data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

#split it into columns
data.insert(5, 'publish_date', data['publish_time'].dt.date)
data['publish_time'] = data['publish_time'].dt.time
data['publish_date'] = pd.to_datetime(data['publish_date'])

#Encode features
data["video_id"] = feature_encoder(data["video_id"])
data["title"] = feature_encoder(data["title"])
data["channel_title"] = feature_encoder(data["channel_title"])
data["comments_disabled"] = feature_encoder(data["comments_disabled"])
data["ratings_disabled"] = feature_encoder(data["ratings_disabled"])
data["video_error_or_removed"] = feature_encoder(data["video_error_or_removed"])
data["tags"] = feature_encoder(data["tags"])

#correlation matrix to help us in feature selection
columns_of_interest = ['views', 'likes','comment_count', 'channel_title', 'category_id']
corr_matrix = data[columns_of_interest].corr()
print(corr_matrix)

corr = data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['likes']>0.5)]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


#Extract Featues and output
X = data[['views', 'comment_count', 'tags']].iloc[:,:]
print("before\n", X)
X = feature_scaling(X,0,10)
print("after\n", X)
Y=data[['likes']].iloc[:,:]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)

#Multiple linear regression
cls = linear_model.LinearRegression()
fitModel = cls.fit(X_train,y_train)
prediction = cls.predict(X_test)
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error of linear regression model', metrics.mean_squared_error(np.asarray(y_test), prediction))
print(fitModel.score(X_test,y_test))

#Polynomial regession
poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

