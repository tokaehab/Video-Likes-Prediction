import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def pre_processing():
    # Read data
    data = pd.read_csv("VideoLikesDatasetClassification.csv")

    # Drop rows of blank values
    data.dropna(how='any', inplace=True)

    # Handle date-time format
    data['trending_date'] = pd.to_datetime(data['trending_date'], format='%y.%d.%m')
    data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    # Split date into 2 columns
    data.insert(5, 'publish_date', data['publish_time'].dt.date)
    data['publish_time'] = data['publish_time'].dt.time
    data['publish_date'] = pd.to_datetime(data['publish_date'])

    # Encode features
    data["video_id"] = feature_encoder(data["video_id"])
    data["title"] = feature_encoder(data["title"])
    data["channel_title"] = feature_encoder(data["channel_title"])
    data["comments_disabled"] = feature_encoder(data["comments_disabled"])
    data["ratings_disabled"] = feature_encoder(data["ratings_disabled"])
    data["video_error_or_removed"] = feature_encoder(data["video_error_or_removed"])
    data["tags"] = feature_encoder(data["tags"])
    data["VideoPopularity"] = feature_encoder(data["VideoPopularity"])

    # Correlation matrix to help us in features selection
    columns_of_interest = ['views', 'VideoPopularity', 'comment_count', 'channel_title', 'category_id']
    corr_matrix = data[columns_of_interest].corr()
    print("Correlation Matrix:\n", corr_matrix)
    print("-----------------------------------------------------------------------------\n\n")

    # Get features with more than 50% correlation with VideoPopularity using heatmap
    corr = data.corr()
    best_features = corr.index[abs(corr['VideoPopularity']) > 0.02]  # TODO:Threshold?
    best_features = best_features.delete(-1)
    X = data[best_features]
    plt.subplots(figsize=(6, 4))
    top_corr = data[best_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    # Extract features and output
    # scaler = MinMaxScaler()
    # scaled = scaler.fit_transform(data)
    return data, X


def feature_scaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def feature_encoder(X):
    lbl = LabelEncoder()
    lbl.fit(list(X.values))
    X = lbl.transform(list(X.values))
    return X
