import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import random
from transformers import AutoTokenizer, AutoModel
import torch

# Load data
df = pd.read_csv('sp500.csv')

# Use actual data instead of simulated data
data = pd.DataFrame({
    'Price': df['price'],
    'Market Cap': df['market_cap'],
    'P/E Ratio': df['pe_ratio'],
    'Sector': df['sector'],
    'Volatility': df['volatility'],
    'Volatility_Category': df['volatility_category'],
    'Summary': df['Summary']
})

# You can also keep track of tickers if needed
data['Ticker'] = df['ticker']

# Step 2: Feature Normalization and Encoding
scaler = MinMaxScaler()
numerical_features = ['Volatility', 'Market Cap', 'P/E Ratio', 'Price']
data_scaled = scaler.fit_transform(data[numerical_features])

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(data[['Sector']])

# Combine numerical and categorical features
features = np.hstack((data_scaled, categorical_encoded))

# Generate text features first
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    # Add padding/truncation
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding (first token) as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    
    return embeddings.flatten()

text_features = np.vstack([get_bert_embedding(text) for text in data['Summary']])
text_features_scaled = MinMaxScaler().fit_transform(text_features)

# Then do DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(features)
data['Cluster'] = clusters

# Filter out outliers (-1 cluster)
mask = data['Cluster'] != -1
data = data[mask]
features = features[mask.values]
text_features_scaled = text_features_scaled[mask.values]  # Also filter text features

# Step 4: Initialize Feedback System
feature_weights = np.zeros(features.shape[1])  # Initialize weights to zero

# Map categorical feature names for interpretability
categorical_feature_names = encoder.get_feature_names_out(['Sector'])
all_feature_names = numerical_features + list(categorical_feature_names)

# Step 5: Swipe Simulation
swipe_count = 0
max_swipes = 10  # Stopping criterion

# Optional: Add feature group weights
numerical_weight = 1.0
categorical_weight = 0.8
text_weight = 0.5

features = np.hstack((
    numerical_weight * data_scaled,
    categorical_weight * categorical_encoded,
    text_weight * text_features_scaled
))

# Update feature names to include text features
text_feature_names = [f'text_component_{i}' for i in range(text_features_scaled.shape[1])]
all_feature_names = numerical_features + list(categorical_feature_names) + text_feature_names

# Update feature weights initialization to account for all features
feature_weights = np.zeros(features.shape[1])

while swipe_count < max_swipes:
    # Score stocks based on current weights
    scores = features @ feature_weights
    recommended_idx = np.argmax(scores)
    recommended_stock = data.iloc[recommended_idx]

    # Show recommendation
    print("\n=== Recommendation " + str(swipe_count + 1) + " ===")
    print("Ticker: " + str(recommended_stock['Ticker']))
    print("Sector: " + str(recommended_stock['Sector']))
    
    print("\nActual Values:")
    for feature in numerical_features:
        print("  " + feature + ": " + str(round(recommended_stock[feature], 2)))
    
    print("\nScaled Values and Weights:")
    # First show numerical features
    for i, feature in enumerate(numerical_features):
        print("  " + feature + ":")
        print("    Scaled Value: " + str(round(features[recommended_idx][i], 4)))
        print("    Weight: " + str(round(feature_weights[i], 4)))
    
    # Then show sector encoding
    sector_start_idx = len(numerical_features)
    for i, sector_feature in enumerate(categorical_feature_names):
        if features[recommended_idx][sector_start_idx + i] > 0:  # Only show active sector
            print("  " + str(sector_feature) + ":")
            print("    Encoded Value: " + str(round(features[recommended_idx][sector_start_idx + i], 4)))
            print("    Weight: " + str(round(feature_weights[sector_start_idx + i], 4)))
    
    print("\nTotal Score: " + str(round(scores[recommended_idx], 4)))

    # Get user feedback
    feedback = input("Do you like this stock? (y/n): ").strip().lower()
    if feedback == 'y':
        # Positive feedback: increase weights for this stock's features
        feature_weights += features[recommended_idx]
    elif feedback == 'n':
        # Negative feedback: decrease weights for this stock's features
        feature_weights -= features[recommended_idx]
    else:
        print("Invalid input. Skipping this recommendation.")

    # Remove the recommended stock from the pool
    data = data.drop(recommended_stock.name)
    features = np.delete(features, recommended_idx, axis=0)

    swipe_count += 1

# Final Output
print("\n=== Final Preferences ===")
print("Feature Weights:")
for i, feature in enumerate(all_feature_names):
    print("  " + str(feature) + ": " + str(round(feature_weights[i], 4)))

print("\nTop Recommended Stocks:")
top_indices = np.argsort(features @ feature_weights)[-5:][::-1]  # Top 5 stocks
print(data.iloc[top_indices])
