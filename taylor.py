import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.optim as optim
import torch
#Torch model
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Load the CSV file
file_path = 'https://raw.githubusercontent.com/revquant/Taylor-Popularity-Prediction-Test/main/taylor_swift_spotify.csv'
data = pd.read_csv(file_path)
scaler = StandardScaler()
features = data[['number', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
features = scaler.fit_transform(features)
popularity = data['popularity']

# PCA Stuff
pca = PCA()
pca_factors = pca.fit(features)
cumulative_variance = pca_factors.explained_variance_ratio_.cumsum()
threshold = 0.90
n_components = next(i for i, total_var in enumerate(cumulative_variance) if total_var >= threshold) + 1
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(features)

# Check PCA data
data['first_principal_component'] = principal_components[:, 0]
correlation_1 = data['first_principal_component'].corr(data['popularity'])
print(f'Correlation between first principal component and popularity: {correlation_1}')

data['mean_principal_components'] = principal_components.mean(axis=1)
correlation_2 = data['mean_principal_components'].corr(data['popularity'])
print(f'Correlation between mean principal component and popularity: {correlation_2}')
correlationx = 0
maxvar = 0
correlationmin = 0
minvar = 0
for i in range(principal_components.shape[1]):
    correlationv = pd.Series(principal_components[:, i]).corr(data['popularity'])
    print(f'Principal Component {i+1}: {correlationv}')
    if correlationv > correlationx:
        correlationx = correlationv
        maxvar = i+1
    if correlationv < correlationmin:
        correlationmin = correlationv
        minvar = i+1
print(f"Max correlation: {correlationx} at location {maxvar}")
print(f"Min correlation: {correlationmin} at location {minvar}")
# Check factor analysis
fa = FactorAnalysis(n_components=1)
factors = fa.fit_transform(features)
data['factor_score'] = factors
correlation = data['factor_score'].corr(data['popularity'])
print("Correlation between mean and popularity: ", data['mean'].corr(data['popularity']))
print(f'Correlation between factor score and popularity: {correlation}')
print(data.head())
for i in range(n_components):
    data[f'principal_component_{i+1}'] = principal_components[:, i]
# Current best: principal component 6
def evaluate_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Evaluate KNN using the principal components
weights = [1] * n_components
if n_components >= 6:
    weights[5] = 2  # Double the weight of Principal Component 6 (index 5)

# Combine weighted components into a new feature
weighted_components = sum(weights[i] * data[f'principal_component_{i+1}'] for i in range(n_components))
data['weighted_components'] = weighted_components
X = data[[f'principal_component_{i+1}' for i in range(n_components)]]
X2 = data[['principal_component_6']]
X3 = pd.DataFrame(scaler.fit_transform(data[['weighted_components']]), columns=['weighted_components'])
X4 = data[['factor_score']]
mse = evaluate_knn(X, popularity)
mse2 = evaluate_knn(X2, popularity)
mse3 = evaluate_knn(X3, popularity)
mse4 = evaluate_knn(X4, popularity)
print(f'Mean Squared Error using principal components: {mse}')
print(f'Mean Squared Error using best principal components: {mse2}')
print(f'Mean Squared Error using weighted principal components: {mse3}')
print(f'Mean Squared Error using factor analysis: {mse4}')
#Test Torch
y = popularity
X_train, X_test, y_train, y_test = train_test_split(X3, y, test_size=0.1, random_state=45)
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
input_dim = X_train.shape[1]
model = RegressionModel(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
num_epochs = 50000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 15000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
def predict_pytorch(model, input_features):
    model.eval()
    input_tensor = torch.tensor([input_features], dtype=torch.float32)
    with torch.no_grad():
        prediction_tensor = model(input_tensor)
    return prediction_tensor.item()
knn = KNeighborsRegressor(n_neighbors=5)  # Adjust n_neighbors if needed
knn.fit(X_train, y_train)
def predict_knn(knn_model, input_features):
    return knn_model.predict([input_features])
with torch.no_grad():
    # Predictions
    y_pred_tensor = model(X_test_tensor)    
    # Calculate MSE
    mse = nn.functional.mse_loss(y_pred_tensor, y_test_tensor)
    print(f'Mean Squared Error using PyTorch model: {mse}')
    print(f'Mean of X_train: {X_train.mean().values}')
    print(f'Standard deviation of X_train: {X_train.std().values}')

input_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Replace with actual features
predict_knn(knn, input_features)