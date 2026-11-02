# ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

digits = datasets.load_digits()

X = digits.data
Y = digits.target

print("Shape of data:", X.shape)
print("shape of labels:", Y.shape)

plt.figure(figsize=(10, 5))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label:{digits.target[i]}")
    plt.axis('off')

plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', gamma='scale', C=10)
svm_model.fit(X_train_scaled, Y_train)

y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy:{accuracy*100:.2f}%\n")

cm = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", cm, "\n")

print("classification Report:\n", classification_report(Y_test, y_pred))

plt.figure(figsize=(10, 5))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Predicted:{y_pred[i]}")
    plt.axis('off')

plt.show()



#mlpr 01 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
data = pd.read_csv("Iris.csv") 
print("First 5 rows of dataset:") 
print(data.head()) 
print("\nDataset Info:") 
print(data.info()) 
X = data[[ 
"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" 
]] 
y = data["Species"] 
le = LabelEncoder() 
y_encoded = le.fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split( 
X, y_encoded, test_size=0.2, random_state=42 
) 
lda = LinearDiscriminantAnalysis() 
lda.fit(X_train, y_train) 
y_pred = lda.predict(X_test) 
print("\nAccuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report( 
y_test, y_pred, target_names=le.classes_ 
)) 
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
sample_flower = pd.DataFrame( 
[[5.2, 3.4, 1.5, 0.2]], 
columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"] 
) 
predicted_class = lda.predict(sample_flower) 
predicted_species = le.inverse_transform(predicted_class) 
print("\nThe given flower belongs to species:", predicted_species[0])

ml pr 02
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import mean_squared_error, r2_score 
data = pd.read_csv("uber.csv") 
data.head() 
data = data.dropna(subset=[ 
"fare_amount", "pickup_datetime", 
"pickup_longitude", "pickup_latitude", 
"dropoff_longitude", "dropoff_latitude", 
"passenger_count" 
]).copy() 
data["pickup_datetime"] = pd.to_datetime( 
data["pickup_datetime"], errors="coerce" 
) 
data = data.dropna(subset=["pickup_datetime"]) 
data.head() 
data["pickup_hour"] = data["pickup_datetime"].dt.hour 
data["pickup_day"] = data["pickup_datetime"].dt.day 
data["pickup_month"] = data["pickup_datetime"].dt.month 
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday 
data = data.drop(columns=["pickup_datetime"]) 
data.head() 
data = data[ 
(data["fare_amount"] > 0) & 
(data["fare_amount"] < 500) 
] 
data = data[ 
(data["passenger_count"] > 0) & 
(data["passenger_count"] <= 6) 
] 
data = data[ 
data["pickup_longitude"].between(-75, -72) & 
data["pickup_latitude"].between(40, 42) 
] 
data = data[ 
data["dropoff_longitude"].between(-75, -72) & 
data["dropoff_latitude"].between(40, 42) 
] 
data.shape 
def haversine_distance(lat1, lon1, lat2, lon2): 
R = 6371 
phi1 = np.radians(lat1) 
phi2 = np.radians(lat2) 
delta_phi = np.radians(lat2 - lat1) 
delta_lambda = np.radians(lon2 - lon1) 
a = ( 
np.sin(delta_phi / 2) ** 2 
+ np.cos(phi1) * np.cos(phi2) 
* np.sin(delta_lambda / 2) ** 2 
) 
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) 
return R * c 
data["distance_km"] = haversine_distance( 
data["pickup_latitude"], 
data["pickup_longitude"], 
data["dropoff_latitude"], 
data["dropoff_longitude"] 
) 
data.head() 
sns.boxplot(y=data["fare_amount"]) 
plt.title("Outliers in Fare Amount") 
plt.show() 
sns.boxplot(y=data["distance_km"]) 
plt.title("Outliers in Distance (km)") 
plt.show() 
data = data.drop(columns=[ 
"pickup_latitude", "pickup_longitude", 
"dropoff_latitude", "dropoff_longitude" 
]) 
plt.figure(figsize=(10, 7)) 
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm") 
plt.title("Correlation Heatmap") 
plt.tight_layout() 
plt.show() 
X = data.drop("fare_amount", axis=1) 
y = data["fare_amount"] 
X = X.select_dtypes(include=[np.number]) 
valid = X.notna().all(axis=1) & y.notna() 
X, y = X.loc[valid], y.loc[valid] 
X_train, X_test, y_train, y_test = train_test_split( 
X, y, test_size=0.2, random_state=42 
) 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
lr = LinearRegression() 
lr.fit(X_train_scaled, y_train) 
y_pred_lr = lr.predict(X_test_scaled) 
ridge = Ridge(alpha=1.0) 
ridge.fit(X_train_scaled, y_train) 
y_pred_ridge = ridge.predict(X_test_scaled) 
lasso = Lasso(alpha=0.1) 
lasso.fit(X_train_scaled, y_train) 
y_pred_lasso = lasso.predict(X_test_scaled) 
def evaluate(y_test, y_pred, model_name): 
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2 = r2_score(y_test, y_pred) 
print(f"{model_name} -> RMSE: {rmse:.2f}, R2: {r2:.2f}") 
evaluate(y_test, y_pred_lr, "Linear Regression") 
evaluate(y_test, y_pred_ridge, "Ridge Regression") 
evaluate(y_test, y_pred_lasso, "Lasso Regression") 
results = pd.DataFrame({ 
'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'], 
'RMSE': [ 
np.sqrt(mean_squared_error(y_test, y_pred_lr)), 
np.sqrt(mean_squared_error(y_test, y_pred_ridge)), 
np.sqrt(mean_squared_error(y_test, y_pred_lasso)) 
], 
'R2': [ 
r2_score(y_test, y_pred_lr), 
r2_score(y_test, y_pred_ridge), 
r2_score(y_test, y_pred_lasso) 
] 
}) 
print("\nModel Comparison:\n", results)

# ml pr 04 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
iris = pd.read_csv("Iris.csv") 
print(iris.head()) 
X = iris.iloc[:, :-1].values 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
inertia = [] 
K = range(1, 11) 
for k in K: 
kmeans = KMeans(n_clusters=k, random_state=42) 
kmeans.fit(X_scaled) 
inertia.append(kmeans.inertia_) 
plt.figure(figsize=(8,5)) 
plt.plot(K, inertia, 'bo-') 
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42) 
y_kmeans = kmeans.fit_predict(X_scaled) 
iris['Cluster'] = y_kmeans 
print(iris.head()) 
plt.figure(figsize=(8,5)) 
plt.scatter(X_scaled[y_kmeans==0, 0], X_scaled[y_kmeans==0, 1], s=100, c='red', 
label='Cluster 1') 
plt.scatter(X_scaled[y_kmeans==1, 0], X_scaled[y_kmeans==1, 1], s=100, c='blue', 
label='Cluster 2') 
plt.scatter(X_scaled[y_kmeans==2, 0], X_scaled[y_kmeans==2, 1], s=100, c='green', 
label='Cluster 3') 
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', 
label='Centroids') 
plt.xlabel('Sepal Length (scaled)') 
plt.ylabel('Sepal Width (scaled)') 
plt.title('K-Means Clustering on Iris Dataset') 
plt.legend() 
plt.show()

# ml pr 05 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt 
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] 
data = pd.read_csv('car_evaluation.csv', names=columns, header=None) 
print(data.head()) 
le = LabelEncoder() 
for col in data.columns: 
data[col] = le.fit_transform(data[col]) 
X = data.drop('class', axis=1) 
y = data['class'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_model.fit(X_train, y_train) 
y_pred = rf_model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Accuracy: {accuracy*100:.2f}%\n") 
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
features = X.columns 
importances = rf_model.feature_importances_ 
indices = importances.argsort()[::-1] 
plt.figure(figsize=(8,5)) 
plt.bar(range(X.shape[1]), importances[indices], color='skyblue', 
align='center') 
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45) 
plt.title("Feature Importance") 
plt.show()

# ml 06 
import numpy as np 
import random 
class TicTacToe: 
def __init__(self): 
self.board = [' ']*9 
self.current_winner = None 
def reset(self): 
self.board = [' ']*9 
self.current_winner = None 
return self.get_state() 
def get_state(self): 
return ''.join(self.board) 
def available_actions(self): 
return [i for i, spot in enumerate(self.board) if spot == ' '] 
def make_move(self, square, letter): 
if self.board[square] == ' ': 
self.board[square] = letter 
if self.winner(square, letter): 
self.current_winner = letter 
return True 
return False 
def winner(self, square, letter): 
row_ind = square // 3 
if all([self.board[row_ind*3 + i] == letter for i in range(3)]): 
return True 
col_ind = square % 3 
if all([self.board[col_ind + 3*i] == letter for i in range(3)]): 
return True 
if square % 2 == 0: 
if all([self.board[i] == letter for i in [0,4,8]]) or all([self.board[i] == letter for i in 
[2,4,6]]): 
return True 
return False 
def is_done(self): 
return self.current_winner is not None or ' ' not in self.board 
def render(self): 
print() 
for i in range(3): 
print(self.board[i*3:(i+1)*3])
