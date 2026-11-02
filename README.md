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
