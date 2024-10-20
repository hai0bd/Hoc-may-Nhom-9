from flask import Flask, request, render_template
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__) #khởi tạo 1 web với flask

# Bước 1: Tải dữ liệu và huấn luyện mô hình

# Tải và cbi dữ liệu từ sklearn
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Chia data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu đảm bảo các đặc trưng có cùng tầm gtri
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình trên tập dl đã được chuẩn hóa

# Khởi tạo SVC với tham số probability = true để dự đoán xác suất
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)

#Khởi tạo CART phân loại dựa trên các điều kiện quyết định.
cart_model = DecisionTreeClassifier()
cart_model.fit(X_train_scaled, y_train)

#Khởi tạo mô hình MLP
mlp_model = MLPClassifier(max_iter=1000)
mlp_model.fit(X_train_scaled, y_train)

# Tạo mô hình Stacking với Logistic Regression
estimators = [
    ('svm', svm_model),
    ('cart', cart_model),
    ('mlp', mlp_model)
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    # Lấy tên thuộc tính từ wine_data
    feature_names = wine_data.feature_names
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    input_data = [float(x) for x in request.form.values()]
    input_data_np = np.array(input_data).reshape(1, -1)
    
    # Sử dụng scaler để chuẩn hóa dữ liệu đầu vào
    input_data_scaled = scaler.transform(input_data_np)

    # Dự đoán và độ tin cậy
    prediction = stacking_model.predict(input_data_scaled)
    confidence = stacking_model.predict_proba(input_data_scaled).max() * 100

    return render_template('index.html', prediction=f"Loại rượu: {wine_data.target_names[prediction[0]]}",
                           confidence=f"Độ tin cậy: {confidence:.2f}%", feature_names=wine_data.feature_names)

if __name__ == '__main__':
    app.run(debug=True)

from sklearn.ensemble import StackingClassifier
estimators = [
    ('svm', svm_model),
    ('cart', cart_model),
    ('mlp', mlp_model)
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=SVC())
stacking_model.fit(X_train_scaled, y_train)
stacking_pred = stacking_model.predict(X_test_scaled)
stacking_acc = accuracy_score(y_test, stacking_pred)
print(f"Độ chính xác Stacking: {stacking_acc * 100:.2f}%")