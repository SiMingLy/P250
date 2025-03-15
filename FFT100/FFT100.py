import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 读取CSV文件
df = pd.read_csv('output_data.csv', header=None)

# 解析每行数据为二维点，并统一长度
samples = []
max_points = 0

for line in df[0]:
    points = line.split(',')
    current_points = []
    for p in points:
        x, y = p.split(':')
        current_points.append((float(x), float(y)))
    samples.append(current_points)
    max_points = max(max_points, len(current_points))

# 填充到统一长度并展平
processed_samples = []
for sample in samples:
    padded = sample + [(0.0, 0.0)] * (max_points - len(sample))
    flattened = [coord for point in padded for coord in point]
    processed_samples.append(flattened)

X = np.array(processed_samples)

# 生成目标值（基于行号）
n_samples = len(samples)
y = np.array([2 * (n_samples - 1 - i) for i in range(n_samples)])

# 顺序划分数据集（前80%训练，后20%测试）
split_idx = int(n_samples * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 评估模型
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# 保存模型（可选）
# model.save('my_model.h5')