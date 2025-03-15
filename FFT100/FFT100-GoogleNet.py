import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# 解析数据
def parse_data(file_path):
    df = pd.read_csv(file_path, header=None)
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
    n_samples = len(samples)
    y = np.array([2 * (n_samples - 1 - i) for i in range(n_samples)])

    return X, y

# 构建模型
def build_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),  # 使用 padding='same'
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)  # 回归任务，输出为 1 个值
    ])
    return model

# 可视化训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 解析数据
    X, y = parse_data('output_data.csv')

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 调整输入形状
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 构建模型
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 可视化训练过程
    plot_training_history(history)

    # 评估模型
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test MAE: {test_mae:.4f}')