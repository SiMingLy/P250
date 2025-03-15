import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    y = np.array([2 * (n_samples - 1 - i) / 3000 for i in range(n_samples)])

    return X, y

# 构建更复杂的模型
def build_complex_model(input_shape):
    model = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 使用 padding='same'
        Conv1D(128, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 使用 padding='same'
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 使用 padding='same'
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
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

# 绘制真实值与预测值的折线图
def plot_true_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Value', marker='o')
    plt.plot(y_pred, label='Predicted Value', marker='x')
    plt.title('True Value vs Predicted Value')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 解析训练数据
    X, y = parse_data('G1-C1-FFT100.csv')

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 五折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    all_scores = []

    for train_idx, val_idx in kfold.split(X, y):
        print(f'Training fold {fold_no}...')

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 构建更复杂的模型
        input_shape = (X_train.shape[1], 1)
        model = build_complex_model(input_shape)

        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=250,
            batch_size=256,
            validation_data=(X_val, y_val),
            verbose=1
        )

        # 可视化训练过程
        plot_training_history(history)

        # 评估模型
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold {fold_no} - Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}')
        all_scores.append(val_mae)

        fold_no += 1

    # 输出五折交叉验证的平均 MAE
    print(f'Average Validation MAE across all folds: {np.mean(all_scores):.4f}')

    # 加载额外的测试集
    X_test, y_test = parse_data('G2-C1-FFT100.csv')  # 假设测试集文件为 G2-C1-FFT100.csv
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 评估测试集
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test MAE: {test_mae:.4f}')

    # 预测测试集
    y_pred = model.predict(X_test).flatten()

    # 计算其他指标
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test R²: {test_r2:.4f}')

    # 绘制真实值与预测值的折线图
    plot_true_vs_predicted(y_test * 3000, y_pred * 3000)

    # 将模型输出与理想输出存入表格
    results = pd.DataFrame({
        'True Value': y_test * 3000,
        'Predicted Value': y_pred * 3000
    })
    results.to_csv('test_results.csv', index=False)
    print('Test results saved to test_results.csv')