import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Input

# 解析数据并生成滑动窗口
def parse_data(file_path, window_size=127, factor=1):
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
    y = np.array([2 * (n_samples - 1 - i) / factor for i in range(n_samples)])

    # 生成滑动窗口数据
    X_windowed = []
    y_windowed = []
    for i in range(len(X) - window_size):
        X_windowed.append(X[i:i + window_size])
        y_windowed.append(y[i + window_size - 1])  # 使用窗口的最后一行对应的输出

    X_windowed = np.array(X_windowed)
    y_windowed = np.array(y_windowed)

    return X_windowed, y_windowed

# 构建更复杂的模型
def build_complex_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  # 添加 Input 层
        Conv1D(64, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(128, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1)
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
    train_path = '../Datasets/G1-C1-FFT100.csv'
    test_path = '../Datasets/G2-C1-FFT100.csv'
    factor = 1
    window_size = 127
    epoch = 500
    batch_size = 512

    X, y = parse_data(train_path, window_size=window_size, factor=factor)

    # 数据标准化
    scaler = StandardScaler()
    X = np.array([scaler.fit_transform(x) for x in X])  # 对每个窗口单独标准化
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # 调整形状为 (n_samples, window_size, n_features)

    # 五折交叉验证
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    fold_no = 1
    all_scores = []

    for train_idx, val_idx in kfold.split(X, y):
        print(f'Training fold {fold_no}...')

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 构建更复杂的模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_complex_model(input_shape)

        # 编译模型
        model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse', metrics=['mae'])

        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=epoch,
            batch_size=batch_size,
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
    X_test, y_test = parse_data(test_path, window_size=window_size, factor=factor)
    X_test = np.array([scaler.transform(x) for x in X_test])  # 使用训练集的 scaler 标准化
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

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
    plot_true_vs_predicted(y_test * factor, y_pred * factor)

    # 将模型输出与理想输出存入表格
    results = pd.DataFrame({
        'True Value': y_test * factor,
        'Predicted Value': y_pred * factor
    })
    # results.to_csv('../Outputs/test_results.csv', index=False)
    print('Test results saved to test_results.csv')