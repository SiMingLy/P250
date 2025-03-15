import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 解析数据
def parse_data(file_path, scale_factor=1/3000):
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
    y = np.array([2 * (n_samples - 1 - i) for i in range(n_samples)]) * scale_factor  # 对标签乘以比例系数

    return X, y

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
    # 设置比例系数
    scale_factor = 1 / 3000

    # 解析训练数据
    X, y = parse_data('G1-C1-FFT100.csv', scale_factor=scale_factor)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 五折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    all_scores = []

    for train_idx, val_idx in kfold.split(X, y):
        print(f'Training fold {fold_no}...')

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 构建随机森林回归模型
        model = RandomForestRegressor(n_estimators=50, random_state=42)

        # 训练模型
        model.fit(X_train, y_train)

        # 评估模型
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        print(f'Fold {fold_no} - Validation MAE: {val_mae:.4f}')
        all_scores.append(val_mae)

        fold_no += 1

    # 输出五折交叉验证的平均 MAE
    print(f'Average Validation MAE across all folds: {np.mean(all_scores):.4f}')

    # 加载额外的测试集
    X_test, y_test = parse_data('G2-C1-FFT100.csv', scale_factor=scale_factor)  # 假设测试集文件为 G2-C1-FFT100.csv
    X_test = scaler.transform(X_test)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 计算其他指标
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R²: {test_r2:.4f}')

    # 绘制真实值与预测值的折线图
    plot_true_vs_predicted(y_test, y_pred)

    # 将模型输出与理想输出存入表格
    results = pd.DataFrame({
        'True Value': y_test,
        'Predicted Value': y_pred
    })
    results.to_csv('test_results_rf.csv', index=False)
    print('Test results saved to test_results_rf.csv')