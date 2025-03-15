import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
class ComplexModel(nn.Module):
    def __init__(self, input_shape):
        super(ComplexModel, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv1d(input_shape[1], 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # 卷积层 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)  # Batch Normalization
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # 卷积层 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)  # Batch Normalization
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # 卷积层 4
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)  # Batch Normalization
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        # 动态计算全连接层的输入维度
        self.fc1_input_dim = 512 * (input_shape[0] // 16)  # 根据池化层计算
        self.flatten = nn.Flatten()

        # 全连接层 1
        self.fc1 = nn.Linear(4608, 1024)
        self.bn5 = nn.BatchNorm1d(1024)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.5)

        # 全连接层 2
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)  # Batch Normalization
        self.dropout2 = nn.Dropout(0.5)

        # 全连接层 3
        self.fc3 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)  # Batch Normalization
        self.dropout3 = nn.Dropout(0.5)

        # 输出层
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        # 卷积层 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 卷积层 2
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 卷积层 3
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 卷积层 4
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # 展平
        x = self.flatten(x)

        # 全连接层 1
        x = torch.relu(self.bn5(self.fc1(x)))
        x = self.dropout1(x)

        # 全连接层 2
        x = torch.relu(self.bn6(self.fc2(x)))
        x = self.dropout2(x)

        # 全连接层 3
        x = torch.relu(self.bn7(self.fc3(x)))
        x = self.dropout3(x)

        # 输出层
        x = self.fc4(x)
        return x

# 可视化训练过程
def plot_training_history(train_losses, val_losses, train_maes, val_maes):
    plt.figure(figsize=(12, 6))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 MAE
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Training MAE')
    plt.plot(val_maes, label='Validation MAE')
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
    train_path = '../Datasets/G1-C1-10.csv'
    test_path = '../Datasets/G1-C1-10.csv'
    test_path2 = '../Datasets/G2-C1-10.csv'
    factor = 1
    window_size = 127
    epochs = 2000
    batch_size = 2048
    learning_rate = 0.001

    X, y = parse_data(train_path, window_size=window_size, factor=factor)

    # 数据标准化
    scaler = StandardScaler()
    X = np.array([scaler.fit_transform(x) for x in X])  # 对每个窗口单独标准化
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # 调整形状为 (n_samples, window_size, n_features)

    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # 调整形状为 (n_samples, n_features, window_size)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 调整形状为 (n_samples, 1)

    # 五折交叉验证
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    fold_no = 1
    all_scores = []

    for train_idx, val_idx in kfold.split(X, y):
        print(f'Training fold {fold_no}...')

        # 划分训练集和验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 创建 DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 构建模型
        input_shape = (X_train.shape[2], X_train.shape[1])  # (window_size, n_features)
        model = ComplexModel(input_shape).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        train_losses = []
        val_losses = []
        train_maes = []  # 记录训练 MAE
        val_maes = []  # 记录验证 MAE

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            epoch_train_mae = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                epoch_train_mae += mean_absolute_error(batch_y.cpu().numpy(), outputs.detach().cpu().numpy())
            train_losses.append(epoch_train_loss / len(train_loader))
            train_maes.append(epoch_train_mae / len(train_loader))  # 记录训练 MAE

            model.eval()
            epoch_val_loss = 0
            epoch_val_mae = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
                    epoch_val_mae += mean_absolute_error(batch_y.cpu().numpy(), outputs.detach().cpu().numpy())
            val_losses.append(epoch_val_loss / len(val_loader))
            val_maes.append(epoch_val_mae / len(val_loader))  # 记录验证 MAE

            if (epoch + 1) % 50 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
                    f'Train MAE: {train_maes[-1]:.4f}, Val MAE: {val_maes[-1]:.4f}')

        # 可视化训练过程
        plot_training_history(train_losses, val_losses, train_maes, val_maes)

        # 评估模型
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val.to(device)).cpu().numpy()
            val_mae = mean_absolute_error(y_val.numpy(), val_preds)
            print(f'Fold {fold_no} - Validation MAE: {val_mae:.4f}')
            all_scores.append(val_mae)

        fold_no += 1

    # 输出五折交叉验证的平均 MAE
    print(f'Average Validation MAE across all folds: {np.mean(all_scores):.4f}')

    # 加载额外的测试集
    X_test, y_test = parse_data(test_path, window_size=window_size, factor=factor)
    X_test = np.array([scaler.transform(x) for x in X_test])  # 使用训练集的 scaler 标准化
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)  # 调整形状为 (n_samples, n_features, window_size)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 调整形状为 (n_samples, 1)

    # 预测测试集
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()

    # 计算其他指标
    test_mse = mean_squared_error(y_test.numpy(), y_pred)
    test_mae = mean_absolute_error(y_test.numpy(), y_pred)
    test_r2 = r2_score(y_test.numpy(), y_pred)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R²: {test_r2:.4f}')

    # 绘制真实值与预测值的折线图
    plot_true_vs_predicted(y_test.numpy() * factor, y_pred * factor)

    X_test, y_test = parse_data(test_path2, window_size=window_size, factor=factor)
    X_test = np.array([scaler.transform(x) for x in X_test])  # 使用训练集的 scaler 标准化
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)  # 调整形状为 (n_samples, n_features, window_size)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # 调整形状为 (n_samples, 1)

    # 预测测试集
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()

    # 计算其他指标
    test_mse = mean_squared_error(y_test.numpy(), y_pred)
    test_mae = mean_absolute_error(y_test.numpy(), y_pred)
    test_r2 = r2_score(y_test.numpy(), y_pred)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test R²: {test_r2:.4f}')

    # 绘制真实值与预测值的折线图
    plot_true_vs_predicted(y_test.numpy() * factor, y_pred * factor)

    # 将模型输出与理想输出存入表格
    results = pd.DataFrame({
        'True Value': y_test.numpy().flatten() * factor,
        'Predicted Value': y_pred.flatten() * factor
    })
    # results.to_csv('../Outputs/test_results.csv', index=False)
    print('Test results saved to test_results.csv')