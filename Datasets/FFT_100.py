import numpy as np
import pandas as pd
from scipy.fft import fft

def compute_top_frequencies(row, sampling_rate, number):
    """
    对一行数据进行傅里叶变换，并返回前十个最大频率分量及其幅值

    参数:
    row (numpy array): 一维数据
    sampling_rate (float): 采样率

    返回:
    top_frequencies (list): 前十个最大频率分量
    top_magnitudes (list): 对应的幅值
    """
    n = len(row)
    fft_result = fft(row)
    frequencies = np.fft.fftfreq(n, 1 / sampling_rate)
    magnitudes = np.abs(fft_result) / n

    # 只取正频率部分
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    magnitudes = magnitudes[positive_freq_idx]

    # 按幅值排序并取前十个
    sorted_indices = np.argsort(magnitudes)[::-1]
    top_ten_indices = sorted_indices[:number]
    top_frequencies = frequencies[top_ten_indices]
    top_magnitudes = magnitudes[top_ten_indices]

    return list(zip(top_frequencies, top_magnitudes))

def process_fft_csv(input_csv, output_csv, sampling_rate, number):
    """
    处理CSV文件，对每一行进行FFT并保存前十个频率分量及其幅值

    参数:
    input_csv (str): 输入CSV文件路径
    output_csv (str): 输出CSV文件路径
    sampling_rate (float): 采样率
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv, header=None)

    # 对每一行进行处理
    results = []
    for index, row in df.iterrows():
        row_data = row.to_numpy()
        top_freq_mag = compute_top_frequencies(row_data, sampling_rate, number)
        results.append(top_freq_mag)

    # 将结果保存为新的CSV文件
    output_data = []
    for result in results:
        # 将每个频率和幅值对转换为字符串形式，例如 "频率:幅值"
        row_output = [f"{freq:.2f}:{mag:.4f}" for freq, mag in result]
        output_data.append(row_output)

    # 创建DataFrame并保存为CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv, index=False, header=False)

# 示例使用
if __name__ == "__main__":
    sampling_rate = 12800  # 采样率
    number = 100
    for i in range(1,6):
        for j in range(1, 12):
            input_csv = "..\data\训练集\G_5\Test{}\Channel{}.csv".format(i, j)  # 输入CSV文件路径
            output_csv = ".\G5\Test{}\Channel{}.csv".format(i, j)  # 输出CSV文件路径
            process_fft_csv(input_csv, output_csv, sampling_rate, number)
            print(f"处理完成，结果已保存到 {output_csv}")


    
    