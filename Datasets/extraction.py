import os
import pandas as pd
from pathlib import Path

def extract_and_combine_columns(folder_path, output_file, n=10):
    """
    从文件夹中的所有 CSV 文件中提取前 n 列，并横向拼接保存到新的 CSV 文件中。

    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径。
        output_file (str): 输出文件的名称。
        n (int): 提取的列数，默认为 10。
    """
    # 获取文件夹中的所有 CSV 文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("文件夹中没有 CSV 文件。")
        return

    # 初始化一个空列表，用于存储提取的数据
    combined_data = []

    # 遍历每个 CSV 文件
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path, header=None)
            # 提取前 n 列
            extracted_data = df.iloc[:, :n]
            # 将提取的数据添加到 combined_data 中
            combined_data.append(extracted_data)
            print(f"已处理文件: {csv_file}")
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    # 将所有提取的数据横向拼接成一个 DataFrame
    if combined_data:
        combined_df = pd.concat(combined_data, axis=1)  # 横向拼接
        # 保存到新的 CSV 文件
        combined_df.to_csv(output_file, index=False, header=False)
        print(f"数据已保存到 {output_file}")
    else:
        print("没有提取到任何数据。")

if __name__ == "__main__":
    # 设置文件夹路径和输出文件名
    folder_path = 'G2'
    output_file = "G2-C1-20.csv"  # 输出文件名
    n = 20  # 提取的列数

    # 调用函数
    extract_and_combine_columns(folder_path, output_file, n)