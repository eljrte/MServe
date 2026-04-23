# 处理CDF数据
import numpy as np
import matplotlib.pyplot as plt

def calculate_and_visualize_cdf(data_list, output_file="./cdf_data_new/cdf_data_2451.txt", plot_file="./cdf_picture_new/cdf_plot_2451.png"):
    """
    处理列表数据，计算CDF并生成可视化图
    
    参数:
    data_list: 输入的数据列表
    output_file: 输出文件名，默认为 "cdf_data.txt"
    plot_file: 图片文件名，默认为 "cdf_plot.png"
    """
    
    # 从大到小排序
    sorted_data = sorted(data_list, reverse=True)
    
    # 归一化
    total_sum = sum(sorted_data)
    if total_sum == 0:
        normalized_data = [0 for _ in sorted_data]
    else:
        normalized_data = [x / total_sum for x in sorted_data]
    
    # 计算累积和
    cumulative_sum = []
    running_sum = 0
    for value in normalized_data:
        running_sum += value
        cumulative_sum.append(running_sum)
    
    # 写入文件
    with open(output_file, 'w') as f:
        for i, (orig_val, norm_val, cum_val) in enumerate(zip(sorted_data, normalized_data, cumulative_sum)):
            f.write(f"{cum_val:.5f}\n")
    
    # 生成CDF图
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_sum, marker='o', linestyle='-', markersize=3)
    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Index (Sorted Descending)')
    plt.ylabel('Cumulative Sum')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(plot_file)
    plt.show()
    
    print(f"CDF data saved to {output_file}")
    print(f"CDF plot saved to {plot_file}")
    
    return cumulative_sum
