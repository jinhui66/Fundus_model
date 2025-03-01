import pandas as pd


def xlsx_to_csv_pd(src_file, dst_file):
    data_xls = pd.read_excel(src_file, index_col=0)
    data_xls.to_csv(dst_file, encoding='utf-8')


# if __name__ == '__main__':
#     src_file = r"C:\Users\y8549\Desktop\confuse\外包数据集\Left_Fundus_Classification.xlsx"
#     dst_file = r"C:\Users\y8549\Desktop\confuse\外包数据集\Left_Fundus_Classification.csv"
#     xlsx_to_csv_pd(src_file, dst_file)

which = 'Left'
whcih = 'Right'

# 读取excel文件
# df = pd.read_excel(f"./data/{which}_Fundus_Classification.xlsx")
df = pd.read_excel(f"./data/Training_Dataset.xlsx")
print(len(df))
prefix_path = r"./data"

# 假设第一列是名字，第二列是正常，接下来的7列是疾病数据
# 提取正常列和疾病列
normal_and_disease_columns = df.columns[7:15] 

# 筛选“无疾病或只有一个疾病”的记录
valid_data = df
if not valid_data.empty:
    valid_data.to_csv(fr"{prefix_path}/double_valid_data.csv", index=False)
    print(f"已导出：double_valid_data.csv, 记录数:{len(valid_data)}")

# # 筛选“无疾病或只有一个疾病”的记录
# group_0_or_1 = df[(df['disease_count'] <= 1) & (df[normal_and_disease_columns].sum(axis=1) > 0)]
# if not group_0_or_1.empty:
#     group_0_or_1.to_csv(fr"{prefix_path}/Left_group_0_or_1_diseases.csv", index=False)
#     print(f"已导出：group_0_or_1_diseases.csv, 记录数:{len(group_0_or_1)}")

# # 筛选“有两个及以上疾病”的记录，并分别导出
# for count in range(2, 9):  # 疾病数量从2到8
#     group_df = df[(df['disease_count'] == count) & (df[normal_and_disease_columns].sum(axis=1) > 0)]
#     if not group_df.empty:  # 仅在有记录时导出
#         filename = fr"{prefix_path}/Left_group_{count}_diseases.csv"
#         group_df.to_csv(filename, index=False)
#         print(f"已导出：{filename}, 记录数:{len(group_df)}")
