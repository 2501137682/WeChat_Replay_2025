import pandas as pd

# 读取Excel（xlsx用openpyxl，xls用xlrd）
df = pd.read_excel("副本微博热搜榜-2025年1月.xlsx", sheet_name=0)  # sheet_name指定工作表

# 获取全部字段（表头）
columns = df.columns.tolist()
print("所有字段：", columns)