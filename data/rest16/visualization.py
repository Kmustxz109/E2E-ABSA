import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

x = open('outputs5_file.txt', mode='r')
A = []
A_row=0
lines = x.readlines()
data_list = []
data = []
for l in lines:
    if l.startswith('[[') & len(data) > 0:
        data_list.append(data)
        data = []
    d = l.strip('\n')
    d = d.split(',')
    d = [num.strip(']').strip('[') for num in d]
    data.append(d)
data_list.append(data)
print(len(data_list))

A = data_list[0]
# A = A[:17]
# A = A[:][:17]
# print(A)
A = np.array(A, dtype=float)
A = A[:18, :18]
for i in range(18):
    A[i][i]=0
# print(A)
print(A.shape)




# a = np.random.rand(4, 3)
fig, ax = plt.subplots(figsize=(6, 6))
# 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
# 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(A, columns=['The','duck','confit','is','always','amazing','and','the','foie','gras','terrine','with','figs','was','out','of','this','world'], index=['The','duck','confit','is','always','amazing','and','the','foie','gras','terrine','with','figs','was','out','of','this','world']),
                         annot=False, vmax=0.14, vmin=0, xticklabels=True, yticklabels=True, cbar=True,cbar_kws={"shrink": 0.8},square=True, cmap="YlGnBu")
# sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
#            square=True, cmap="YlGnBu")
# ax.set_title('dsf', fontsize=18)
# ax.set_ylabel('df', fontsize=18)
# ax.set_xlabel('er', fontsize=18)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.show()