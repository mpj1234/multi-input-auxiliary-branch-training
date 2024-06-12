# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2024/3/24 15:26
  @version V1.0
"""
import os

import torch

import matplotlib.pyplot as plt

# 设置字体为times new roman
plt.rcParams['font.family'] = 'Times New Roman'

origin_model_path = './runs-experiment/train/v9-c-double-implicitConvCBFuse/weights/best.pt'
###########
index = 36

device = torch.device("cpu")
ckpt = torch.load(origin_model_path, map_location='cpu')

model = ckpt['model']
m = model.model[index]
###################
v = m.ia.implicit.numpy().squeeze()
# v = m.im.implicit.numpy().squeeze()
v = list(v)

# 横坐标是从0到v的长度，纵坐标是v的值
# plt.bar(range(len(v) + 60), v + [0] * 60)
plt.figure(figsize=(4, 3))
plt.bar(range(len(v)), v)
# plt.xlabel('index')
# plt.ylabel('value')
# plt.title('implicitA-' + str(index))
plt.savefig('implicit_ia_{}.png'.format(index), transparent=True, bbox_inches='tight', pad_inches=0.0, dpi=600)
# plt.savefig('implicit_im_{}.png'.format(index), transparent=True, bbox_inches='tight', pad_inches=0.0, dpi=600)

# with open('implicit_ia_{}.csv'.format(index), 'w') as f:
# 	for i, item in enumerate(v):
# 		f.write(str(i) + ',' + str(item) + '\n')
