# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2024/3/19 22:01
  @version V1.0
"""
import os

import torch

origin_model_path = './runs-experiment/train/v9-c-double-ImplicitConvCBFuse-dbb-shapeIou-1/weights/last.pt'

save_model_path = os.path.split(origin_model_path)[0] + '/converted.pt'
device = torch.device("cpu")
ckpt = torch.load(origin_model_path, map_location='cpu')

model = ckpt['model']
idx = 0
del model.model[idx]  # Silence
idx += 22
for _ in range(15):
	del model.model[idx]  # multi-level reversible auxiliary branch
# Detect head
if hasattr(model.model[idx], 'cv2'):
	del model.model[idx].cv2
if hasattr(model.model[idx], 'cv3'):
	del model.model[idx].cv3
if hasattr(model.model[idx], 'dfl'):
	del model.model[idx].dfl

# f
save = []
for mi, m in enumerate(model.model):
	m.i = mi
	if isinstance(m.f, list):
		for i, f in enumerate(m.f):
			if f != -1:
				m.f[i] -= 1
				save.append(m.f[i])
model.save = sorted(save)

m_ckpt = {
	'model': model.half(),
	'optimizer': None,
	'best_fitness': None,
	'ema': None,
	'updates': None,
	'opt': None,
	'git': None,
	'date': None,
	'epoch': -1}
torch.save(m_ckpt, save_model_path)
