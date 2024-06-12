# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2024/3/8 13:26
  @version V1.0
"""
import torch
import os
from models.yolo import Model

cfg = "./models/detect/gelan-cn.yaml"
origin_model_path = './runs-experiment/train/v9-cn-double-ImplicitConvCBFuse-shapeIou/weights/last.pt'

save_model_path = os.path.split(origin_model_path)[0] + '/converted.pt'
device = torch.device("cpu")
model = Model(cfg, ch=3, nc=9, anchors=3)
#model = model.half()
model = model.to(device)
_ = model.eval()
ckpt = torch.load(origin_model_path, map_location='cpu')
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

idx = 0
for k, v in model.state_dict().items():
	if "model.{}.".format(idx) in k:
		if idx < 22:
			kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
		elif "model.{}.cv2.".format(idx) in k:
			kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 16))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
		elif "model.{}.cv3.".format(idx) in k:
			kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 16))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
		elif "model.{}.dfl.".format(idx) in k:
			kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 16))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
	else:
		while True:
			idx += 1
			if "model.{}.".format(idx) in k:
				break
		if idx < 22:
			kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
		elif "model.{}.cv2.".format(idx) in k:
			kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 16))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
		elif "model.{}.cv3.".format(idx) in k:
			kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 16))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
		elif "model.{}.dfl.".format(idx) in k:
			kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 16))
			model.state_dict()[k] -= model.state_dict()[k]
			model.state_dict()[k] += ckpt['model'].state_dict()[kr]
_ = model.eval()

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
