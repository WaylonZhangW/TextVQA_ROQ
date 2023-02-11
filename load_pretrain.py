import torch
import collections
import numpy as np

path = '/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_spatial_val.npy'
imdb = np.load(path, allow_pickle=True)

print(type(imdb))


assert False

path = '/public/home/zhangwei1/TQD/pre-train/finetuned/textvqa_tap_ocrcc_best.ckpt'
# path = '/public/home/zhangwei1/TQD/pre-train/finetuned/textvqa_tap_ocrcc_best.ckpt'
print(path)
model = torch.load(path,map_location=torch.device('cpu'))['model']

print(type(model))
print(model.keys())

new_dict ={}
for k,v in model.items():
    if (k[:3] == 'mmt') or ("ocr_ptr_net" in k) or ('cls' in k) or ('classifier' in k):
        continue
    else:
        new_dict[k] = v

del model
print('---'*100)
print('new dict',new_dict.keys())

assert False

mmt_4_layers = {}

for key in model.keys():
    if key[:4] == "mmt." :
        mmt_4_layers[key] = model[key]


new_dict ={k.replace("mmt.encoder.","") : v for k,v in mmt_4_layers.items() if k[:12] == 'mmt.encoder.' }
print(new_dict.keys())


# dir = "./TAP_ocrcc_best_mmt12.pth"
# torch.save(new_dict,dir)
