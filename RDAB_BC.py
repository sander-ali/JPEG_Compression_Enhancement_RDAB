import os.path
import torch
import cv2
from utils import utils_image as util

testset_name = 'Mytestset'
show_img = False                 # default: False
testsets = 'testsets'    
results = 'results'     

H_path = os.path.join(testsets, testset_name)
E_path = results   # E_path, for Estimated images
util.mkdir(E_path)

model_path = 'model_zoo/RDAB_BC.pth'
if os.path.exists(model_path):
    print(f'loading model from {model_path}')
else:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.rdab_network import RDAB as net
model = net(in_nc=3, out_nc=3, nc=[64,128,256,512], nb=4, act_mode='R')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

H_paths = util.get_image_paths(H_path)
for idx, img in enumerate(H_paths):
    img_name, ext = os.path.splitext(os.path.basename(img))
    img_L = util.imread_uint(img, n_channels=3)
    img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR) 
    _, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_L = cv2.imdecode(encimg, 3)
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)               
    img_L = util.uint2tensor4(img_L)
    img_L = img_L.to(device)
    img_E,QF = model(img_L)
    QF = 1 - QF
    img_E = util.tensor2single(img_E)
    img_E = util.single2uint(img_E)
    img_H = util.imread_uint(H_paths[idx], n_channels=3).squeeze()

    util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

