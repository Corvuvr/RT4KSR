import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Tuple
from torch.nn.functional import interpolate


import data
import model
from PIL import Image
from utils import utils, metrics, parser, image as im


def load_checkpoint(model, device, time_stamp=None):       
    checkpoint = glob.glob(os.path.join("code/checkpoints", time_stamp + ".pth"))
    if isinstance(checkpoint, List):
        checkpoint = checkpoint.pop(0)
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
 

def reparameterize(config, net, device, save_rep_checkpoint=False):
    config.is_train = False
    rep_model = torch.nn.DataParallel(model.__dict__[config.arch](config)).to(device)
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = net.state_dict()
    
    for k, v in rep_state_dict.items():            
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]
            
            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]
            
            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0
        
            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).cuda()
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0
            
            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()
            
        elif "rep_conv.bias" in k:
            pass
            
        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    if save_rep_checkpoint:
        torch.save(rep_state_dict, f"rep_model_{config.checkpoint_id}.pth")
        
    return rep_model


def test(config):
    """
    SETUP METRICS
    """
    test_results = OrderedDict()
    test_results["psnr_rgb"] = []
    test_results["psnr_y"] = []
    test_results["ssim_rgb"] = []
    test_results["ssim_y"] = []
    # test_results = test_results

    """
    SETUP MODEL
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(
        model.__dict__[config.arch](config)
        ).to(device)
    net = load_checkpoint(net, device, config.checkpoint_id)
    
    if config.rep:
        rep_model = reparameterize()
        net = rep_model

    net.eval()

    import time
    import json
    import cv2
    from copy import deepcopy
    from pathlib import Path

    JSON_FOLDER = f"results/{Path(args.dataroot).stem}"
    JSON_PATH = f"{JSON_FOLDER}/data.json"
    Path(JSON_FOLDER).mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)

    with torch.no_grad():
        images = []
        transform = transforms.Compose([
            # transforms.PILToTensor()
            transforms.ToTensor()
        ])

        for filepath in Path(f"./{args.dataroot}/").rglob("*"):
            if filepath.suffix.lower() != '.png':
                continue
            
            image = cv2.imread(filepath)
            hr_img = deepcopy(image)
            lr_img = deepcopy(image)
            
            # if bool(image.shape[0] % 2) or bool(image.shape[1] % 2):
                
            shp = list(int(np.ceil(x/args.scale) // 2 * 2) for x in image.shape) # 210/2 = 104 ??

            lr_img = cv2.resize(lr_img, (shp[1], shp[0]))
            scale_ok: bool = (tuple(hr_img.shape[0:1])==tuple(x*2 for x in lr_img.shape[0:1]))
            print(f"HR: {hr_img.shape}\t LR: {lr_img.shape}\t{'OK' if (scale_ok) else 'BAD'}")

            lr_img = transform(lr_img).to(torch.float32)
            hr_img = transform(hr_img).to(torch.float32)
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            
            batch = lr_img.unsqueeze(0)
            start = time.time()
            out = net(batch)
            time_elapsed = time.time() - start

            diagnostic_data = [time_elapsed, *shp]
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                df = json.load(f)
                df.append(diagnostic_data)
            with open(JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(df, f, ensure_ascii=False, indent=4)

            out *= 255.
            #out = np.array(out)
            out = im.tensor2uint(out)
            
            cv2.imwrite(f"{JSON_FOLDER}/{filepath.stem}{args.suffix}.png", out)
            
            

if __name__ == "__main__":
    args = parser.base_parser()
    
    test(args)