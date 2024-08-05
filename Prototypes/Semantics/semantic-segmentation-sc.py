import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchvision.transforms as transforms
import torchvision

width = height = 2048
device = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(Encoder, self).__init__()

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_gap = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Atrous Spatial Pyramid Pooling
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.conv3x3_5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[3], dilation=rates[3])
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        # Concatenation layer
        self.conv1x1_out = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        # Global average pooling
        gap = self.gap(x)
        gap = self.conv1x1_gap(gap)
        gap = F.interpolate(gap, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Atrous Spatial Pyramid Pooling
        conv1x1_1 = self.conv1x1_1(x)
        conv3x3_2 = self.conv3x3_2(x)
        conv3x3_3 = self.conv3x3_3(x)
        conv3x3_4 = self.conv3x3_4(x)
        conv3x3_5 = self.conv3x3_5(x)
        pooled = self.pooling(x)
        pooled = F.interpolate(pooled, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenation
        Encoder_out = torch.cat([gap, conv1x1_1, conv3x3_2, conv3x3_3, conv3x3_4, conv3x3_5, pooled], dim=1)

        return Encoder_out
class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder, self).__init__()

        # 1x1 convolution to adjust the number of channels
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        # 3x3 convolution
        self.conv3x3 = nn.Conv2d(32, mid_channels, kernel_size=3, padding=1)

        # Upsample by 4
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Final 1x1 convolution
        self.final_conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x, encoder_output):
        # 1x1 convolution
        x = self.conv1x1(x)

        # Concatenate with the encoder output
        x_concatenated = torch.cat([x, encoder_output], dim=1)

        # 3x3 convolution
        x = self.conv3x3(x_concatenated)

        # Final 1x1 convolution
        x = self.final_conv1x1(x)

        # Upsample by 4 again
        x = self.upsample4(x)

        return x
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()

        # Backbone (ResNet101 in this example)
        self.backbone = models.resnet101(pretrained=False)
        
        # Encoder module
        self.encoder = Encoder(in_channels=height, out_channels=num_classes, rates=[6, 12, 18, 24]) 
        
        # Decoder
        self.decoder = Decoder(in_channels=height, mid_channels=4, out_channels=num_classes)

    def forward(self, x):
        # Backbone - ResNet
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Encoder module
        encoder_output = self.encoder(x)
        
        # Forward pass in your model
        x = self.decoder(x, encoder_output)

        return x

prot = input("Which prototype would you like to load:")
if int(prot) >= 13:
    width = height = 2048
    transformImg = transforms.Compose([
        transforms.ToPILImage(),
        transforms.transforms.RandomCrop(height, padding=None, pad_if_needed=True, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    model = DeepLabV3Plus(num_classes=4)
    model = model.to(device).eval()

elif int(prot) >= 6 and int(prot) <13:
    transformImg = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=7).eval()

elif int(prot) < 6:
    transformImg = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2).eval()

model.load_state_dict(torch.load("Models/Deeplab/DCNN-Prot-" + prot + ".torch"), strict=False)

img_path = r"C:\Users\rough\OneDrive\Desktop\Coding\BTYSTE-2024\Datasets\ADE20K_2021_17_01\images\ADE\training\nature_landscape\boardwalk\ADE_train_00004331.jpg"


# Define the helper function
def decode_segmap(image, label_colors):
    nc = len(label_colors)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def segment(model=model, img_path=img_path, show_orig=True, dev="cpu"):
    # ... (rest of your code)

    # Use the loaded labels in the decode_segmap function
    if int(prot) < 6:
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=forestry
                                 (128, 0, 0)])
    elif int(prot) >= 6 and int(prot) < 13:
        label_colors = np.array([(0, 0, 0),     # 0=background
                                 (128, 0, 0),   # 1=forestry
                                 (0, 128, 0),   # 2=label2
                                 (0, 0, 128),   # 3=label3
                                 (128, 128, 0), # 4=label4
                                 (128, 0, 128), # 5=label5
                                 (0, 128, 128)  # 6=label6
                                 ])
    elif int(prot) >= 13:
        label_colors = np.array([(0, 0, 0),     # 0=background
                                 (128, 0, 0),   # 1=forestry
                                 (0, 128, 0),   # 2=label2
                                 (0, 0, 128),   # 3=label3
                                 ])
        
    img = cv2.imread(img_path)

    img = transformImg(img).unsqueeze(0)

    img = torch.autograd.Variable(img, requires_grad=False)

    # Forward pass through the model
    output = model(img)['out'][0].to(dev)
    mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(mask, label_colors)

    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

segment(model, img_path)
