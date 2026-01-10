import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from visionrt import Camera

cam1 = Camera("/dev/video0")
cam2 = Camera("/dev/video2")


def preprocess(frame: torch.Tensor):
    return frame.unsqueeze(0)


model = (
    nn.Sequential(
        nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        nn.Softmax(dim=1),
    )
    .cuda()
    .eval()
)
labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

print_camera_predictions = lambda out1, out2: print(
    f"{'Camera 1:':<12}{labels[out1.argmax(dim=1).item()]:<25} \
    | {'Camera 2:':<12}{labels[out2.argmax(dim=1).item()]:<25}"
)

### api usage

import torq as tq
from torq import config

config.verbose = True

tq.register(cls=Camera, adapter=lambda x: next(x.stream()))

system = tq.Sequential(
    tq.Concurrent(
        tq.Sequential(cam1, preprocess, model),
        tq.Sequential(cam2, preprocess, model),
    ),
    print_camera_predictions,
)

system = tq.compile(system)
system.run()

print(system)

###

cam1.close()
cam2.close()
