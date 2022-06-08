import torch
import os
import cv2
import yaml
from time import time

from utils.metrics import bbox_iou
from yaml.loader import SafeLoader
from argparse import ArgumentParser

annotate_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Path
ROOT = os.getcwd()
data = ''
weights = ''
dist = ''
imgsz = 0

def parse_opt():
    global data, weights, dist, imgsz
    parser = ArgumentParser()
    parser.add_argument('--weights', type=str, help='model path(s)')
    parser.add_argument('--data', type=str)
    parser.add_argument('--dist', type=str)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w')
    opt = parser.parse_args()
    data = opt.data
    weights = opt.weights
    dist = opt.dist
    imgsz = opt.imgsz

parse_opt()

# Read YAML
with open(data) as f:
    data = yaml.load(f, Loader=SafeLoader)

# Load the model
model = torch.hub.load(os.getcwd(), 'custom', source='local', path=weights)
model.conf = 0.4
model.iou = 0.5
os.chdir(data['path'])

def annotate(img, xywh, c, p, iou):
    color = annotate_colors[int(c) % len(annotate_colors)]
    # Image, list of X Y W H, class index, confidence probability, IOU
    x = int(xywh[0])
    y = int(xywh[1])
    w = int(xywh[2] / 2)
    h = int(xywh[3] / 2)

    pt1 = (x - w, y - h)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, color, 2)

    text = data['names'][int(c)] + "  " + str(round(float(p), 2)) + "  " + str(round(iou, 2))
    (w2, h2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    pt3 = (x - w + w2, y - h - h2)
    cv2.rectangle(img, pt1, pt3, color, -1)
    cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


def run():
    for img in os.listdir(data['test']):
        p_time = time()
        name = img.split(".")[0]

        # Read Image
        img_d = cv2.imread(data['test'] + "/" + img)
        img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)

        # Get ground truth values
        truth = torch.tensor([list(map(float, x.split())) for x in open("./labels/test/" + name + ".txt").readlines()])

        # Predict
        result = model(img_d, size=imgsz)

        for r in result.xywh[0]:
            iou = float(max(bbox_iou(r[None, :4], truth[:, 1:] * imgsz)))
            *xywh, p, c = r
            annotate(img_d, xywh, c, p, iou)

        e_time = time() - p_time;
        cv2.putText(img_d, "Time:" + str(round(e_time,4)) + "s", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        # Save
        cv2.imwrite(ROOT + "/" + dist + img, img_d[:, :, ::-1])


run()
