#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
#matplotlib.use('Agg')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
#from fast_rcnn.nms_wrapper import nms
from nms.py_cpu_nms import py_cpu_nms
from utils.timer import Timer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'AnPan', 'AppleWerm', 'BaconnCheese', 'ButterSugarLoaf', 'CheesySausage', 
           'ChickenParmesan', 'CranberryCreamCheese', 'CurryChicken', 'DoubleUp', 'FireFlosss', 
           'Flosss', 'GarlicButterBaguette', 'GoldenNachoCheese', 'HamnCheese', 
           'HamnCheeseCroissant', 'MoshiMushroom', 'MushroomChezSausage', 'RaisinCreamCheese', 'RedBeanQueen',
           'SausageStandard', 'SeaweedFlosss', 'SnowWhiteYam', 'StandardSausage', 'SugarDonut', 
           'Sunflower', 'SweetPotatoVolcano', 'TunaBun', 'YamRoyale')

NETS = { 'vgg16': ('VGG16', 'vgg16_faster_rcnn_iter_%d000.caffemodel') }
ax = None 
np.set_printoptions(precision=2,suppress=True)

def vis_detections(im, ax, dets, thresh=0.4, animated=False):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, 4] >= thresh)[0]
    if len(inds) == 0:
        return
    visible = not animated
    
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, 4]
        cls_ind = dets[i, 5].astype(np.int)
        class_name = CLASSES[cls_ind]
        
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2, animated=animated, visible=visible)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=12, color='white', animated=animated, visible=visible)

def vis_detections_animate(i):
    global ax
    ax.patches[i].set_visible(True)
    ax.texts[i].set_visible(True)
    return ax.patches + ax.texts
    
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    global ax
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3

    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(im, aspect='equal')
    animated = False
    all_dets = []
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_inds = np.ones( (len(cls_scores), 1) ) * cls_ind
        # col 4 and col -1 are both cls_scores, to satisfy nms()
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis], cls_inds, 
                            cls_scores[:, np.newaxis])).astype(np.float32)
        all_dets.append(dets)
    
    all_dets = np.concatenate(all_dets)
    conf_inds = all_dets[:, 4] >= CONF_THRESH
    conf_dets = all_dets[conf_inds]
    keep = py_cpu_nms(conf_dets, NMS_THRESH) # nms(conf_dets, NMS_THRESH) #
    nms_dets = conf_dets[keep, :]
    vis_detections(im, ax, nms_dets, thresh=CONF_THRESH, animated=animated)

    plt.axis('off')
    plt.tight_layout()
    plt.title(image_name)
    
    if animated:
        ani = FuncAnimation(fig, vis_detections_animate, 
                            frames=xrange(len(ax.patches)), interval=500, blit=True)
        ani.save('/tmp/animation.gif', writer='imagemagick', fps=1)
    else:                        
        plt.draw()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--nid', dest='net_id', help='Network ID to use [10]',
                        default=10, type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    prototxt = os.path.join(cfg.ROOT_DIR, "models", "bread", 'test.prototxt')
    outputDir = "/home/shaohua/py-faster-rcnn/output/faster_rcnn_end2end/trainmult"
    modelname = NETS[args.demo_net][1] % args.net_id
    caffemodel = os.path.join(outputDir, modelname)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = [ '8.jpg', '9.jpg', '10.jpg', '11.jpg', 'sidebyside.jpg', 'sidebyside2.jpg',
                 '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', 
                 '6.jpg', '7.jpg'
               ]
    # difficult images
    im_ids = [ 7, 12, 13, 29, 34 ]
    #for im_name in im_names:
    for im_seq in im_ids:
        im_name = "20170913_%d.jpg" %im_seq
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
