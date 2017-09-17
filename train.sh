rm data/cache/trainmult_gt_roidb.pkl
python tools/train_net.py --solver models/bread/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb bread_trainmult --cfg experiments/cfgs/faster_rcnn_end2end.yml --iters 40000
