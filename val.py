# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
import sys


sys.path.append('./pdq_evaluation')
from read_files import convert_coco_det_to_rvc_det
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.yolo import Model
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, intersect_dicts,clip_coords,cov,is_pos_semidef,get_near_psd)
from utils.metrics import  ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from ruamel.yaml import YAML
yaml=YAML()


def change_dropout_rate(m, perc):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout') or each_module.__class__.__name__.startswith('GaussianDropout') or each_module.__class__.__name__.startswith('DropBlock2d'):
            each_module.p = perc

def change_second_layer(m,cfg,perc):
    for each_module in m.modules():
        if cfg=='yolov5s-dg.yaml':
            if each_module.__class__.__name__.startswith('GaussianDropout'):
                each_module.p = perc
        if cfg=='yolov5s-db.yaml':
            if each_module.__class__.__name__.startswith('DropBlock2d'):
                each_module.p = perc         
        if cfg=='yolov5s-gd.yaml':
            if each_module.__class__.__name__.startswith('DropBlock2d'):
                each_module.p = perc

def change_nums_sample(num,cfg):
        if cfg =='yolov5s-dropout.yaml':
            with open('yolov5s-dropout.yaml') as f:
                doc=yaml.load(f)
                doc['num_samples'] =num
            with open('yolov5s-dropout.yaml','w') as f:
                yaml.dump(doc,f)

        if cfg =='yolov5s-gdropout.yaml':
            with open('yolov5s-gdropout.yaml') as f:
                doc=yaml.load(f)
                doc['num_samples'] =num
            with open('yolov5s-gdropout.yaml','w') as f:
                yaml.dump(doc,f)

        if cfg =='yolov5s-dropblock.yaml':
            with open('yolov5s-dropblock.yaml') as f:
                doc=yaml.load(f)
                doc['num_samples'] =num
            with open('yolov5s-dropblock.yaml','w') as f:
                yaml.dump(doc,f)
         

def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        num_samples=1,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        corruption_num=None,
        severity=None,
        new_drop_rate=None,
        second_drop_rate=None,
        cfg=None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
 #       save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        change_nums_sample(num_samples,cfg) 

        model = Model(cfg).to(device)
        ckpt = torch.load(weights, map_location='cpu')
          # create
        exclude = []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        stride=max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.float()
        if half:
            model.half()
        if new_drop_rate is not None:
            print('Changing default dropout rate...')
            change_dropout_rate(m=model, perc=new_drop_rate)
        if second_drop_rate is not None:
            print('Changing second dropout rate')
            change_second_layer(m=model,cfg=cfg,perc=second_drop_rate)

        # Data
        data = check_dataset(data)  # check

    # Configure
    old_time = time.time()
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train() #enable dropout
    
    for m in model.modules():
        if m.__class__.__name__.startswith('DropBlock2d'):
            m.train() #enable dropout

    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else True  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        if corruption_num is not None:
            print(f'Dataloader will have corrupted images with number {corruption_num} and severity {severity}')
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=rect,
                                       workers=workers, prefix=colorstr(f'{task}: '),corruption_num=corruption_num,severity=severity)[0]

    seen = 0

    names = ['item'] if single_cls and len(data['names']) != 1 else data['names']
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):

        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad():
            t = time_sync()
            if num_samples ==1:
                inf_out, _ = model(im, augment=augment)
            else:
                infs_all, _ = model(im, augment=augment)
                inf_mean = torch.mean(torch.stack(infs_all), dim=0)
                infs_all.insert(0, inf_mean)
                inf_out = torch.cat(infs_all, dim=2)

            t0 += time_sync() - t

                # Loss
            t = time_sync()
            output, all_scores, sampled_coords = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres,
                                                                      multi_label=True,
                                                                      max_width=width, max_height=height)
            t1 += time_sync() - t

        for si, pred in enumerate(output):
             labels = targets[targets[:, 0] == si, 1:]
             nl = len(labels)
             tcls = labels[:, 0].tolist() if nl else []  # target class
             seen += 1

             if pred is None:
                  if nl:
                       stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                  continue



                # Clip boxes to image bounds
             clip_coords(pred, (height, width))

                # Append to pycocotools JSON dictionary
             if save_json:

                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(Path(paths[si]).stem.split('_')[-1])
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(im[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                    # Getting covariances
                    # The transformations to coordinates follow the ones that are done below here after the if clause
                    if num_samples > 1:
                        # output: BS(list) x NUM_DETECTIONS x 6
                        # sampled_coords : BS(list) x NUM_DETECTIONS x NUM_SAMPLES x 4
                        # sampled_boxes : NUM_DETECTIONS x NUM_SAMPLES x 4
                        sampled_boxes = xywh2xyxy(sampled_coords[si].reshape(-1, 4)).reshape(sampled_coords[si].shape)
                        clip_coords(sampled_boxes.reshape(-1, 4), (height, width))

                        scale_coords(im[si].shape[1:], sampled_boxes.reshape(-1, 4), shapes[si][0], shapes[si][1])

                        # It will have 2 covariances matrices of 2X2 for each one of the two xy coordinates
                        covar_batch = torch.zeros(sampled_boxes.shape[0], 2, 2, 2)
                        for det_id in range(sampled_boxes.shape[0]):
                            covar_batch[det_id, 0, ...] = cov(sampled_boxes[det_id, :, :2])
                            covar_batch[det_id, 1, ...] = cov(sampled_boxes[det_id, :, 2:])

                        # Rounding it for smaller size
                        covar_batch = np.around(covar_batch.numpy(), 5).tolist()
                    else:
                        # Just dummy covars for the json zip() down below
                        covar_batch = [None] * pred.shape[0]

                    for p, b, p_all, covar_xyxy in zip(pred.tolist(), box.tolist(), all_scores[si].tolist(),
                                                       covar_batch):
                        if covar_xyxy is not None:
                            # Covariances need to be positive semi-definite, so just transform them here already
                            for i, covar_tmp in enumerate(covar_xyxy):
                                covar_tmp = np.array(covar_tmp)
                                if not is_pos_semidef(covar_tmp):
                                    print('Warning: Converted covar to near PSD')
                                    covar_xyxy[i] = get_near_psd(covar_tmp).tolist()

                        jdict.append({'image_id': image_id,
                                      'category_id': class_map[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5),
                                      'all_scores': [round(x, 5) for x in p_all],
                                      'covars': covar_xyxy})

             # Assign all predictions as incorrect
             correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
             if nl:
                 detected = []  # target indices
                 tcls_tensor = labels[:, 0]

                 # target boxes
                 tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                 # Per target class
                 for cls in torch.unique(tcls_tensor):
                     ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                     pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                     # Search for detections
                     if pi.shape[0]:
                         # Prediction to target ious
                         ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                         # Append detections
                         for j in (ious > iouv[0]).nonzero():
                             d = ti[i[j]]  # detected target
                             if d not in detected:
                                 detected.append(d)
                                 correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                 if len(detected) == nl:  # all targets already located in image
                                     break

             # Append statistics (correct, conf, pcls, tcls)
             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    #    if batch_i < 3:
    #        f = save_dir / f'test_batch{batch_i}_labels.jpg'
    #        plot_images(im, targets, paths=paths, names=names, fname=f, max_subplots=batch_size)  # ground truth
    #        f = save_dir / f'test_batch{batch_i}_pred.jpg'
    #        plot_images(im, output_to_target(output, width, height), paths=paths, names=names, fname=f, max_subplots=batch_size)  # predictions


            # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

            # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    current_time = time.time()
    print('time:'+str(current_time-old_time))
            # Print results per class
    if verbose and nc > 1 and len(stats):
           for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
    if verbose or save_json:
            t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        # Save JSON
    if save_json and len(jdict):
            pred_json = str(save_dir / f"dets_{name}_{conf_thres}_{iou_thres}.json")
            with open(pred_json, 'w') as file:
                json.dump(jdict, file)

            '''
            No need for this part as it will be evaluated later
            try:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval
                # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                cocoGt = COCO(glob.glob(data['instances_path'])[0])  # initialize COCO ground truth api
                cocoDt = cocoGt.loadRes(f'output/dets_{name}_{conf_thres}_{iou_thres}.json')  # initialize COCO pred api
                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(e)
                print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                      'See https://github.com/cocodataset/cocoapi/issues/356')
            '''
            del jdict
            print('Converting to RVC1 format...')
            convert_coco_det_to_rvc_det(det_filename=save_dir/f'dets_{name}_{conf_thres}_{iou_thres}.json',
                                       gt_filename='../datasets/coco/annotations/instances_val2017.json',
                                      save_filename=save_dir/f'dets_converted_{name}_{conf_thres}_{iou_thres}.json')


        # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
            maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps,

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='best.pt', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s-dropout.yaml', help='model.yaml path')
    parser.add_argument('--num_samples', type=int, default=10, help='How many times to sample if doing MC-Dropout')
    parser.add_argument('--corruption_num', type=int, help='which corruption number to use from imagecorruptions')
    parser.add_argument('--new_drop_rate', type=float, help='change the dropout rate of Dropout layers')
    parser.add_argument('--second_drop_rate', type=float, help='change the dropout rate of the second Dropout layers')
    parser.add_argument('--severity', type=int, help='which severity to use for the corruption in --corruption_num')



    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
