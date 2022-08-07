# stochastic-yolov5
Install
------- 
Activate the conda environment.
```
conda create -n env python=3.9
conda activate env
```

It is highly recommended to install PyTorch under the official website https://pytorch.org/.

Clone the repo and install requirements.txt in the root directory.
```
pip install -r requirements.txt
```

By far, all the packages are installed including cocoapi, in our conda environment. To calculate the PDQ score using PDQ_evaluation, the cocoapi needs to make some adjustments. Instructions there: https://github.com/tjiagoM/pdq_evaluation

Note: cocoapi is installed in conda's virtual environment: /anaconda3/envs/env/lib/python3.9/site-packages/pycocotools

You will also require code for using LRP evaluation measures. To do this you need to simply copy the cocoevalLRP.py file from the LRP github repository to the pycocotools folder within the PythonAPI. You can download the specific file here https://github.com/cancam/LRP/blob/master/cocoLRPapi-master/PythonAPI/pycocotools/cocoevalLRP.py

After cocoevalLRP.py is located in your pycocotools folder, adjust the system path on line 10 of coco_LRP.py, line 11 of coco_mAP.py and line 16 of read_file.py to match your PythonAPI folder.

Usage
------- 
We are using the weight provided by YOLOv5:
YOLOv5s https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt
YOLOv5x https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x.pt
You need to rename the weight name to best.pt, if you want to use the YOLOv5x or other weights, you need to change the depth_multiple and width_multiple from yolov5s-dropblock.yaml, yolov5s-dropout.yaml and yolov5s-gdropout.yaml.

Run the model
```
python val.py --cfg yolov5s-dropout.yaml --batch 4 --data coco.yaml --imgsz 640 --iou-thres 0.6 --num_samples 10 --conf-thres 0.5 --new_drop_rate 0.1 --corruption_num 7 --severity 2
```
Optional flags for model changes include ```--cfg```, ```num_samples```,```new_drop_rate```,```corruption_num```and```severity```.
```--cfg``` defines which sampling layer we are using. Options are 'yolov5s-dropout.yaml' for dropout, 'yolov5s-gdropout.yaml' for Gaussian dropout and 'yolov5s-dropblock.yaml' for DropBlock.

```num_samples``` defines the number of samples of the model.
```new_drop_rate``` defines the sampling rate of the model.

```corruption_num```and ```severity```define the type and the severity of the corruption. More detail: https://github.com/bethgelab/imagecorruptions

The model will generate a json file in ```dets_converted_exp_0.5_0.6.json ```

Evaluate PDQ and mAP
```
python pdq_evaluation/evaluate.py --test_set coco --gt_loc ../datasets/coco/annotations/instances_val2017.json --det_loc dets_converted_exp_0.5_0.6.json --save_folder output --num_workers 16
```
For each evaluation, result will be saved in ```output/scores.txt```.
##Hyperparameter Search
For each evaluation, we set the population size to 50 and run one epoch per run because PDQ evaluation is quite time-consuming. The checkpoint is saved as ```checkpoint.npy```. If there is a checkpoint exists, it will directly start from the checkpoint. So it is necessary to clear the previous checkpoint to
Run the baseline evaluation
```
python nsga2.py
```
Run the FPS evaluation
```
python nsga2_fps.py
```
Run the evaluations under different weather conditions
```
python nsga2_snow.py
python nsga2_fog.py
python nsga2_frost.py
```
All the checkpoints are saved in ```checkpoint``` Folder. To check the detailed results, please check https://pymoo.org/interface/result.html.
 
