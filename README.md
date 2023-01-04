# Object Detection in an Urban Environment

## Project Overview
Object detection is one of the most powerful autonomous driving technologies that can be done with cameras, which are low-cost sensors.

- Download waymo dataset and try to EDA.  
- Training & evaluation. (Detect Vehicle, Pedestrian, Cyclist in Waymo datasets)  
- Improve the performances.
- Video Inference.

<br><br>

## Set up
### 1. Download dataset

```sh
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

### 2. Split downloaded dataset to train & validation & test

```sh
python create_splits.py --data-dir /home/workspace/data
```

### 3. EDA
The analyzed images are in the `/EDA_images` folder.

![image](https://user-images.githubusercontent.com/54730375/210225311-4d1dba00-3c2f-4f75-be4a-177736d8ae7b.png)


### 4. Edit the config file

 Download pretrained model and move it to /home/workspace/experiments/pretrained_model/.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:

```sh
python edit_config.py --train_dir /your_workspace/data/train/ \
--eval_dir /your_workspace/data/val/ \
--batch_size 2 \
--checkpoint /your_workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 \
--label_map /your_workspace/experiments/label_map.pbtxt
```
A new config file has been created, pipeline_new.config.  
move pipeline_new.config to `/reference/`


### 5. Training
- training process:
```sh
python experiments/model_main_tf2.py --model_dir=experiments/reference/ \
 --pipeline_config_path=experiments/reference/pipeline_new.config
```

- evaluation process:
```sh
python experiments/model_main_tf2.py --model_dir=experiments/reference/ \
--pipeline_config_path=experiments/reference/pipeline_new.config \
--checkpoint_dir=experiments/reference/
```

Note: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using CTRL+C.

To monitor the training, you can launch a tensorboard instance by running python -m tensorboard.main --logdir experiments/reference/. You will report your findings in the writeup.

<br><br>

## Dataset
- Dataset analysis results are in "/EDA_images"
- Cross validation
  - train: 80
  - validation: 10
  - test: 10
  - total: 100


<br><br>

## Training Monitoring(Before improving the performance)

![image](https://user-images.githubusercontent.com/54730375/210224751-16721b2a-5fa7-479a-ae7c-98f3e0b283f2.png)

![image](https://user-images.githubusercontent.com/54730375/210224803-c3679e9e-fd7a-465f-ae89-9a9efa5b1f22.png)

- training loss: 3.0
- evaluation loss: 4.8


<br><br>

## Improve the Performances
- dataset 100 -> 300
  - train: 270
  - validation: 15
  - test: 15
- reason
  - I thought there were too few datasets.(too large loss value)

- Hyper parameters control
  - lr: 0.04 -> 0.03999999910593033
  - warmup_learnig_rate: 0.013333 -> 0.013333000242710114
  - box prediction l2_weight: 0.0004 -> 0.000397
  - conv_hyperparams l2_regularizer_weight: 0.0004 -> 0.00037
- reason
![image](https://user-images.githubusercontent.com/54730375/210499957-d85d314c-7994-4d69-9acc-1e5b54dc7517.png)

It was confirmed that the regularization loss decreases and then increases significantly from a certain point.  
Due to this phenomenon, we decided to reduce the regularization loss weight value.  

<br>

Other parameters have been verified to be rounded automatically when a new config file is created.  
I just eliminated the phenomenon of rounding.  

- Increase learning steps: 25000 -> 35000
- reason  
  - I thought 25000 times Iterations were too few.
  
![image](https://user-images.githubusercontent.com/54730375/210225352-cfcd5583-8244-48a4-8afe-06ca9b1bdc63.png)

As the learning progresses, the gap between evaluation loss and train loss does not widen, indicating that overfitting did not occur.  

- train loss: 1
- eval_loss: 1.5

## Video inference
![animation (1)](https://user-images.githubusercontent.com/54730375/210226556-462f86f1-61d5-485a-bd23-4df002e03e10.gif)
