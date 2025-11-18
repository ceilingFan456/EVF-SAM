#!/bin/bash

## if there is any error installing segment anything 2
## check nvcc version and g++ version compatibility.


## to run inference. 
# python inference_robot.py  \
#   --version YxZhang/evf-sam2 \
#   --precision='fp16' \
#   --model_type sam2   \
#   --prompt "dummy"


python inference_robot.py  \
  --version YxZhang/evf-sam2-multitask \
  --precision='fp16' \
  --model_type sam2   \
  --prompt "dummy"


python inference_robot.py  \
  --version YxZhang/evf-sam2-multitask \
  --precision='fp16' \
  --model_type sam2   \
  --prompt "dummy" \
  --vis_save_path "/home/t-qimhuang/disk/robot_dataset/final_test_set/roboengine_test_video_evf-sam" \
  --image_path "/home/t-qimhuang/disk/robot_dataset/final_test_set/roboengine_test_video"


## if wanna use sam model, need to refer to evf_sam.py caz its sam scale is using huge. and it is not implemented for sam 2. 