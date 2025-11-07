#!/bin/bash

## if there is any error installing segment anything 2
## check nvcc version and g++ version compatibility.


## to run inference. 
python inference_robot.py  \
  --version YxZhang/evf-sam2 \
  --precision='fp16' \
  --model_type sam2   \
  --prompt "dummy"


# python inference_robot.py  \
#   --version YxZhang/evf-sam-multitask \
#   --precision='fp16' \
#   --model_type sam2   \
#   --prompt "dummy"