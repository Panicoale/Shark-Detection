# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training executable for detection models.

This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:

    python3 train.py --logtostderr --train_dir=training  --pipeline_config_path=training/FASTER_RCNN_RESNET101_COCO_eval.config


    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""

# import functools
# import json
import os
import time
from shutil import copyfile



# import tensorflow as tf

# from object_detection.builders import dataset_builder
# from object_detection.builders import graph_rewriter_builder
# from object_detection.builders import model_builder
# from object_detection.legacy import trainer
# from object_detection.utils import config_util

dir_path = os.path.dirname(os.path.realpath(__file__))
input_path='training'
output_path='training_eval'

input_path=os.path.join(dir_path,input_path)
output_path=os.path.join(dir_path,output_path)

Number=[]
FILES=os.listdir(input_path)
for element in FILES :
    #print(element[0:11])
    if element[0:11]=='model.ckpt-':
        #print(element)
        if 'data-00000-of-00001' in element:
            
            element2=element.replace('.data-00000-of-00001','')
            
            Number.append(int(element2[11:]))
print(Number)
Number.sort(reverse = False)
print(Number)

model_checkpoint= "model.ckpt-"
variable = 'model_checkpoint_path: "' 
for element in Number:
    element=str(element)
    for file in FILES:
        if element in file:
            copyfile(os.path.join(input_path,file), os.path.join(output_path,file))
        model_checkpoint_element= str(model_checkpoint)+element
        String=variable+model_checkpoint+element+'"'
        f=open(os.path.join(output_path,'checkpoint'),"w+")
        f.write(String)
        f.close()

    time.sleep(350) 

