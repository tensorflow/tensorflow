#!/bin/bash
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

source onednn_benchmark_config.sh

# Store models in home directory
mkdir -p ${TF_GRAPHS}
cd ${TF_GRAPHS}

# Download TF graphs linked from MLPerf Inference v2.0
# https://github.com/mlcommons/inference#mlperf-inference-v20-submission-02252022
wget -c https://zenodo.org/record/2535873/files/resnet50_v1.pb -O resnet50_v1-5.pb
wget -c https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
wget -c https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz -O mobilenet-v1.tgz  # 90MB
wget -c http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz -O ssd-mobilenet-v1.tar.gz
wget -c https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip -O ssd-resnet34.zip
wget -c https://zenodo.org/record/3939747/files/model.pb -O bert-large.pb  # 1.2GB. Could take 10+ minutes to download.

# Extract graphs from tar/zip files.
unzip ./inception5h.zip -d inception
mv inception/tensorflow_inception_graph.pb inception.pb
rm -rf inception
tar -xzf mobilenet-v1.tgz ./mobilenet_v1_1.0_224_frozen.pb
mv mobilenet_v1_1.0_224_frozen.pb mobilenet-v1.pb
tar -xzf ssd-mobilenet-v1.tar.gz ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
mv ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb ssd-mobilenet-v1.pb
rm -rf ssd_mobilenet_v1_coco_2018_01_28
unzip ssd-resnet34.zip
mv tf_ssd_resnet34_22.1/resnet34_tf.22.1.pb  ssd-resnet34.pb
rm -rf tf_ssd_resnet34_22.1

# Now we should have the following model files in ~/tf-graphs:
# - bert-large.pb
# - inception.pb
# - mobilenet-v1.pb
# - resnet50_v1-5.pb
# - ssd-mobilenet-v1.pb
# - ssd-resnet34.pb

