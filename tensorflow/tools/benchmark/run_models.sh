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

# Run models downloaded with download_models.sh with the TF benchmark tool.
# This script must be called from its current directory.
set -x
source onednn_benchmark_config.sh
date

# Navigate to the workspace root directory and configure build configurations. 
configure_build
pwd
date

for ONEDNN in 0 1; do
  # Build the benchmark tool. Can add code to handle different build configs
  # when TF-oneDNN is disabled/enabled.
  build_benchmark_tool ${ONEDNN}
  date

  for BATCH in 1 16 64; do
    # Print information for parse_onednn_benchmarks.py
    echo "BATCH=${BATCH}, ONEDNN=${ONEDNN}"
    export TF_ENABLE_ONEDNN_OPTS=${ONEDNN}

    # Run each graph.
    date
    ${BENCH} \
      --graph=${TF_GRAPHS}/resnet50_v1-5.pb \
      --input_layer="input_tensor:0" \
      --input_layer_shape="${BATCH},224,224,3" \
      --input_layer_type="float" \
      --output_layer="softmax_tensor:0"
    date
    ${BENCH} \
      --graph=${TF_GRAPHS}/inception.pb \
      --input_layer="input:0" \
      --input_layer_shape="${BATCH},224,224,3" \
      --input_layer_type="float" \
      --output_layer="output:0"
    date
    ${BENCH} \
      --graph=${TF_GRAPHS}/mobilenet-v1.pb \
      --input_layer="input:0" \
      --input_layer_shape="${BATCH},224,224,3" \
      --input_layer_type="float" \
      --output_layer="MobilenetV1/Predictions/Reshape_1:0"
    date
    ${BENCH} \
      --graph=${TF_GRAPHS}/ssd-mobilenet-v1.pb \
      --input_layer="image_tensor:0" \
      --input_layer_shape="${BATCH},300,300,3" \
      --input_layer_type="uint8" \
      --output_layer="detection_classes:0"
    date
    ${BENCH} \
      --graph=${TF_GRAPHS}/ssd-resnet34.pb \
      --input_layer="image:0" \
      --input_layer_shape="${BATCH},3,1200,1200" \
      --input_layer_type="float" \
      --output_layer="detection_classes:0"
    date
    # Only run BERT with batch size 1 for now.
    if [[ $BATCH == 1 ]]; then
      ${BENCH} \
        --graph=${TF_GRAPHS}/bert-large.pb \
        --input_layer="input_ids:0,input_mask:0,segment_ids:0" \
        --input_layer_shape="1,384:1,384:1,384" \
        --input_layer_type="int32,int32,int32" \
        --output_layer="logits:0"
    fi
  done
done
