#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
set -e

# Build the docker image
cd tensorflow/tools/ci_build
docker build -t horovod_test_container:latest -f Dockerfile.horovod.gpu .

docker run --rm \
  --gpus all \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  horovod_test_container:latest bash -c "python3.7 -m pytest"

