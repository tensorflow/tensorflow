#!/usr/bin/env bash
# Copyright 2015 Google Inc. All Rights Reserved.
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

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ ! -d ${CUDA_HOME}/lib64 ]; then
  echo "Failed to locate CUDA libs at ${CUDA_HOME}/lib64."
  exit 1
fi

export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | \
                    xargs -I{} echo '-v {}:{}')
export DEVICES=$(\ls /dev/nvidia* | \
                    xargs -I{} echo '--device {}:{}')

if [[ "${DEVICES}" = "" ]]; then
  echo "Failed to locate NVidia device(s). Did you want the non-GPU container?"
  exit 1
fi

docker run -it $CUDA_SO $DEVICES "$@"
