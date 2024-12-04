#!/usr/bin/env bash
#
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
#
# setup.cuda.sh: Clean up and prepare the CUDA installation on the container.
# TODO(@perfinion) Review this file

# Delete uneccessary static libraries
find /usr/local/cuda-*/lib*/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete

# Link the compat driver to the location where tensorflow is searching for it
ln -s /usr/local/cuda-12.3/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
