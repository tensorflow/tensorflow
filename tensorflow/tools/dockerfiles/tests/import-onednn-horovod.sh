#!/usr/bin/env bash

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ============================================================================

python -c 'from tensorflow.python import _pywrap_util_port; print(_pywrap_util_port.IsMklEnabled()); import horovod.tensorflow as hvd'
new_mkl_horovod_enabled=$?

python -c 'from tensorflow.python import pywrap_tensorflow; print(pywrap_tensorflow.IsMklEnabled()); import horovod.tensorflow as hvd'
old_mkl_horovod_enabled=$?

if [[ $new_mkl_horovod_enabled -eq 0 ]]; then
   echo "PASS: Horovod with MKL is enabled"
elif [[ $old_mkl_horovod_enabled -eq 0]]; then
   echo "PASS: Horovod with Old MKL is detected"
else
   die "FAIL: Horovod with MKL is not enabled"
fi
