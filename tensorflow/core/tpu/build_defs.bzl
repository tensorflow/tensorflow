# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""A helper to include the TF C API implementations or only headers depending on whether building Tensorflow or LibTPU."""

load("//tensorflow:tensorflow.bzl", "if_libtpu")

# Returns the appropriate TF_Status C API def based on whether building LibTPU or Tensorflow.
def if_libtpu_tf_status():
    return if_libtpu(
        if_true = [Label("//tensorflow/c:tf_status_headers")],
        if_false = [Label("//tensorflow/c:tf_status")],
    )

# Returns the appropriate TF_Tensor C API def based on whether building LibTPU or Tensorflow.
def if_libtpu_tf_tensor():
    return if_libtpu(
        if_true = [Label("//tensorflow/c:tf_tensor_hdrs")],
        if_false = [Label("//tensorflow/c:tf_tensor")],
    )

# Returns the C API headers or full implementation based on whether building LibTPU or Tensorflow.
def if_libtpu_tf_c_api():
    return if_libtpu(
        if_true = [Label("//tensorflow/c:c_api_headers")],
        if_false = [Label("//tensorflow/c:c_api")],
    )
