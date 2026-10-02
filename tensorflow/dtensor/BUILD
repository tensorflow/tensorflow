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

# DTensor is a TensorFlow extension that enables large-scale modeling with
# minimal changes to user code.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

package_group(
    name = "dtensor-users",
    packages = [
        "//tensorflow_models/google/...",
        "//third_party/aimee/clara2_labs/...",
        "//third_party/py/jax_tpu_embedding/...",
        "//third_party/py/tf_keras/dtensor/...",
    ],
)

package_group(
    name = "dtensor-internal",
    packages = [
        "//learning/brain/experimental/jax_tpu_embedding/...",
        "//learning/brain/public/...",
        "//tensorflow/core/...",
        "//tensorflow/dtensor/...",
        "//tensorflow/python/...",
        "//tensorflow_federated/...",  #  Needs access to C++ API for dtensor device
    ],
)
