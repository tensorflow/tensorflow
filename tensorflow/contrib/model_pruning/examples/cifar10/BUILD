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
# Description:
# Example TensorFlow models for CIFAR-10

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "cifar10_input",
    srcs = ["cifar10_input.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "cifar10_pruning",
    srcs = ["cifar10_pruning.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar10_input",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/model_pruning:pruning",
    ],
)

py_binary(
    name = "cifar10_eval",
    srcs = [
        "cifar10_eval.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar10_pruning",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "cifar10_train",
    srcs = [
        "cifar10_train.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cifar10_pruning",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/model_pruning:pruning",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
