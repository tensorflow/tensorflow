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
package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "core_layers",
    srcs = ["python/layers/core_layers.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:layers",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "layers",
    srcs = ["python/layers/layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core_layers",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "layers_test",
    size = "small",
    srcs = ["python/layers/layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "learning",
    srcs = ["python/learning.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "rnn_cells",
    srcs = ["python/layers/rnn_cells.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":core_layers",
    ],
)

py_library(
    name = "pruning_utils",
    srcs = ["python/pruning_utils.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "pruning",
    srcs = ["python/pruning.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":core_layers",
        ":pruning_utils",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "pruning_utils_test",
    size = "small",
    srcs = ["python/pruning_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pruning_utils",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "pruning_test",
    size = "small",
    srcs = ["python/pruning_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pruning",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "rnn_cells_test",
    size = "small",
    srcs = ["python/layers/rnn_cells_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pruning",
        ":rnn_cells",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "init_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
)

# Top-level library
py_library(
    name = "model_pruning",
    srcs_version = "PY2AND3",
    deps = [
        ":init_py",
        ":layers",
        ":learning",
        ":pruning",
        ":rnn_cells",
    ],
)
