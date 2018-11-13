# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# SavedModel contrib libraries.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "saved_model_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ] + glob(
        ["python/saved_model/*.py"],
        exclude = ["python/saved_model/*_test.py"],
    ),
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":keras_saved_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:util",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_library(
    name = "reader",
    srcs = ["python/saved_model/reader.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    visibility = ["//visibility:public"],
    deps = [
        ":saved_model_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/saved_model:builder",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_test(
    name = "reader_test",
    size = "small",
    srcs = ["python/saved_model/reader_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    visibility = ["//visibility:private"],
    deps = [
        ":reader",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "signature_def_utils_test",
    size = "small",
    srcs = ["python/saved_model/signature_def_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saved_model_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/saved_model:signature_constants",
        "//tensorflow/python/saved_model:signature_def_utils",
        "//tensorflow/python/saved_model:utils",
    ],
)

py_library(
    name = "keras_saved_model",
    srcs = ["python/saved_model/keras_saved_model.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:metrics",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver",
        "//tensorflow/python:util",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:export",
        "//tensorflow/python/estimator:keras",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/saved_model",
    ],
)

py_test(
    name = "keras_saved_model_test",
    size = "medium",
    srcs = ["python/saved_model/keras_saved_model_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_saved_model",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
