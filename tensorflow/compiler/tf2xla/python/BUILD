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

load("//tensorflow:tensorflow.default.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_py_clif_cc",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

tf_py_clif_cc(
    name = "xla_op_registry",
    srcs = ["xla_op_registry.clif"],
    pyclif_deps = [
        "//tensorflow/core/framework:kernel_def_pyclif",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
    ],
)

tf_custom_op_py_library(
    name = "xla",
    srcs = ["xla.py"],
    dso = ["//tensorflow/compiler/tf2xla/ops:_xla_ops.so"],
    kernels = [
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/python/ops:random_ops_util",
        "//tensorflow/python/ops/numpy_ops:np_utils",
        "@xla//xla:xla_data_proto_py",
    ],
)
