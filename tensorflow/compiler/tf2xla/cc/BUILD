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

load("//tensorflow:tensorflow.default.bzl", "tf_gen_op_wrapper_cc")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/tf2xla:friends"],
    licenses = ["notice"],
)

tf_gen_op_wrapper_cc(
    name = "xla_ops_gen",
    out_ops_file = "ops/xla_ops",
    deps = ["//tensorflow/compiler/tf2xla/ops:xla_ops"],
)

cc_library(
    name = "xla_ops",
    srcs = ["ops/xla_ops.cc"],
    hdrs = ["ops/xla_ops.h"],
    deps = [
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_gen_op_wrapper_cc(
    name = "xla_jit_op_gen",
    include_internal_ops = 1,
    out_ops_file = "ops/xla_jit_ops",
    deps = ["//tensorflow/compiler/jit/ops:xla_ops"],
)

cc_library(
    name = "xla_jit_ops",
    srcs = ["ops/xla_jit_ops.cc"],
    hdrs = ["ops/xla_jit_ops.h"],
    deps = [
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)
