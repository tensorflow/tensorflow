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

# Description:
# JavaScript/TypeScript code generation for TensorFlow.js

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

visibility = [
    "//tensorflow:internal",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

cc_library(
    name = "ts_op_gen",
    srcs = [
        "ops/ts_op_gen.cc",
    ],
    hdrs = [
        "ops/ts_op_gen.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "ts_op_gen_test",
    srcs = [
        "ops/ts_op_gen.cc",
        "ops/ts_op_gen.h",
        "ops/ts_op_gen_test.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
