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

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.cc"],
    data = [
        "testdata/test.mlir",
        "testdata/xla_launch.mlir",
        "testdata/xla_launch_xla_reduce_window.mlir",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:saved_model",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:bef",
        "@tf_runtime//:hostcontext",
        "@xla//xla/tsl/platform:statusor",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(
        ["testdata/**"],
    ),
    visibility = ["//visibility:public"],
)
