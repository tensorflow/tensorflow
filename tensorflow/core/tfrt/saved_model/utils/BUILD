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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/core/tfrt/saved_model/...",
        "//tensorflow/compiler/mlir/tensorflow/...",
        "//learning/brain/tfrt/cpp_tests/gpu_inference/...",
        "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/compiler/mlir/tfrt/translate/...",
    ],
)

cc_library(
    name = "serialize_utils",
    srcs = ["serialize_utils.cc"],
    hdrs = ["serialize_utils.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@tf_runtime//:bef",
        "@tsl//tsl/platform:env",
    ],
)

tf_cc_shared_test(
    name = "serialize_utils_test",
    srcs = ["serialize_utils_test.cc"],
    data = [
        "//tensorflow/compiler/mlir/tfrt/tests/saved_model:testdata",
    ],
    tags = ["no_oss"],
    deps = [
        ":serialize_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt/transforms/mlrt:import_model",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/core/tfrt/saved_model:saved_model_util",
        "//tensorflow/core/tfrt/utils",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:bef",
        "@tsl//tsl/platform:env",
        "@xla//xla/tsl/lib/core:status_test_util",
    ],
)
