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

load("//tensorflow:tensorflow.bzl", "if_oss", "tf_cc_test")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    exclude = ["testdata/**"],
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        # copybara:uncomment "@llvm-project//mlir:run_lit.sh",
        "//tensorflow/compiler/mlir:tf-mlir-translate",
        "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
    ],
)

tf_cc_test(
    name = "update_op_cost_in_tfrt_mlir_test",
    srcs = ["update_op_cost_in_tfrt_mlir_test.cc"],
    data = [
        "testdata/test.mlir",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt:transforms/update_op_cost_in_tfrt_mlir",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@tf_runtime//:init_tfrt_dialects",
    ],
)
