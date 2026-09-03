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

load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # Custom driver is unsupported in OSS. Fails if one is provided.
    # copybara:uncomment driver = "//tensorflow/compiler/mlir:run_lit.sh",
    # copybara:comment_begin(JitRt/Auto fusion depreciated)
    exclude = [
        "auto-fusion.mlir",
        "tf_to_corert_pipeline_cpurt.mlir",
        "outline-cpurt-cluster.mlir",
    ],
    # copybara:comment_end
    features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
    size_override = {
        "fallback.mlir": "medium",
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
    ],
)
