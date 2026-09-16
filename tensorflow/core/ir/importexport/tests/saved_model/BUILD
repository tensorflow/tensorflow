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

# Description: Tests for SavedModel model roundtrip via MLIR TFG module

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "saved_model_test_files",
    srcs = [
        "savedmodel_v1/saved_model.pb",
        "savedmodel_v2/saved_model.pb",
    ],
)

tf_cc_test(
    name = "saved_model_roundtrip_test",
    srcs = ["saved_model_roundtrip_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    deps = [
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:graph_debug_info_proto_cc",
        "//tensorflow/core/ir/importexport:savedmodel_export",
        "//tensorflow/core/ir/importexport:savedmodel_import",
        "//tensorflow/core/ir/importexport/tests/roundtrip",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_googletest//:gtest_main",
    ],
)
