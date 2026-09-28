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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt:__subpackages__",
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_native_lowering_stub",
    srcs = ["tfrt_native_lowering_stub.cc"],
    hdrs = ["tfrt_native_lowering_stub.h"],
    deps = [
        "//tensorflow/core/tfrt/graph_executor:executable_context",
        "//tensorflow/core/tfrt/graph_executor:sync_resource_state",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "model_config_stub",
    srcs = ["model_config_stub.cc"],
    hdrs = ["model_config_stub.h"],
    deps = [
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model:saved_model_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)
