# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/mlir/lite/experimental/lrt:__subpackages__"],
)

cc_library(
    name = "lite_rt_model_api",
    hdrs = ["lite_rt_model_api.h"],
    deps = [
        "//tensorflow/compiler/mlir/lite/experimental/lrt:lite_rt_common",
        "//tensorflow/compiler/mlir/lite/experimental/lrt:lite_rt_op_code",
        "//tensorflow/lite/core/c:c_api_types",
    ],
)

cc_library(
    name = "graph_tools",
    hdrs = [
        "graph_tools.h",
    ],
    deps = [
        ":lite_rt_model_api",
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/lrt:lite_rt_common",
        "//tensorflow/compiler/mlir/lite/experimental/lrt:lite_rt_op_code",
        "//tensorflow/compiler/mlir/lite/experimental/lrt:lite_rt_support",
        "//tensorflow/lite/c:c_api_types",
        "@llvm-project//llvm:Support",
    ],
)
