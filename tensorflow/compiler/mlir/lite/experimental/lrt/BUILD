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

cc_binary(
    name = "apply_plugin",
    srcs = [
        "apply_plugin.cc",
        # TODO: b/366821557 - Support pre-compiled plugins as data dependencies.
        "//tensorflow/compiler/mlir/lite/experimental/lrt/examples:mul_op_plugin_so",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:algo",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:api_internal",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:lite_rt_model_init",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:model",
        "//tensorflow/lite/schema:schema_fbs",
        "@llvm-project//llvm:Support",
    ],
)
