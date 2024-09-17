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
    name = "lite_rt_cc_api",
    hdrs = [
        "lite_rt_support.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/c:c_api_types",
    ],
)

exports_files(srcs = glob(["lite_rt*.h"]))
