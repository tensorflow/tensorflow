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
    default_visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
)

cc_library(
    name = "lite_rt_c_api",
    hdrs = [
        "lite_rt_common.h",
        "lite_rt_model.h",
        "lite_rt_op_code.h",
        "lite_rt_support.h",
    ],
    deps = [
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite/core/c:c_api_types",
    ],
)

cc_library(
    name = "lite_rt_tensor_buffer_c_api",
    hdrs = [
        "lite_rt_event.h",
        "lite_rt_tensor_buffer.h",
        "lite_rt_tensor_buffer_requirements.h",
    ],
    deps = [
        ":lite_rt_c_api",
    ],
)

cc_library(
    name = "lite_rt_dispatch_c_api",
    hdrs = [
        "lite_rt_dispatch.h",
        "lite_rt_dispatch_api.h",
    ],
    deps = [
        ":lite_rt_c_api",
        ":lite_rt_tensor_buffer_c_api",
    ],
)

exports_files(srcs = glob(["lite_rt*.h"]))
