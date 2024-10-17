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
    name = "lite_rt_cc_api",
    hdrs = [
        "lite_rt_model.h",
        "lite_rt_support.h",
        "//tensorflow/lite/experimental/lrt/vendors/c:lite_rt_compiler_plugin.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/core:model_builder_base",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lite_rt_tensor_buffer_cc_api",
    hdrs = [
        "lite_rt_handle.h",
        "lite_rt_tensor_buffer.h",
        "lite_rt_tensor_buffer_requirements.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_tensor_buffer_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "lite_rt_tensor",
    srcs = ["lite_rt_tensor.cc"],
    hdrs = ["lite_rt_tensor.h"],
    deps = [
        ":lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "lite_rt_tensor_test",
    srcs = ["lite_rt_tensor_test.cc"],
    data = [
        "//tensorflow/lite/experimental/lrt/test:tflite_test_data",
    ],
    deps = [
        ":lite_rt_tensor",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "lite_rt_op",
    srcs = ["lite_rt_op.cc"],
    hdrs = ["lite_rt_op.h"],
    deps = ["//tensorflow/lite/experimental/lrt/c:lite_rt_c_api"],
)

cc_test(
    name = "lite_rt_op_test",
    srcs = ["lite_rt_op_test.cc"],
    data = [
        "//tensorflow/lite/experimental/lrt/test:tflite_test_data",
    ],
    deps = [
        ":lite_rt_op",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/test:common",
        "@com_google_googletest//:gtest_main",
    ],
)

exports_files(srcs = glob(["lite_rt_*.h"]))
