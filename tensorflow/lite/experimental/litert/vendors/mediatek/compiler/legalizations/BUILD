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
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/mediatek/compiler:__subpackages__"],
)

cc_library(
    name = "operand_map",
    srcs = ["operand_map.cc"],
    hdrs = ["operand_map.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "neuron_utils",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "neuron_utils",
    srcs = ["neuron_utils.cc"],
    hdrs = ["neuron_utils.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "add_op_legalization",
    srcs = ["add_op_legalization.cc"],
    hdrs = ["add_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "mul_op_legalization",
    srcs = ["mul_op_legalization.cc"],
    hdrs = ["mul_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "batch_matmul_op_legalization",
    srcs = ["batch_matmul_op_legalization.cc"],
    hdrs = ["batch_matmul_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "fully_connected_op_legalization",
    srcs = ["fully_connected_op_legalization.cc"],
    hdrs = ["fully_connected_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "reshape_op_legalization",
    srcs = ["reshape_op_legalization.cc"],
    hdrs = ["reshape_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "transpose_op_legalization",
    srcs = ["transpose_op_legalization.cc"],
    hdrs = ["transpose_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "rsqrt_op_legalization",
    srcs = ["rsqrt_op_legalization.cc"],
    hdrs = ["rsqrt_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "concat_op_legalization",
    srcs = ["concat_op_legalization.cc"],
    hdrs = ["concat_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "quantize_op_legalization",
    srcs = ["quantize_op_legalization.cc"],
    hdrs = ["quantize_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "common_op_legalization",
    srcs = ["common_op_legalization.cc"],
    hdrs = ["common_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "sub_op_legalization",
    srcs = ["sub_op_legalization.cc"],
    hdrs = ["sub_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "softmax_op_legalization",
    srcs = ["softmax_op_legalization.cc"],
    hdrs = ["softmax_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "mean_op_legalization",
    srcs = ["mean_op_legalization.cc"],
    hdrs = ["mean_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)

cc_library(
    name = "gelu_op_legalization",
    srcs = ["gelu_op_legalization.cc"],
    hdrs = ["gelu_op_legalization.h"],
    tags = [
        # Don't build/test in OS until MediaTek SDK is available.
        "nobuilder",
        "notap",
    ],
    deps = [
        "operand_map",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/mediatek:neuron_adapter_api",
    ],
)
