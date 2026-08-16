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

# XLA Ops for TPUs

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "host_compute_ops",
    srcs = [
        "host_compute_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:mlir_xla_op_kernel",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:if_op",
        "//tensorflow/compiler/tf2xla/kernels:index_ops",
        "//tensorflow/compiler/tf2xla/kernels:while_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu/kernels:cross_replica_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla:shape_util",
        "@xla//xla:side_effect_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:sharding_builder",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_ops",
    srcs = [
        "get_item_op.cc",
        "index_ops.cc",
        "infeed_op.cc",
        "inplace_ops.cc",
        "outfeed_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":host_compute_ops",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_context",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:if_op",
        "//tensorflow/compiler/tf2xla/kernels:index_ops",
        "//tensorflow/compiler/tf2xla/kernels:while_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu/kernels:cross_replica_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/tpu:c_api_conversions",
        "@xla//xla/tpu:c_api_decl",
        "@xla//xla/tpu:tpu_executor_api",
        "@xla//xla/tsl/platform:logging",
    ],
    alwayslink = 1,
)
