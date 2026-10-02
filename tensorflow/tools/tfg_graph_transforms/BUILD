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

# Description:
#   Utilities to perform MLIR TFG graph transformations.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//tensorflow/cc/saved_model/image_format:internal_api",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

TFG_GRAPH_TRANSFORM_DEPS = [
    ":utils",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:Transforms",
    "//tensorflow/compiler/mlir:init_mlir",
    "//tensorflow/compiler/mlir/tensorflow",
    "//tensorflow/compiler/mlir/tensorflow:error_util",
    "//tensorflow/core:lib",
    "//tensorflow/core/ir:Dialect",
    "//tensorflow/core/protobuf:for_core_protos_cc",
    "//tensorflow/core/transforms:PassRegistration",
    "//tensorflow/compiler/tf2xla/ops:xla_ops",
    "//tensorflow/core:ops",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/ir/importexport:graphdef_export",
    "//tensorflow/core/ir/importexport:graphdef_import",
    "//tensorflow/core/ir/importexport:savedmodel_export",
    "//tensorflow/core/ir/importexport:savedmodel_import",
    "//tensorflow/core/ir:tf_op_registry",
]

# Description:
#   A tool that provides a mechanism to run TFG graph optimizations operating on the
#   GraphDef from the supplied SavedModel as an input.
tf_cc_binary(
    name = "tfg_graph_transforms",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    deps = TFG_GRAPH_TRANSFORM_DEPS + [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:Support",
    ],
)

# Description:
#    The tool wrapped into a library, so that it could be used
#    when custom defined ops and passes need to be added.
cc_library(
    name = "tfg_graph_transforms_main",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    visibility = ["//visibility:public"],
    deps = TFG_GRAPH_TRANSFORM_DEPS + [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:Support",
    ],
)
