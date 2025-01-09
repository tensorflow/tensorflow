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

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_lib")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_lib(
    name = "legalization",
    hdrs = ["legalization.h"],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
    ],
)

litert_lib(
    name = "add_op_legalization",
    srcs = ["add_op_legalization.cc"],
    hdrs = ["add_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "batch_matmul_op_legalization",
    srcs = ["batch_matmul_op_legalization.cc"],
    hdrs = ["batch_matmul_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "cast_op_legalization",
    srcs = ["cast_op_legalization.cc"],
    hdrs = ["cast_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "concatenation_op_legalization",
    srcs = ["concatenation_op_legalization.cc"],
    hdrs = ["concatenation_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "cos_op_legalization",
    srcs = ["cos_op_legalization.cc"],
    hdrs = ["cos_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "div_op_legalization",
    srcs = ["div_op_legalization.cc"],
    hdrs = ["div_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "embedding_lookup_op_legalization",
    srcs = ["embedding_lookup_op_legalization.cc"],
    hdrs = ["embedding_lookup_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "transpose_op_legalization",
    srcs = ["transpose_op_legalization.cc"],
    hdrs = ["transpose_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "fully_connected_op_legalization",
    srcs = ["fully_connected_op_legalization.cc"],
    hdrs = ["fully_connected_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "gelu_op_legalization",
    srcs = ["gelu_op_legalization.cc"],
    hdrs = ["gelu_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "greater_op_legalization",
    srcs = ["greater_op_legalization.cc"],
    hdrs = ["greater_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "less_op_legalization",
    srcs = ["less_op_legalization.cc"],
    hdrs = ["less_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "logical_and_op_legalization",
    srcs = ["logical_and_op_legalization.cc"],
    hdrs = ["logical_and_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "mul_op_legalization",
    srcs = ["mul_op_legalization.cc"],
    hdrs = ["mul_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "reshape_op_legalization",
    srcs = ["reshape_op_legalization.cc"],
    hdrs = ["reshape_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "rsqrt_op_legalization",
    srcs = ["rsqrt_op_legalization.cc"],
    hdrs = ["rsqrt_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "sin_op_legalization",
    srcs = ["sin_op_legalization.cc"],
    hdrs = ["sin_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "select_op_legalization",
    srcs = ["select_op_legalization.cc"],
    hdrs = ["select_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "slice_op_legalization",
    srcs = ["slice_op_legalization.cc"],
    hdrs = ["slice_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "sum_op_legalization",
    srcs = ["sum_op_legalization.cc"],
    hdrs = ["sum_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "sub_op_legalization",
    srcs = ["sub_op_legalization.cc"],
    hdrs = ["sub_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "softmax_op_legalization",
    srcs = ["softmax_op_legalization.cc"],
    hdrs = ["softmax_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "tanh_op_legalization",
    srcs = ["tanh_op_legalization.cc"],
    hdrs = ["tanh_op_legalization.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        ":util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)

litert_lib(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":legalization",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/tools:dump",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)
