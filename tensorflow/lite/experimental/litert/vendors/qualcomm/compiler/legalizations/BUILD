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
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/cc:litert_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
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
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "//tensorflow/lite/experimental/litert/cc:litert_op",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
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
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "//tensorflow/lite/experimental/litert/cc:litert_op",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor",
        "//tensorflow/lite/experimental/litert/core:graph_tools",
        "//tensorflow/lite/experimental/litert/tools:dump",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:graph_mapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)
