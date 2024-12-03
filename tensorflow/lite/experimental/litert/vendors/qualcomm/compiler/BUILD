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

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_dynamic_lib", "litert_lib", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

litert_dynamic_lib(
    name = "qnn_compiler_plugin",
    srcs = ["qnn_compiler_plugin.cc"],
    hdrs = ["//tensorflow/lite/experimental/litert/vendors/c:litert_compiler_plugin.h"],
    export_litert_only = True,
    shared_lib_name = "qnn_compiler_plugin_so",
    so_name = "libLiteRtCompilerPlugin_Qualcomm.so",
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    ungrte = True,
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        ":qnn_compose_graph",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
    ],
)

litert_test(
    name = "qnn_compiler_plugin_test",
    srcs = [
        "qnn_compiler_plugin_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/test:mlir_test_data",
        "//tensorflow/lite/experimental/litert/test:tflite_test_data",
    ],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    use_sys_malloc = True,
    deps = [
        ":qnn_compiler_plugin",  # buildcleaner: keep
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:test_macros",
        "//tensorflow/lite/experimental/litert/test:test_models",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
    ],
)

litert_lib(
    name = "qnn_compose_graph",
    srcs = ["qnn_compose_graph.cc"],
    hdrs = ["qnn_compose_graph.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        ":graph_mapper",
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
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:add_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:batch_matmul_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:cast_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:concatenation_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:cos_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:div_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:fully_connected_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:mul_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:reshape_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:rsqrt_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:select_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:sin_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:slice_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:softmax_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:sub_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:sum_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:tanh_op_legalization",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:transpose_op_legalization",
    ],
)

litert_lib(
    name = "graph_mapper",
    srcs = [
        "graph_mapper.cc",
    ],
    hdrs = ["graph_mapper.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)
