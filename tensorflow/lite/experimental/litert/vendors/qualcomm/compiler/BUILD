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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
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
    # This test can be run only on Android and Linux.
    target_compatible_with = select({
        "@platforms//os:android": [],
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    use_sys_malloc = True,
    deps = [
        ":qnn_compiler_plugin",  # buildcleaner: keep
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers_oss",
        "//tensorflow/lite/experimental/litert/test:test_models",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR:qnn_tensor",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations:quantize_op_legalization",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_op_code",
        "//tensorflow/lite/experimental/litert/c:litert_options",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_model_predicates",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/tools:dump",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:common",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core:tensor_pool",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:cast_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:concatenation_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:conv2d_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:depthwise_conv2d_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:dynamic_update_slice_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:elementwise_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:embedding_lookup_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:fully_connected_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:fully_connected_op_builder_htp",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:gather_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:gelu_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:hard_swish_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:leaky_relu_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:matmul_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:mean_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:pack_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:pool2d_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:quantize_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:reduce_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:relu_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:reshape_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:resize_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:rms_norm_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:select_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:slice_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:softmax_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:spatial_transform_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:split_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:tanh_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders:transpose_op_builder",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
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
