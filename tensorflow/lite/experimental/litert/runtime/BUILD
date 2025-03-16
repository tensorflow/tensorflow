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

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "gtest_main_no_heapcheck_deps")
load("//tensorflow/lite/experimental/litert/build_common:special_rule.bzl", "gles_deps", "gles_linkopts", "lite_rt_friends")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "event",
    srcs = [
        "event.cc",
    ],
    hdrs = [
        "event.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
    ],
)

cc_library(
    name = "tensor_buffer",
    srcs = [
        "ahwb_buffer.cc",
        "dmabuf_buffer.cc",
        "fastrpc_buffer.cc",
        "gl_buffer.cc",
        "gl_texture.cc",
        "ion_buffer.cc",
        "open_cl_buffer.cc",
        "tensor_buffer.cc",
    ],
    hdrs = [
        "ahwb_buffer.h",
        "dmabuf_buffer.h",
        "event.h",
        "fastrpc_buffer.h",
        "gl_buffer.h",
        "gl_texture.h",
        "ion_buffer.h",
        "open_cl_buffer.h",
        "tensor_buffer.h",
        "tensor_buffer_requirements.h",
        "//tensorflow/lite/experimental/litert/c:litert_event.h",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer.h",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer_requirements.h",
    ],
    linkopts = gles_linkopts(),
    deps = [
        ":environment",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_layout",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_event",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/core/util:tensor_type_util",
        "//tensorflow/lite/experimental/litert/runtime/opencl:buffer",
        "//tensorflow/lite/experimental/litert/runtime/opencl:cl_command_queue",
        "//tensorflow/lite/experimental/litert/runtime/opencl:cl_context",
        "//tensorflow/lite/experimental/litert/runtime/opencl:opencl_wrapper",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@opencl_headers",
    ] + gles_deps() + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
            "//tensorflow/lite/delegates/gpu/gl:gl_texture",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "environment",
    srcs = [
        "environment.cc",
    ],
    hdrs = [
        "environment.h",
    ],
    visibility = [
        "//third_party/odml/infra/ml_drift_delegate:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_any",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/core:environment",
        "//tensorflow/lite/experimental/litert/runtime/opencl:cl_command_queue",
        "//tensorflow/lite/experimental/litert/runtime/opencl:cl_context",
        "//tensorflow/lite/experimental/litert/runtime/opencl:cl_device",
        "//tensorflow/lite/experimental/litert/runtime/opencl:opencl_wrapper",
        "@opencl_headers",
    ],
)

cc_test(
    name = "environment_test",
    srcs = ["environment_test.cc"],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":environment",
        "@com_google_googletest//:gtest_main",
        # copybara:uncomment_begin(google-only)
        # "//third_party/ml_drift/cl:environment",
        # "//third_party/ml_drift/cl:opencl_wrapper",
        # copybara:uncomment_end
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_any",
        "//tensorflow/lite/experimental/litert/runtime/opencl:opencl_wrapper",
    ],
)

cc_library(
    name = "tfl_utils",
    srcs = [
        "tfl_utils.cc",
    ],
    hdrs = [
        "tfl_utils.h",
    ],
    deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_detail",
        "//tensorflow/lite/experimental/litert/cc:litert_element_type",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
    ],
)

cc_library(
    name = "external_litert_buffer_context",
    srcs = ["external_litert_buffer_context.cc"],
    hdrs = ["external_litert_buffer_context.h"],
    visibility = [
        "//tensorflow/lite/experimental/litert:__subpackages__",
    ] + lite_rt_friends(),
    deps = [
        ":tfl_utils",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer_requirements",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "compiled_model",
    srcs = ["compiled_model.cc"],
    hdrs = ["compiled_model.h"],
    deps = [
        ":external_litert_buffer_context",
        ":tensor_buffer",
        "//tensorflow/compiler/mlir/lite:allocation",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:model_builder",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/experimental/litert/c:litert_accelerator_options",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model_options",
        "//tensorflow/lite/experimental/litert/c:litert_dispatch_delegate",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_buffer_ref",
        "//tensorflow/lite/experimental/litert/cc:litert_event",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer_requirements",
        "//tensorflow/lite/experimental/litert/compiler/plugin:compiler_plugin",
        "//tensorflow/lite/experimental/litert/core:accelerator_model_compilation_data",
        "//tensorflow/lite/experimental/litert/core:build_stamp",
        "//tensorflow/lite/experimental/litert/core:environment",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/runtime/dispatch:dispatch_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "compiled_model_test",
    srcs = ["compiled_model_test.cc"],
    data = [
        "//tensorflow/lite/experimental/litert/test:testdata/simple_model.tflite",
    ],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    # require GPU to run OpenCL tests.
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":compiled_model",
        ":tensor_buffer",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_opaque",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_compiled_model_options",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/c:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "//tensorflow/lite/experimental/litert/test:simple_model",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gl_buffer_test",
    srcs = ["gl_buffer_test.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-landroid",
        ],
        "//conditions:default": [],
    }),
    tags = [
        "notap",
    ],
    deps = [
        ":tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/test:matchers",
    ] + gtest_main_no_heapcheck_deps() + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu/gl:egl_environment",
            "//tensorflow/lite/delegates/gpu/gl:gl_buffer",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensor_buffer_conversion",
    srcs = ["tensor_buffer_conversion.cc"],
    hdrs = ["tensor_buffer_conversion.h"],
    linkopts = gles_linkopts(),
    deps = [
        ":tensor_buffer",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_expected",
        "//tensorflow/lite/experimental/litert/cc:litert_macros",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer_utils",
        "@com_google_absl//absl/strings:str_format",
    ] + gles_deps(),
)

cc_test(
    name = "tensor_buffer_conversion_test",
    srcs = ["tensor_buffer_conversion_test.cc"],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = [
        "notap",
    ],
    deps = [
        ":tensor_buffer",
        ":tensor_buffer_conversion",
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/c:litert_environment",
        "//tensorflow/lite/experimental/litert/c:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_layout",
        "//tensorflow/lite/experimental/litert/core:environment",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu/gl:egl_environment",
        ],
        "//conditions:default": [],
    }),
)
