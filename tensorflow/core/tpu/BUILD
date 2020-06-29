# Description: Utilities for TPU Operations

load(
    "//tensorflow:tensorflow.bzl",
    "if_windows",
)

package(
    default_visibility = [
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/stream_executor/tpu:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "libtftpu_header",
    hdrs = ["libtftpu.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "tpu_embedding_optimization_parameters_utils",
    srcs = ["tpu_embedding_optimization_parameters_utils.cc"],
    hdrs = ["tpu_embedding_optimization_parameters_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "tpu_embedding_output_layout_utils",
    srcs = ["tpu_embedding_output_layout_utils.cc"],
    hdrs = ["tpu_embedding_output_layout_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_output_layout_proto_cc",
    ],
)

cc_library(
    name = "tpu_compilation_device",
    srcs = ["tpu_compilation_device.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_defs",
        ":tpu_node_device_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_node_device_util",
    srcs = ["tpu_node_device_util.cc"],
    hdrs = ["tpu_node_device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tpu_defs",
    srcs = ["tpu_defs.cc"],
    hdrs = ["tpu_defs.h"],
    deps = ["//tensorflow/core:protos_all_cc"],
)

cc_library(
    name = "tpu_configuration",
    srcs = ["tpu_configuration.cc"],
    hdrs = ["tpu_configuration.h"],
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "tpu_init_mode",
    srcs = ["tpu_init_mode.cc"],
    hdrs = ["tpu_init_mode.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tpu_config_c_api",
    hdrs = ["tpu_config_c_api.h"],
    deps = [
        ":libtftpu_header",
        "//tensorflow/c:tf_status",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_api",
    srcs = ["tpu_api.cc"],
    hdrs = ["tpu_api.h"],
    deps = [
        ":libtftpu_header",
        ":tpu_config_c_api",
        "//tensorflow/core/tpu/kernels:tpu_compile_c_api_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_mesh_state_c_api_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_util_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_node_context_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_api_dlsym_initializer",
    srcs = if_windows(
        ["tpu_api_dlsym_initializer_windows.cc"],
        otherwise = ["tpu_api_dlsym_initializer.cc"],
    ),
    hdrs = ["tpu_api_dlsym_initializer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        ":tpu_api",
        ":tpu_config_c_api",
        ":tpu_library_init_fns",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu/kernels:tpu_compile_c_api_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_mesh_state_c_api_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_util_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:tpu_node_context_c_api_hdrs",
    ] + select({
        "//tensorflow:oss": [
            "//tensorflow/stream_executor/tpu:tpu_executor_base",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tpu_library_init_fns",
    hdrs = ["tpu_library_init_fns.inc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "virtual_device",
    srcs = ["virtual_device.cc"],
    hdrs = ["virtual_device.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:protos_all_cc",
    ],
)
