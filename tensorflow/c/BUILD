# Description:
# C API for TensorFlow, for use by client language bindings.

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "if_google",
    "if_not_mobile",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
    "tf_custom_op_library",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_cuda_cc_test")
load(
    "//tensorflow/core/tpu:build_defs.bzl",
    "if_libtpu_tf_status",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "safe_ptr_hdr",
    srcs = ["safe_ptr.h"],
    visibility = [
        "//tensorflow:internal",
    ],
)

cc_library(
    name = "safe_ptr",
    srcs = [
        "safe_ptr.cc",
        "//tensorflow/c/eager:headers",
    ],
    hdrs = ["safe_ptr.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":c_api_internal",
    ],
)

# -----------------------------------------------------------------------------
# Public targets

filegroup(
    name = "headers",
    srcs = [
        "c_api.h",
        "c_api_experimental.h",
        "c_api_macros.h",
        "tensor_interface.h",
        "tf_attrtype.h",
        "tf_buffer.h",
        "tf_datatype.h",
        "tf_file_statistics.h",
        "tf_status.h",
        "tf_tensor.h",
        "tf_tensor_helper.h",
        "tf_tstring.h",
        "//tensorflow/core/platform:ctstring",
        "@local_xla//xla/tsl/c:headers",
    ] + if_tensorrt([
        "//tensorflow/compiler/tf2tensorrt:headers",
    ]),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "c_api_experimental.cc",
            "c_api_experimental.h",
            "python_api.cc",
            "python_api.h",
            "*test*",
        ],
    ) + [
        "//tensorflow/cc:srcs_no_runtime",
        "//tensorflow/core/distributed_runtime:server_lib.h",
        "@local_tsl//tsl/platform:ctstring",
        "@local_xla//xla/tsl/c:srcs",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pywrap_required_hdrs",
    textual_hdrs = [
        "c_api_internal.h",
        "c_api_macros.h",
        "conversion_macros.h",
        "python_api.h",
        "tensor_interface.h",
        "tf_status_helper.h",
        "tf_buffer_internal.h",
        "tf_status_internal.h",
        "tf_tensor_helper.h",
        "tf_tensor_internal.h",
        "@local_xla//xla/tsl/c:tsl_status_internal_headers",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)

cc_library(
    name = "c_api_headers",
    hdrs = [
        "c_api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_attrtype",
        ":tf_buffer_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        ":tf_tensor_hdrs",
        # TODO: Only include tf_tstring_hdrs. Don't expose the implementation of TF_TString to API
        # users.
        ":tf_tstring",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cuda_library(
    name = "c_api_internal",
    hdrs = [
        "c_api.h",
        "c_api_internal.h",
        "c_api_macros.h",
        "tf_buffer.h",
        "tf_datatype.h",
        "tf_tensor.h",
        "tf_tstring.h",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/c:__subpackages__",
    ],
    deps = selects.with_or({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        ("//tensorflow:chromiumos", "//tensorflow:fuchsia"): [
            ":tf_attrtype",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core/platform:platform",
        ],
        "//conditions:default": [
            ":tf_attrtype",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core/platform:platform",
            "//tensorflow/core:op_gen_lib",
            "//tensorflow/core/distributed_runtime:server_lib",
        ],
    }) + [
        ":tf_buffer_internal",
        ":tf_status_helper",
        ":tf_status_internal",
        ":tf_tensor_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
)

filegroup(
    name = "pywrap_tf_session_hdrs",
    srcs = [
        "python_api.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "tf_attrtype",
    hdrs = ["tf_attrtype.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "c_api_macros_hdrs",
    hdrs = [
        "c_api_macros.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "c_api_macros",
    hdrs = [
        "c_api_macros.h",
        "c_api_macros_internal.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cuda_library(
    name = "c_api",
    hdrs = [
        "c_api.h",
        "tf_attrtype.h",
        "tf_buffer.h",
        "tf_datatype.h",
        "tf_file_statistics.h",
        "tf_status.h",
        "tf_tensor.h",
        "tf_tstring.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_internal",
        ":c_api_macros_hdrs",
        ":c_api_no_xla",
        ":tf_attrtype",
        ":tf_buffer",
        ":tf_file_statistics",
        ":tf_status_helper",
        ":tf_status_internal",
        ":tf_tensor_internal",
        ":tf_tstring",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/c:tsl_status",
    ] + select({
        "//tensorflow:with_xla_support": [
            "//tensorflow/compiler/jit",
            "//tensorflow/compiler/tf2xla:xla_compiler",
        ],
        "//conditions:default": [],
    }) + if_tensorrt([
        "//tensorflow/compiler/tf2tensorrt:trt_convert_api",
    ]),
)

# Check that c_api_no_xla does not depend on xla.
check_deps(
    name = "c_api_no_xla_check_deps",
    disallowed_deps = ["//tensorflow/compiler/jit:xla_kernel_creator"],
    deps = [":c_api_no_xla"],
)

tf_cuda_library(
    name = "c_api_no_xla",
    srcs = [
        "c_api.cc",
        "c_api_function.cc",
    ],
    hdrs = [
        "c_api.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow/python:__subpackages__",
    ],
    deps = [
        ":c_api_internal",
        ":tf_attrtype",
        ":tf_buffer",
        ":tf_buffer_internal",
        ":tf_datatype",
        ":tf_status_helper",
        ":tf_status_internal",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":env",
            ":logging",
            ":tf_status",
            ":tf_tensor",
            "//tensorflow/c/experimental/filesystem:modular_filesystem",
            "//tensorflow/cc:grad_ops",
            "//tensorflow/cc:gradients",
            "//tensorflow/cc:ops",
            "//tensorflow/cc:scope_internal",
            "//tensorflow/cc:while_loop",
            "//tensorflow/cc/saved_model:loader_lite",
            "//tensorflow/compiler/mlir/tfr:graph_decompose_pass",
            "//tensorflow/compiler/mlir/tfr:node_expansion_pass",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:op_gen_lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/config:flag_defs",
            "//tensorflow/core/config:flags",
            "//tensorflow/core/distributed_runtime:server_lib",
            "//tensorflow/core/kernels:logging_ops",
            "@com_google_absl//absl/strings",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:stringprintf",
    ],
)

tf_cuda_library(
    name = "tf_status_internal",
    hdrs = [
        "tf_status.h",
        "tf_status_internal.h",
        "@local_xla//xla/tsl/c:tsl_status_internal_headers",
    ],
    visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/cc/experimental/libtf:__pkg__",
        "//tensorflow/cc/experimental/libtf:__subpackages__",
        # copybara:uncomment_begin(google-only)
        # "//tensorflow/cc/experimental/tf2:__pkg__",
        # "//tensorflow/cc/experimental/tf2:__subpackages__",
        # copybara:uncomment_end
        "//tensorflow/compiler/mlir/tensorflow/c:__subpackages__",
        "//tensorflow/core/transforms:__subpackages__",
    ],
    deps = [
        ":c_api_macros_hdrs",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla/tsl/c:tsl_status",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
        ],
    }),
)

filegroup(
    name = "tf_status_internal_headers",
    srcs = [
        "tf_status_internal.h",
        "@local_xla//xla/tsl/c:tsl_status_internal_headers",
    ],
    visibility = [
        "//tensorflow/python:__subpackages__",
    ],
)

cc_library(
    name = "tf_shape",
    srcs = ["tf_shape.cc"],
    hdrs = ["tf_shape.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        ":tf_shape_internal",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tf_shape_internal",
    hdrs = ["tf_shape_internal.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:internal"],
    deps = [
        ":conversion_macros",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tf_status",
    srcs = ["tf_status.cc"],
    hdrs = ["tf_status.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_status_internal",
        "@local_xla//xla/tsl/c:tsl_status",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
        ],
    }),
)

cc_library(
    name = "tf_status_headers",
    hdrs = ["tf_status.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        "@local_xla//xla/tsl/c:tsl_status_headers",
    ],
)

cc_library(
    name = "tf_tstring",
    srcs = [
        "tf_tstring.cc",
    ],
    hdrs = [
        "tf_tstring.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        ":tf_tensor_hdrs",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:tstring",
        "@local_xla//xla/tsl/c:tsl_status",
    ],
)

cc_library(
    name = "tf_file_statistics",
    hdrs = ["tf_file_statistics.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_interface",
    hdrs = ["tensor_interface.h"],
    visibility = ["//tensorflow:internal"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
        ],
    }),
)

cc_library(
    name = "tf_datatype_hdrs",
    hdrs = ["tf_datatype.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api_macros_hdrs",
    ],
)

cc_library(
    name = "tf_datatype",
    srcs = ["tf_datatype.cc"],
    hdrs = ["tf_datatype.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tf_tensor_hdrs",
    hdrs = [
        "tf_tensor.h",
        "tf_tensor_helper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "tf_tensor",
    srcs = ["tf_tensor.cc"],
    hdrs = [
        "tf_tensor.h",
        "tf_tensor_helper.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        ":tensor_interface",
        ":tf_datatype",
        ":tf_status",
        ":tf_status_helper",
        ":tf_tensor_internal",
        "//tensorflow/core/platform:status",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:casts",
        ],
    }),
)

tf_cuda_library(
    name = "tf_tensor_internal",
    hdrs = [
        "tf_tensor.h",
        "tf_tensor_helper.h",
        "tf_tensor_internal.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api_macros",
        ":tensor_interface",
        ":tf_datatype",
        ":tf_status",
        "//tensorflow/core/platform:status",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:casts",
        ],
    }),
)

cc_library(
    name = "tf_buffer_hdrs",
    hdrs = [
        "tf_buffer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
    ],
)

cc_library(
    name = "tf_buffer",
    srcs = [
        "tf_buffer.cc",
    ],
    hdrs = [
        "tf_buffer.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_buffer_internal",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
    ],
)

tf_cuda_library(
    name = "tf_buffer_internal",
    hdrs = [
        "tf_buffer.h",
        "tf_buffer_internal.h",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/c:__subpackages__",
    ],
    deps = [
        ":c_api_macros_hdrs",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
    ],
)

tf_cuda_library(
    name = "c_api_experimental",
    srcs = [
        "c_api_experimental.cc",
    ],
    hdrs = [
        "c_api_experimental.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_api_macros_hdrs",
        ":checkpoint_reader",
        ":tf_buffer",
        ":tf_buffer_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:get_compiler_ir",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_plugin_init",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:blocking_counter",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "version_script.lds",
        "exported_symbols.lds",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "checkpoint_reader_hdrs",
    srcs = [
        "checkpoint_reader.h",
        "tf_status_helper.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

tf_cuda_library(
    name = "tf_status_helper",
    srcs = ["tf_status_helper.cc"],
    hdrs = ["tf_status_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status_headers",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla/tsl/c:tsl_status_helper",
    ],
)

tf_cc_test(
    name = "tf_status_helper_test",
    srcs = ["tf_status_helper_test.cc"],
    deps = [
        ":tf_status",
        ":tf_status_helper",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

tf_cuda_library(
    name = "checkpoint_reader",
    srcs = ["checkpoint_reader.cc"],
    hdrs = ["checkpoint_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status",
        ":tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/core/util/tensor_bundle:naming",
    ],
)

tf_cuda_library(
    name = "env",
    srcs = [
        "env.cc",
    ],
    hdrs = [
        "env.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
        ],
    }) + [
        ":c_api_macros",
        ":tf_file_statistics",
        ":tf_status",
        ":tf_status_helper",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "kernels_hdrs",
    hdrs = [
        "kernels.h",
        "kernels_experimental.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api_headers",
        ":c_api_macros_hdrs",
        ":tf_buffer_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        ":tf_tensor_hdrs",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api",
        "@local_tsl//tsl/platform",
    ],
)

tf_cuda_library(
    name = "kernels",
    srcs = [
        "kernels.cc",
    ],
    hdrs = [
        "kernels.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_internal",
        ":c_api_macros_hdrs",
        ":tf_buffer",
        ":tf_buffer_internal",
        ":tf_status",
        ":tf_status_helper",
        ":tf_tensor_internal",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api_internal",
        "//tensorflow/core/platform:notification",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":tf_tensor",
            "//tensorflow/c/experimental/stream_executor",
            "//tensorflow/c/experimental/stream_executor:stream_executor_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_lite",
            "//tensorflow/core:protos_all_cc",
            "@local_tsl//tsl/framework:device_id_utils",
            "@local_tsl//tsl/platform:statusor",
            "@local_xla//xla/stream_executor",
        ],
    }),
)

cc_library(
    name = "kernels_experimental_hdrs",
    hdrs = ["kernels_experimental.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api_macros_hdrs",
        ":kernels_hdrs",
    ],
)

tf_cuda_library(
    name = "kernels_experimental",
    srcs = ["kernels_experimental.cc"],
    hdrs = ["kernels_experimental.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":kernels",
        ":tf_status_helper",
        ":tf_status_internal",
        ":tf_tensor_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/core/lib/gtl:cleanup",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:refcount",
    ] + if_not_mobile([
        "//tensorflow/core/kernels:tensor_list",
        "//tensorflow/core/kernels:tensor_list_util",
        "//tensorflow/core/kernels:variant_ops_util",
        "//tensorflow/core/kernels/data:optional_ops_util",
        "//tensorflow/core/platform:abi",
    ]),
    alwayslink = 1,
)

tf_cuda_library(
    name = "ops",
    srcs = [
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype",
        ":tf_status_helper",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
        ],
    }) + if_libtpu_tf_status(),
    alwayslink = 1,
)

cc_library(
    name = "ops_hdrs",
    hdrs = ["ops.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype",
        ":tf_status_headers",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cuda_library(
    name = "c_test_util",
    testonly = 1,
    srcs = ["c_test_util.cc"],
    hdrs = ["c_test_util.h"],
    visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "c_test",
    srcs = ["c_test.c"],
    extra_copts = ["-std=c11"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":env",
        ":kernels",
    ],
)

tf_cuda_cc_test(
    name = "c_api_test",
    size = "medium",
    srcs = ["c_api_test.cc"],
    data = [
        ":test_op1.so",
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = [
        "no_cuda_asan",  # TODO(b/181771536)
        "no_windows",  # TODO(b/155444728)
    ],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_test_util",
        ":test_op_kernel",
        ":tf_buffer",
        ":tf_buffer_internal",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/compiler/jit",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:spectral_ops_op_lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/platform:resource_loader",
    ],
)

tf_cc_test(
    name = "c_api_experimental_test",
    size = "medium",
    srcs = ["c_api_experimental_test.cc"],
    data = [
        "testdata/tf_record",
        "//tensorflow/c/experimental/stream_executor/test:test_pluggable_device.so",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:test_next_pluggable_device_plugin.so",
    ],
    extra_copts = if_google(["-DTENSORFLOW_NO_SHARED_OBJECTS=1"]),
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_test_util",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "c_api_function_test",
    size = "medium",
    srcs = ["c_api_function_test.cc"],
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_test_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "while_loop_test",
    size = "medium",
    srcs = ["while_loop_test.cc"],
    deps = [
        ":c_api",
        ":c_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_custom_op_library(
    name = "test_op1.so",
    srcs = ["test_op1.cc"],
)

tf_kernel_library(
    name = "test_op_kernel",
    srcs = ["test_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "env_test",
    size = "medium",
    srcs = ["env_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":env",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "kernels_test",
    size = "medium",
    srcs = ["kernels_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = ["no_cuda_on_cpu_tap"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "ops_test",
    size = "medium",
    srcs = ["ops_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

# -----------------------------------------------------------------------------
# Python API target

tf_cuda_library(
    name = "python_api",
    srcs = ["python_api.cc"],
    hdrs = ["python_api.h"],
    visibility = ["//tensorflow/python:__pkg__"],
    deps = [
        ":c_api",
        ":c_api_internal",
        "//tensorflow/core:protos_all_cc",
        # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    ],
    alwayslink = 1,
)

cc_library(
    name = "conversion_macros",
    hdrs = [
        "conversion_macros.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "c_op_requires",
    hdrs = ["c_op_requires.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/platform:macros"],
)
