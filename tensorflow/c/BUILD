# Description:
# C API for TensorFlow, for use by client language bindings.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
    "tf_custom_op_library",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

# -----------------------------------------------------------------------------
# Public targets

filegroup(
    name = "headers",
    srcs = [
        "c_api.h",
        "c_api_experimental.h",
        "tensor_interface.h",
        "tf_attrtype.h",
        "tf_datatype.h",
        "tf_file_statistics.h",
        "tf_status.h",
        "tf_tensor.h",
    ],
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
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "c_api_internal.h",
        "conversion_macros.h",
        "python_api.h",
        "tensor_interface.h",
        "tf_status_helper.h",
        "tf_status_internal.h",
        "tf_tensor_internal.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

tf_cuda_library(
    name = "c_api_internal",
    hdrs = [
        "c_api.h",
        "c_api_internal.h",
        "tf_datatype.h",
        "tf_tensor.h",
    ],
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/c:__subpackages__",
    ],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//tensorflow:chromiumos": [
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
        ":tf_status_internal",
        ":tf_tensor_internal",
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
    name = "c_api_macros",
    hdrs = ["c_api_macros.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
)

tf_cuda_library(
    name = "c_api",
    hdrs = [
        "c_api.h",
        "tf_attrtype.h",
        "tf_datatype.h",
        "tf_file_statistics.h",
        "tf_status.h",
        "tf_tensor.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_no_xla",
        ":c_api_internal",
        ":tf_attrtype",
        ":tf_status_internal",
        ":tf_file_statistics",
        ":tf_tensor_internal",
    ] + select({
        "//tensorflow:with_xla_support": [
            "//tensorflow/compiler/tf2xla:xla_compiler",
            "//tensorflow/compiler/jit",
        ],
        "//conditions:default": [],
    }),
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
        "//tensorflow/c:__subpackages__",
        "//third_party/llvm/llvm-project:__subpackages__",
    ],
    deps = [
        ":c_api_internal",
        ":tf_attrtype",
        ":tf_datatype",
        ":tf_status_internal",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":tf_status",
            ":tf_tensor",
            "@com_google_absl//absl/strings",
            "//tensorflow/cc/saved_model:loader_lite",
            "//tensorflow/cc:gradients",
            "//tensorflow/cc:ops",
            "//tensorflow/cc:grad_ops",
            "//tensorflow/cc:scope_internal",
            "//tensorflow/cc:while_loop",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:op_gen_lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core/distributed_runtime:server_lib",
            "//tensorflow/core/kernels:logging_ops",
        ],
    }),
    alwayslink = 1,
)

tf_cuda_library(
    name = "tf_status_internal",
    hdrs = [
        "tf_status.h",
        "tf_status_internal.h",
    ],
    visibility = [
        "//tensorflow/c:__subpackages__",
    ],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:lib",
        ],
    }),
)

cc_library(
    name = "tf_status",
    srcs = ["tf_status.cc"],
    hdrs = ["tf_status.h"],
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":tf_status_internal",
            "//tensorflow/core:lib",
        ],
    }),
)

cc_library(
    name = "tf_status_headers",
    hdrs = ["tf_status.h"],
    visibility = ["//visibility:public"],
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
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tf_datatype",
    srcs = ["tf_datatype.cc"],
    hdrs = ["tf_datatype.h"],
    visibility = ["//visibility:public"],
    deps = select({
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
    name = "tf_tensor",
    srcs = ["tf_tensor.cc"],
    hdrs = ["tf_tensor.h"],
    visibility = ["//visibility:public"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":tensor_interface",
            ":tf_datatype",
            ":tf_status",
            ":tf_status_helper",
            ":tf_tensor_internal",
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
        "tf_tensor_internal.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":tensor_interface",
            ":tf_datatype",
            ":tf_status",
            "//tensorflow/core:framework",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/platform:casts",
        ],
    }),
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
        ":checkpoint_reader",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:attr_builder",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/platform",
        "@com_google_absl//absl/strings",
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
        ":tf_status",
        ":tf_status_internal",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "tf_status_helper_test",
    srcs = ["tf_status_helper_test.cc"],
    deps = [
        ":tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_library(
    name = "checkpoint_reader",
    srcs = ["checkpoint_reader.cc"],
    hdrs = ["checkpoint_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/tensor_bundle",
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
        ":c_api",
        ":tf_status_helper",
        ":c_api_internal",
        ":tf_file_statistics",
        "//tensorflow/core:lib",
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
        ":tf_status",
        ":tf_status_helper",
        ":tf_tensor_internal",
    ] + select({
        "//tensorflow:android": [
            ":c_api_internal",
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            ":c_api_internal",
            ":tf_tensor",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_lite",
        ],
    }),
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
        ":tf_datatype",
        ":tf_status",
        ":tf_status_helper",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
        ],
    }),
    alwayslink = 1,
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
    size = "small",
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
        "no_windows",  # TODO(b/155444728)
        "noasan",
    ],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
    deps = [
        ":c_api",
        ":c_test_util",
        ":test_op_kernel",
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
    data = ["testdata/tf_record"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = ["notsan"],  # b/149031034
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
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
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "c_api_function_test",
    size = "small",
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
    size = "small",
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
    size = "small",
    srcs = ["env_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = ["noasan"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
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
    size = "small",
    srcs = ["kernels_test.cc"],
    linkopts = select({
        "//tensorflow:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = ["no_cuda_on_cpu_tap"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
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
        "//third_party/eigen3",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tf_cc_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    tags = ["noasan"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    # linkstatic = tf_kernel_tests_linkstatic(),
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
        # TODO(b/74620627): remove when _USE_C_SHAPES is removed
        "//tensorflow/python:cpp_shape_inference_proto_cc",
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
