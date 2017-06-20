licenses(["notice"])  # Apache 2.0

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/aot/...",
        "//tensorflow/compiler/jit/...",
        "//tensorflow/compiler/tests/...",
        "//tensorflow/compiler/tf2xla/...",
    ],
)

package_group(
    name = "friends",
    includes = [":internal"],
    packages = ["//tensorflow/..."],
)

package(
    default_visibility = [":internal"],
)

cc_library(
    name = "xla_compiler",
    srcs = [
        "xla_compilation_device.cc",
        "xla_compiler.cc",
        "xla_context.cc",
        "xla_helpers.cc",
        "xla_op_kernel.cc",
        "xla_op_registry.cc",
    ],
    hdrs = [
        "xla_compilation_device.h",
        "xla_compiler.h",
        "xla_context.h",
        "xla_helpers.h",
        "xla_op_kernel.h",
        "xla_op_registry.h",
    ],
    visibility = [":friends"],
    deps = [
        ":common",
        ":dump_graph",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:computation",
        "//tensorflow/compiler/xla/client:computation_builder",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:stream_executor_no_cuda",
    ],
    alwayslink = 1,
)

cc_library(
    name = "common",
    srcs = [
        "literal_util.cc",
        "shape_util.cc",
        "str_util.cc",
        "type_util.cc",
    ],
    hdrs = [
        "literal_util.h",
        "shape_util.h",
        "str_util.h",
        "type_util.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

# Internal targets below this point.

cc_test(
    name = "xla_compiler_test",
    size = "small",
    srcs = ["xla_compiler_test.cc"],
    deps = [
        ":xla_compiler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "str_util_test",
    size = "small",
    srcs = [
        "str_util_test.cc",
    ],
    deps = [
        ":common",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "literal_util_test",
    size = "small",
    srcs = [
        "literal_util_test.cc",
    ],
    deps = [
        ":common",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "const_analysis",
    srcs = ["const_analysis.cc"],
    hdrs = ["const_analysis.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "const_analysis_test",
    size = "small",
    srcs = ["const_analysis_test.cc"],
    deps = [
        ":const_analysis",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "xla_local_runtime_context",
    hdrs = ["xla_local_runtime_context.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:framework_lite"],
)

cc_library(
    name = "dump_graph",
    srcs = [
        "dump_graph.cc",
        "dump_graph_flags.cc",
        "dump_graph_flags.h",
    ],
    hdrs = [
        "dump_graph.h",
    ],
    deps = [
        "//tensorflow/compiler/xla/legacy_flags:parse_flags_from_env",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

# -----------------------------------------------------------------------------

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
