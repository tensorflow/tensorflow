# Legacy command line flags for the XLA libraries.

# Please do not add more flags to this package.

# The XLA libraries were written in an environment that allowed command - line
# flags to be scattered freely throughout the libraries.  This model, while
# initially convenient, leads to a proliferation in unused commnd line flags in
# tests and binaries, and serious problems in servers, where one might wish
# parameters to be different in independent RPC calls to the same routine.
#
# Please don't add more flags.  If you're a library author, pass options and
# parameters explicitly through the library's interface.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "parse_flags_from_env",
    srcs = ["parse_flags_from_env.cc"],
    hdrs = ["parse_flags_from_env.h"],
    deps =
        [
            "//tensorflow/compiler/xla:types",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_test(
    name = "parse_flags_from_env_test",
    size = "small",
    srcs = ["parse_flags_from_env_test.cc"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/compiler/xla:types",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:test",
        ],
)

cc_library(
    name = "layout_util_flags",
    srcs = ["layout_util_flags.cc"],
    hdrs = ["layout_util_flags.h"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/compiler/xla:types",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "util_flags",
    srcs = ["util_flags.cc"],
    hdrs = ["util_flags.h"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "debug_options_flags",
    srcs = ["debug_options_flags.cc"],
    hdrs = ["debug_options_flags.h"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/compiler/xla:types",
            "//tensorflow/compiler/xla:xla_proto",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "cpu_compiler_flags",
    srcs = ["cpu_compiler_flags.cc"],
    hdrs = ["cpu_compiler_flags.h"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/compiler/xla:types",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "cpu_runtime_flags",
    srcs = ["cpu_runtime_flags.cc"],
    hdrs = ["cpu_runtime_flags.h"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "compiler_functor_flags",
    srcs = ["compiler_functor_flags.cc"],
    hdrs = ["compiler_functor_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "convolution_thunk_flags",
    srcs = ["convolution_thunk_flags.cc"],
    hdrs = ["convolution_thunk_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_compiler_flags",
    srcs = ["gpu_compiler_flags.cc"],
    hdrs = ["gpu_compiler_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "gpu_backend_lib_flags",
    srcs = ["gpu_backend_lib_flags.cc"],
    hdrs = ["gpu_backend_lib_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "stream_assignment_flags",
    srcs = ["stream_assignment_flags.cc"],
    hdrs = ["stream_assignment_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_graph_dumper_flags",
    srcs = ["hlo_graph_dumper_flags.cc"],
    hdrs = ["hlo_graph_dumper_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "alias_analysis_flags",
    srcs = ["alias_analysis_flags.cc"],
    hdrs = ["alias_analysis_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "llvm_util_flags",
    srcs = ["llvm_util_flags.cc"],
    hdrs = ["llvm_util_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "service_flags",
    srcs = ["service_flags.cc"],
    hdrs = ["service_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "buffer_assignment_flags",
    srcs = ["buffer_assignment_flags.cc"],
    hdrs = ["buffer_assignment_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "hlo_test_base_flags",
    srcs = ["hlo_test_base_flags.cc"],
    hdrs = ["hlo_test_base_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "backend_flags",
    srcs = ["backend_flags.cc"],
    hdrs = ["backend_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "user_computation_flags",
    srcs = ["user_computation_flags.cc"],
    hdrs = ["user_computation_flags.h"],
    deps = [
        ":parse_flags_from_env",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
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
