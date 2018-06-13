# Legacy command line flags for the XLA bridge libraries.

# Please do not add more flags to this package.

# The XLA bridge libraries were written in an environment that allowed
# command-line flags to be scattered freely throughout the libraries.  This
# model, while initially convenient, leads to a proliferation in unused command
# line flags in tests and binaries, and serious problems in servers, where one
# might wish parameters to be different in independent RPC calls to the same
# routine.
#
# Please don't add more flags.  If you're a library author, pass options and
# parameters explicitly through the library's interface.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

cc_library(
    name = "mark_for_compilation_pass_flags",
    srcs = ["mark_for_compilation_pass_flags.cc"],
    hdrs = ["mark_for_compilation_pass_flags.h"],
    deps =
        [
            "//tensorflow/compiler/xla/legacy_flags:parse_flags_from_env",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "parallel_check_op_flags",
    srcs = ["parallel_check_op_flags.cc"],
    hdrs = ["parallel_check_op_flags.h"],
    deps =
        [
            "//tensorflow/compiler/xla/legacy_flags:parse_flags_from_env",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)

cc_library(
    name = "xla_device_flags",
    srcs = ["xla_device_flags.cc"],
    hdrs = ["xla_device_flags.h"],
    deps =
        [
            "//tensorflow/compiler/xla/legacy_flags:parse_flags_from_env",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
)
