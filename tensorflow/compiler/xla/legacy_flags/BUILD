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
    name = "debug_options_flags",
    srcs = ["debug_options_flags.cc"],
    hdrs = ["debug_options_flags.h"],
    deps =
        [
            ":parse_flags_from_env",
            "//tensorflow/compiler/xla:xla_proto",
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
