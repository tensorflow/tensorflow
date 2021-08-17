# Test helper binaries to use in tests.
#   These are provided to avoid linux-only binaries in tests.
#   Thus helping write cross platform tests.

package(
    default_visibility = [
        "//tensorflow/core/platform:__pkg__",
    ],
    licenses = ["notice"],
)

cc_binary(
    name = "test_echo",
    srcs = ["test_echo.cc"],
)

cc_binary(
    name = "test_noop",
    srcs = ["test_noop.cc"],
)

cc_binary(
    name = "test_echo_argv_1",
    srcs = ["test_echo_argv_1.cc"],
)

cc_binary(
    name = "test_stderr",
    srcs = ["test_stderr.cc"],
)
