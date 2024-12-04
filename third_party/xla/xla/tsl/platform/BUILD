load(
    "@bazel_skylib//:bzl_library.bzl",
    "bzl_library",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_not_fuchsia",
    "internal_visibility",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_platform_alias",
    "tf_windows_aware_platform_deps",
    "tsl_cc_test",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    [
        "subprocess.h",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/platform:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "//xla/tsl:__subpackages__",
    ]),
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
    deps = tf_platform_alias("rules_cc_bzl"),
)

bzl_library(
    name = "build_config_root_bzl",
    srcs = ["build_config_root.bzl"],
    deps = tf_platform_alias("build_config_root_bzl"),
)

filegroup(
    name = "test_hdrs",
    testonly = 1,
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@local_tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "android_test_srcs",
    testonly = 1,
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@local_tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "ctstring",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "subprocess.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Header files for tensorflow/core:platform_base.
filegroup(
    name = "base_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [] + if_not_fuchsia(["subprocess.h"]),
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@local_tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "legacy_lib_internal_headers",
    srcs = glob(
        [
            "*.h",
        ],
        exclude = [
            "dynamic_annotations.h",
        ],
    ) + [
        "//xla/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//xla/tsl/platform/profile_utils:clock_cycle_profiler.h",
        "//xla/tsl/platform/profile_utils:cpu_utils.h",
        "//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "@local_tsl//tsl/platform:__pkg__",
        "//tensorflow/core:__pkg__",
    ]),
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "tflite_portable_logging_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "jpeg_internal_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "gif_internal_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/platform:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "stacktrace_handler_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_internal_private_hdrs",
    srcs = [
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "@local_tsl//tsl/platform:__pkg__",
    ]),
)

filegroup(
    name = "tensor_float_32_hdr",
    srcs = [],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "subprocess",
    textual_hdrs = [
        "subprocess.h",
    ],
    deps = tf_windows_aware_platform_deps("subprocess"),
)

tsl_cc_test(
    name = "subprocess_test",
    size = "small",
    srcs = ["subprocess_test.cc"],
    data = [
        "//xla/tsl/platform/testdata:test_echo",
        "//xla/tsl/platform/testdata:test_echo_argv_1",
        "//xla/tsl/platform/testdata:test_noop",
        "//xla/tsl/platform/testdata:test_stderr",
    ],
    tags = ["no_oss"],  # TODO(b/327036247): revisit after this moves to XLA
    deps = [
        ":subprocess",
        "//xla/tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
