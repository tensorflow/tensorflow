load(
    "@bazel_skylib//:bzl_library.bzl",
    "bzl_library",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_not_fuchsia",
    "internal_visibility",
    "tsl_copts",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_logging_deps",
    "tf_platform_alias",
    "tf_platform_deps",
    "tf_windows_aware_platform_deps",
    "tsl_cc_test",
)
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_static",
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
        "env_time.h",
        "env.cc",
        "file_system.cc",
        "logging.h",
        "file_system.h",
        "file_system_helper.cc",
        "file_system_helper.h",
        "test.h",
        "threadpool.cc",
        "threadpool.h",
        "env.h",
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
        "test.h",
        "test_benchmark.h",
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
        "test.h",
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
        "env.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "statusor.h",
        "subprocess.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Header files for tensorflow/core:platform_base.
filegroup(
    name = "base_hdrs",
    srcs = [
        "env_time.h",
        "macros.h",
        "threadpool.h",
        "threadpool_interface.h",
        "threadpool_options.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
        "macros.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "env.cc",
        "env.h",
        "env_time.h",
        "errors.cc",
        "errors.h",
        "file_statistics.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.h",
        "macros.h",
        "status.cc",
        "status.h",
        "statusor.h",
        "threadpool.cc",
        "threadpool.h",
        "threadpool_interface.h",
    ] + select({
        "//xla/tsl:fuchsia": [],
        "//conditions:default": [
            "file_system_helper.cc",
        ],
    }),
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
        "macros.h",
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
        "status.h",
        "statusor.h",
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
        "macros.h",
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
        "macros.h",
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
        "macros.h",
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
        "macros.h",
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
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:subprocess",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "env",
    textual_hdrs = [
        "env.h",
        "file_system.h",
        "file_system_helper.h",
        "threadpool.h",
    ],
    deps = tf_windows_aware_platform_deps("env") + if_static([":env_impl"]),
)

cc_library(
    name = "env_impl",
    deps = tf_windows_aware_platform_deps("env_impl"),
)

cc_library(
    name = "env_time",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["env_time.h"],
    deps = tf_windows_aware_platform_deps("env_time"),
)

cc_library(
    name = "errors",
    srcs = ["errors.cc"],
    hdrs = ["errors.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
    ],
)

tsl_cc_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.cc"],
    deps = [
        ":errors",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "file_statistics",
    hdrs = ["file_statistics.h"],
    deps = [
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "logging",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["logging.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = tf_logging_deps(),
)

tsl_cc_test(
    name = "logging_test",
    size = "small",
    srcs = [
        "logging_test.cc",
    ],
    deps = [
        ":logging",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:stacktrace_handler",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:stack_frame",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringprintf",
        "@local_tsl//tsl/platform:types",
    ] + tf_platform_deps("status"),
)

tsl_cc_test(
    name = "status_test",
    size = "small",
    srcs = ["status_test.cc"],
    deps = [
        ":status",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:stack_frame",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:status_to_from_proto",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    srcs = ["status_matchers.cc"],
    hdrs = ["status_matchers.h"],
    deps = [
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "status_matchers_test",
    size = "small",
    srcs = ["status_matchers_test.cc"],
    deps = [
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "status_to_from_proto",
    srcs = [
        "status_to_from_proto.cc",
    ],
    hdrs = ["status_to_from_proto.h"],
    deps = [
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@local_tsl//tsl/platform:status",
    ] + tf_platform_deps("status"),
)

cc_library(
    name = "statusor",
    hdrs = ["statusor.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:status",
    ] + tf_platform_deps("statusor"),
)

tsl_cc_test(
    name = "statusor_test",
    size = "small",
    srcs = ["statusor_test.cc"],
    deps = [
        ":statusor",
        "@com_google_absl//absl/base:config",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "test",
    testonly = True,
    srcs = ["test.cc"],
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["test.h"],
    deps = [
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:net",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_benchmark//:benchmark",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    copts = tsl_copts(),
    linkopts = select({
        "//xla/tsl:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:stacktrace_handler",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
    alwayslink = 1,
)

cc_library(
    name = "threadpool_async_executor",
    hdrs = ["threadpool_async_executor.h"],
    deps = [
        "//xla/tsl/concurrency:async_value",
        "@local_tsl//tsl/platform:env",
    ],
)

tsl_cc_test(
    name = "threadpool_async_executor_test",
    srcs = ["threadpool_async_executor_test.cc"],
    deps = [
        ":threadpool_async_executor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:env_impl",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "threadpool_interface",
    hdrs = ["threadpool_interface.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "threadpool_options",
    hdrs = ["threadpool_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/tsl/platform:threadpool_interface",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:bfloat16",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:tstring",
    ] + tf_platform_deps("types"),
)
