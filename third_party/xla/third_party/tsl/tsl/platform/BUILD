# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load(
    "@local_xla//xla/tsl:tsl.bzl",
    "internal_visibility",
    "tsl_copts",
)
load("@local_xla//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "@local_xla//xla/tsl/platform:build_config.bzl",
    "tf_cuda_root_path_deps",
    "tf_error_logging_deps",
    "tf_fingerprint_deps",
    "tf_google_mobile_srcs_no_runtime",
    "tf_platform_deps",
    "tf_stream_executor_deps",
    "tf_windows_aware_platform_deps",
    "tsl_cc_test",
    "tsl_protobuf_deps",
)
load(
    "@local_xla//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "base64",
    srcs = ["base64.cc"],
    hdrs = ["base64.h"],
    deps = [
        ":stringpiece",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "blocking_counter",
    hdrs = ["blocking_counter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":mutex",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "coding",
    srcs = ["coding.cc"],
    hdrs = ["coding.h"],
    deps = [
        ":stringpiece",
        ":tstring",
        "@local_xla//xla/tsl/platform:byte_order",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "cpu_info_test",
    size = "small",
    srcs = ["cpu_info_test.cc"],
    deps = [
        ":platform_port",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "denormal",
    srcs = ["denormal.cc"],
    hdrs = ["denormal.h"],
    deps = [
        ":platform",
        ":platform_port",
        "@local_xla//xla/tsl/platform:macros",
    ],
)

tsl_cc_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.cc"],
    deps = [
        ":denormal",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
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
    deps = [
        "@local_xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "threadpool_async_executor",
    hdrs = ["threadpool_async_executor.h"],
    deps = [
        "@local_xla//xla/tsl/platform:threadpool_async_executor",
    ],
)

cc_library(
    name = "env_impl",
    deps = [
        "@local_xla//xla/tsl/platform:env_impl",
    ],
)

cc_library(
    name = "env_time",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["env_time.h"],
    deps = [
        "@local_xla//xla/tsl/platform:env_time",
    ],
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    deps = [
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "mutex",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["mutex.h"],
    deps = [
        ":thread_annotations",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "numbers",
    srcs = ["numbers.cc"],
    hdrs = ["numbers.h"],
    deps = [
        ":str_util",
        ":stringpiece",
        ":stringprintf",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        ":mutex",
        ":scanner",
        ":str_util",
        ":strcat",
        ":stringpiece",
        "@com_google_absl//absl/algorithm:container",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "protobuf",
    srcs = [
        "protobuf.cc",
        "protobuf_util.cc",
    ],
    hdrs = ["protobuf.h"],
    deps = [
        ":platform",
        "@local_xla//xla/tsl/platform:types",
    ] + tsl_protobuf_deps(),
)

cc_library(
    name = "regexp",
    hdrs = ["regexp.h"],
    deps = [
        ":platform",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    deps = [
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "status_to_from_proto",
    hdrs = ["status_to_from_proto.h"],
    deps = [
        "@local_xla//xla/tsl/platform:status_to_from_proto",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    hdrs = ["status_matchers.h"],
    deps = [
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "statusor",
    hdrs = ["statusor.h"],
    deps = [
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "thread_annotations",
    hdrs = ["thread_annotations.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "test",
    testonly = True,
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["test.h"],
    deps = [
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/platform:test_benchmark",
    ],
)

filegroup(
    name = "test_hdrs",
    testonly = 1,
    srcs = [
        "test.h",
        "test_benchmark.h",
        "@local_xla//xla/tsl/platform:test_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "android_test_srcs",
    testonly = 1,
    srcs = [
        "test.h",
        "@local_xla//xla/tsl/platform:android_test_srcs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

cc_library(
    name = "tracing",
    textual_hdrs = ["tracing.h"],
    deps = tf_platform_deps("tracing"),
)

cc_library(
    name = "tstring",
    hdrs = [
        "ctstring.h",
        "ctstring_internal.h",
        "tstring.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":cord",
        ":platform",
        ":stringpiece",
    ],
)

filegroup(
    name = "ctstring",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
        "@local_xla//xla/tsl/platform:ctstring",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "abi.h",
        "bfloat16.h",
        "casts.h",
        "coding.h",
        "context.h",
        "cpu_info.h",
        "env.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "fingerprint.h",
        "init_main.h",
        "mem.h",
        "net.h",
        "notification.h",
        "null_file_system.h",
        "numa.h",
        "path.h",
        "protobuf.h",
        "random.h",
        "stacktrace.h",
        "stacktrace_handler.h",
        "statusor.h",
        "str_util.h",
        "stringpiece.h",
        "stringprintf.h",
        "thread_annotations.h",
        "@local_xla//xla/tsl/platform:lib_hdrs",
        "@local_xla//xla/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "@local_xla//xla/tsl/platform/profile_utils:clock_cycle_profiler.h",
        "@local_xla//xla/tsl/platform/profile_utils:cpu_utils.h",
        "@local_xla//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
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
        "@local_xla//xla/tsl/platform:base_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
        "bfloat16.h",
        "cpu_info.h",
        "macros.h",
        "platform.h",
        "protobuf.h",
        "thread_annotations.h",
        "@local_xla//xla/tsl/platform:framework_lite_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "abi.cc",
        "abi.h",
        "base64.cc",
        "base64.h",
        "bfloat16.h",
        "blocking_counter.h",
        "casts.h",
        "coding.cc",
        "coding.h",
        "context.h",
        "cpu_info.cc",
        "cpu_info.h",
        "demangle.h",
        "denormal.cc",
        "denormal.h",
        "env.h",
        "env_time.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "hash.cc",
        "hash.h",
        "host_info.h",
        "init_main.h",
        "load_library.h",
        "macros.h",
        "mem.h",
        "numa.h",
        "numbers.cc",
        "numbers.h",
        "path.cc",
        "path.h",
        "platform.h",
        "platform_strings_computed.h",
        "protobuf.cc",
        "protobuf.h",
        "protobuf_util.cc",
        "raw_coding.h",
        "refcount.h",
        "regexp.h",
        "scanner.cc",
        "scanner.h",
        "setround.cc",
        "setround.h",
        "snappy.h",
        "stacktrace.h",
        "status.h",
        "statusor.h",
        "str_util.cc",
        "str_util.h",
        "strcat.cc",
        "strcat.h",
        "stringpiece.h",
        "stringprintf.cc",
        "stringprintf.h",
        "thread_annotations.h",
        "threadpool.h",
        "threadpool_interface.h",
        "tracing.h",
        "@local_xla//xla/tsl/platform:mobile_srcs_no_runtime",
    ] + select({
        "@local_xla//xla/tsl:fuchsia": tf_google_mobile_srcs_no_runtime(),
        "//conditions:default": [
            "tracing.cc",
            "@local_xla//xla/tsl/platform/default:mobile_srcs_no_runtime",
        ],
    }),
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "error_logging.h",
        "fingerprint.h",
        "notification.h",
        "random.cc",
        "random.h",
        "test_benchmark.h",
        "threadpool_options.h",
        "unbounded_work_queue.h",
        "@local_xla//xla/tsl/platform:mobile_srcs_only_runtime",
        "@local_xla//xla/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.cc",
        "@local_xla//xla/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "@local_xla//xla/tsl/platform/profile_utils:cpu_utils.cc",
        "@local_xla//xla/tsl/platform/profile_utils:cpu_utils.h",
        "@local_xla//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
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
        "@local_xla//xla/tsl/platform:legacy_lib_internal_headers",
        "@local_xla//xla/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "@local_xla//xla/tsl/platform/profile_utils:clock_cycle_profiler.h",
        "@local_xla//xla/tsl/platform/profile_utils:cpu_utils.h",
        "@local_xla//xla/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

exports_files(
    [
        "bfloat16.h",
        "context.h",
        "cpu_info.cc",
        "cpu_info.h",
        "cuda_root_path.h",
        "demangle.h",
        "env.h",
        "env_time.h",
        "error_logging.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.h",
        "host_info.h",
        "human_readable_json.h",
        "init_main.h",
        "logging.h",
        "mem.h",
        "net.h",
        "numa.h",
        "snappy.h",
        "stacktrace_handler.h",
        "mutex.h",
        "load_library.h",
        "stringpiece_test.cc",
        "test.h",
        "threadpool.cc",
        "threadpool.h",
        "tracing.cc",
        "tracing.h",
    ],
    visibility = internal_visibility([
        "//tensorflow:__subpackages__",
    ]),
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
        "init_main.h",
        "macros.h",
        "platform.h",
        "protobuf.h",
        "stringpiece.h",
        "@local_xla//xla/tsl/platform:lib_proto_parsing_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
    ]),
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
        "blocking_counter.h",
        "demangle.h",
        "denormal.h",
        "host_info.h",
        "platform.h",
        "refcount.h",
        "setround.h",
        "snappy.h",
        "status.h",
        "statusor.h",
        "tracing.h",
        "unbounded_work_queue.h",
        "@local_xla//xla/tsl/platform:lib_internal_public_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "tflite_portable_logging_hdrs",
    srcs = [
        "bfloat16.h",
        "macros.h",
        "platform.h",
        "@local_xla//xla/tsl/platform:tflite_portable_logging_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
    ]),
)

filegroup(
    name = "jpeg_internal_hdrs",
    srcs = [
        "bfloat16.h",
        "macros.h",
        "mem.h",
        "platform.h",
        "stringpiece.h",
        "@local_xla//xla/tsl/platform:jpeg_internal_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

filegroup(
    name = "gif_internal_hdrs",
    srcs = [
        "bfloat16.h",
        "macros.h",
        "mem.h",
        "platform.h",
        "@local_xla//xla/tsl/platform:gif_internal_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ]),
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/platform:macros",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "context.h",
        "env_time.h",
        "macros.h",
        "platform.h",
        "@local_xla//xla/tsl/platform:xla_cpu_runtime_srcs",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "net",
    textual_hdrs = ["net.h"],
    deps = tf_windows_aware_platform_deps("net"),
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "platform_port",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = [
        "cpu_info.h",
        "demangle.h",
        "host_info.h",
        "init_main.h",
        "mem.h",
        "numa.h",
        "snappy.h",
    ],
    deps = tf_windows_aware_platform_deps("platform_port") + [
        ":platform",
        "@local_xla//xla/tsl/platform:byte_order",
        "@local_xla//xla/tsl/platform:dynamic_annotations",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "platform_strings",
    srcs = [
        "platform_strings_computed.h",
    ],
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        ":mutex",
        "@local_xla//xla/tsl/platform:types",
    ],
)

filegroup(
    name = "stacktrace_handler_hdrs",
    srcs = [
        "stacktrace_handler.h",
        "@local_xla//xla/tsl/platform:stacktrace_handler_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "stacktrace_handler_hdrs_lib",
    hdrs = ["stacktrace_handler.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "stacktrace_handler",
    textual_hdrs = ["stacktrace_handler.h"],
    deps = tf_windows_aware_platform_deps("stacktrace_handler"),
    alwayslink = 1,
)

tsl_cc_test(
    name = "stacktrace_handler_test",
    size = "small",
    srcs = [
        "stacktrace_handler_test.cc",
    ],
    linkstatic = False,  # In the OSS build, static linking affects the stacktrace.
    tags = ["no_windows"],
    deps = [
        ":stacktrace",
        ":stacktrace_handler",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "str_util",
    srcs = ["str_util.cc"],
    hdrs = ["str_util.h"],
    deps = [
        ":stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "strcat",
    srcs = ["strcat.cc"],
    hdrs = ["strcat.h"],
    deps = [
        ":numbers",
        ":stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "stringpiece",
    hdrs = ["stringpiece.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "stringprintf",
    srcs = ["stringprintf.cc"],
    hdrs = ["stringprintf.h"],
    deps = [
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "cord",
    hdrs = ["cord.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "threadpool_interface",
    hdrs = ["threadpool_interface.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/platform:threadpool_interface",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "build_test",
    testonly = 1,
    visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
    ]),
    deps = [
        ":fingerprint",
        ":net",
        ":platform",
        ":platform_port",
        ":platform_strings",
        ":stacktrace_handler",
        ":stringpiece",
        "@local_xla//xla/tsl/platform:byte_order",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:threadpool_interface",
    ],
)

cc_library(
    name = "bfloat16",
    hdrs = ["bfloat16.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/platform:byte_order",
    ],
)

cc_library(
    name = "ml_dtypes",
    hdrs = ["ml_dtypes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@ml_dtypes_py//ml_dtypes:float8",
        "@ml_dtypes_py//ml_dtypes:intn",
        "@ml_dtypes_py//ml_dtypes:mxfloat",
    ],
)

cc_library(
    name = "dso_loader",
    hdrs = ["dso_loader.h"],
    deps = [
        ":platform",
    ] + tf_stream_executor_deps("dso_loader"),
)

cc_library(
    name = "logging",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["logging.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "error_logging",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["error_logging.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + tf_error_logging_deps(),
)

cc_library(
    name = "hash",
    srcs = ["hash.cc"],
    hdrs = ["hash.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":raw_coding",
        ":stringpiece",
        "@local_xla//xla/tsl/platform:macros",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "human_readable_json",
    textual_hdrs = ["human_readable_json.h"],
    deps = tf_platform_deps("human_readable_json"),
)

cc_library(
    name = "raw_coding",
    hdrs = ["raw_coding.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/platform:byte_order",
        "@local_xla//xla/tsl/platform:types",
    ],
)

filegroup(
    name = "lib_internal_private_hdrs",
    srcs = [
        "bfloat16.h",
        "raw_coding.h",
        "scanner.h",
        "str_util.h",
        "@local_xla//xla/tsl/platform:lib_internal_private_hdrs",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("casts"),
)

cc_library(
    name = "setround",
    srcs = ["setround.cc"],
    hdrs = ["setround.h"],
    deps = [
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:macros",
    ],
)

cc_library(
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
    ] + tf_windows_aware_platform_deps("stacktrace"),
)

tsl_cc_test(
    name = "stacktrace_test",
    size = "small",
    srcs = [
        "stacktrace_test.cc",
    ],
    linkstatic = False,  # In the OSS build, static linking affects the stacktrace.
    tags = ["no_windows"],
    deps = [
        ":stacktrace",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "cuda_root_path",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["cuda_root_path.h"],
    deps = tf_cuda_root_path_deps(),
)

cc_library(
    name = "file_statistics",
    hdrs = ["file_statistics.h"],
    deps = [
        "@local_xla//xla/tsl/platform:file_statistics",
    ],
)

cc_library(
    name = "fingerprint",
    hdrs = ["fingerprint.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        ":stringpiece",
        "@local_xla//xla/tsl/platform:types",
    ] + tf_fingerprint_deps(),
)

tsl_cc_test(
    name = "fingerprint_test",
    size = "small",
    srcs = ["fingerprint_test.cc"],
    deps = [
        ":fingerprint",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "tensor_float_32_utils",
    srcs = ["tensor_float_32_utils.cc"],
    hdrs = ["tensor_float_32_utils.h"],
    copts = tsl_copts(),
    alwayslink = 1,
)

cc_library(
    name = "scanner",
    srcs = ["scanner.cc"],
    hdrs = ["scanner.h"],
    deps = [
        ":str_util",
        ":stringpiece",
        "@local_xla//xla/tsl/platform:macros",
    ],
)

filegroup(
    name = "tensor_float_32_hdr",
    srcs = [
        "tensor_float_32_utils.h",
        "@local_xla//xla/tsl/platform:tensor_float_32_hdr",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "tensor_float_32_hdr_lib",
    hdrs = [":tensor_float_32_hdr"],
    compatible_with = get_compatible_with_portable(),
)

tsl_cc_test(
    name = "ctstring_test",
    size = "small",
    srcs = ["ctstring_test.cc"],
    deps = [
        ":tstring",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "hash_test",
    size = "small",
    srcs = ["hash_test.cc"],
    deps = [
        ":hash",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:test_benchmark",
    ],
)

tsl_cc_test(
    name = "path_test",
    size = "small",
    srcs = ["path_test.cc"],
    deps = [
        ":path",
        ":stringpiece",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "random_test",
    srcs = ["random_test.cc"],
    deps = [
        ":random",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "tstring_test",
    size = "small",
    srcs = ["tstring_test.cc"],
    deps = [
        ":cord",
        ":platform",
        ":stringpiece",
        ":tstring",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    deps = [
        "@local_xla//xla/tsl/platform:test_main",
    ],
    alwayslink = 1,
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "threadpool_options",
    hdrs = ["threadpool_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_xla//xla/tsl/platform:threadpool_options",
    ],
)

cc_library(
    name = "unbounded_work_queue",
    hdrs = ["unbounded_work_queue.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
    ] + tf_platform_deps("unbounded_work_queue"),
)

tsl_cc_test(
    name = "unbounded_work_queue_test",
    srcs = ["unbounded_work_queue_test.cc"],
    deps = [
        ":blocking_counter",
        ":random",
        ":unbounded_work_queue",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "context",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["context.h"],
    deps = tf_platform_deps("context"),
)

cc_library(
    name = "load_library",
    textual_hdrs = ["load_library.h"],
    deps = [
        "@local_xla//xla/tsl/platform:status",
    ] + tf_windows_aware_platform_deps("load_library"),
)

cc_library(
    name = "abi",
    srcs = ["abi.cc"],
    hdrs = ["abi.h"],
    deps = [
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "refcount",
    hdrs = ["refcount.h"],
    deps = [
        ":mutex",
        ":thread_annotations",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "null_file_system",
    hdrs = ["null_file_system.h"],
    deps = [
        "@local_xla//xla/tsl/platform:env",
    ],
)

tsl_cc_test(
    name = "numa_test",
    size = "small",
    srcs = ["numa_test.cc"],
    tags = [
        # This test will not pass unless it has access to all NUMA nodes
        # on the executing machine.
        "manual",
        "notap",
    ],
    deps = [
        ":platform_port",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "setround_test",
    size = "small",
    srcs = ["setround_test.cc"],
    tags = ["noclang"],
    deps = [
        ":setround",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "refcount_test",
    size = "small",
    srcs = [
        "refcount_test.cc",
    ],
    deps = [
        ":refcount",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "integral_types_test",
    size = "small",
    srcs = [
        "integral_types_test.cc",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "mutex_test",
    size = "small",
    srcs = [
        "mutex_test.cc",
    ],
    deps = [
        ":mutex",
        ":net",
        ":platform_port",
        ":scanner",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":stringprintf",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:env_time",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "net_test",
    size = "small",
    srcs = [
        "net_test.cc",
    ],
    deps = [
        ":net",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "port_test",
    size = "small",
    srcs = [
        "port_test.cc",
    ],
    tags = [
        "notap",  #TODO(b/245510532) : disabled due to flakiness.
    ],
    deps = [
        ":mutex",
        ":platform_port",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "scanner_test",
    size = "small",
    srcs = [
        "scanner_test.cc",
    ],
    deps = [
        ":scanner",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "str_util_test",
    size = "small",
    srcs = [
        "str_util_test.cc",
    ],
    deps = [
        ":str_util",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "strcat_test",
    size = "small",
    srcs = [
        "strcat_test.cc",
    ],
    deps = [
        ":bfloat16",
        ":strcat",
        ":stringprintf",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tsl_cc_test(
    name = "stringpiece_test",
    size = "small",
    srcs = [
        "stringpiece_test.cc",
    ],
    deps = [
        ":stringpiece",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "stringprintf_test",
    size = "small",
    srcs = [
        "stringprintf_test.cc",
    ],
    deps = [
        ":stringprintf",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "numbers_test",
    size = "small",
    srcs = [
        "numbers_test.cc",
    ],
    deps = [
        ":numbers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "retrying_utils",
    srcs = [
        "retrying_utils.cc",
    ],
    hdrs = [
        "retrying_utils.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":random",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "retrying_file_system",
    hdrs = [
        "retrying_file_system.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":random",
        ":retrying_utils",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "retrying_file_system_test",
    size = "small",
    srcs = ["retrying_file_system_test.cc"],
    deps = [
        ":retrying_file_system",
        ":str_util",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:test",
    ],
)

tsl_cc_test(
    name = "retrying_utils_test",
    size = "small",
    srcs = ["retrying_utils_test.cc"],
    deps = [
        ":retrying_utils",
        ":str_util",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:test",
    ],
)
