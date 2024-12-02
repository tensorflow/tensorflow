# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load(
    "@local_xla//xla/tsl:tsl.bzl",
    "if_not_fuchsia",
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
    "tf_logging_deps",
    "tf_platform_deps",
    "tf_protobuf_compiler_deps",
    "tf_resource_deps",
    "tf_stream_executor_deps",
    "tf_windows_aware_platform_deps",
    "tsl_cc_test",
    "tsl_grpc_credentials_deps",
    "tsl_protobuf_deps",
)
load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_static")
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
        ":errors",
        ":macros",
        ":status",
        ":stringpiece",
        ":types",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "blocking_counter",
    hdrs = ["blocking_counter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":logging",
        ":mutex",
    ],
)

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "coding",
    srcs = ["coding.cc"],
    hdrs = ["coding.h"],
    deps = [
        ":byte_order",
        ":stringpiece",
        ":tstring",
        ":types",
    ],
)

tsl_cc_test(
    name = "cpu_info_test",
    size = "small",
    srcs = ["cpu_info_test.cc"],
    deps = [
        ":platform_port",
        ":test",
        ":test_main",
    ],
)

cc_library(
    name = "criticality",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["criticality.h"],
    deps = tf_platform_deps("criticality"),
)

tsl_cc_test(
    name = "criticality_test",
    size = "small",
    srcs = [
        "criticality_test.cc",
    ],
    deps = [
        ":criticality",
        ":test",
        ":test_main",
    ],
)

cc_library(
    name = "denormal",
    srcs = ["denormal.cc"],
    hdrs = ["denormal.h"],
    deps = [
        ":macros",
        ":platform",
        ":platform_port",
    ],
)

tsl_cc_test(
    name = "denormal_test",
    size = "small",
    srcs = ["denormal_test.cc"],
    deps = [
        ":denormal",
        ":test",
        ":test_main",
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
    name = "threadpool_async_executor",
    hdrs = ["threadpool_async_executor.h"],
    deps = [
        ":env",
        "@local_xla//xla/tsl/concurrency:async_value",
    ],
)

tsl_cc_test(
    name = "threadpool_async_executor_test",
    srcs = ["threadpool_async_executor_test.cc"],
    deps = [
        ":env",
        ":env_impl",
        ":test",
        ":test_main",
        ":threadpool_async_executor",
        "@com_google_absl//absl/synchronization",
    ],
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
        ":logging",
        ":macros",
        ":status",
        ":str_util",
        ":strcat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "dynamic_annotations",
    hdrs = ["dynamic_annotations.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:dynamic_annotations",
    ],
)

cc_library(
    name = "mutex",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["mutex.h"],
    deps = tf_platform_deps("mutex"),
)

cc_library(
    name = "numbers",
    srcs = ["numbers.cc"],
    hdrs = ["numbers.h"],
    deps = [
        ":logging",
        ":macros",
        ":str_util",
        ":stringpiece",
        ":stringprintf",
        ":types",
        "@double_conversion//:double-conversion",
    ],
)

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        ":logging",
        ":mutex",
        ":scanner",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":types",
        "@com_google_absl//absl/algorithm:container",
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
        ":types",
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
    name = "resource",
    textual_hdrs = ["resource.h"],
    deps = [
        ":stringpiece",
    ] + tf_resource_deps(),
)

cc_library(
    name = "stack_frame",
    hdrs = ["stack_frame.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        ":logging",
        ":macros",
        ":mutex",
        ":platform",
        ":stack_frame",
        ":stacktrace",
        ":str_util",
        ":strcat",
        ":stringprintf",
        ":types",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ] + tf_platform_deps("status"),
)

cc_library(
    name = "status_to_from_proto",
    srcs = [
        "status_to_from_proto.cc",
    ],
    hdrs = ["status_to_from_proto.h"],
    deps = [
        ":status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/tsl/protobuf:status_proto_cc",
    ] + tf_platform_deps("status"),
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    srcs = ["status_matchers.cc"],
    hdrs = ["status_matchers.h"],
    deps = [
        ":status",
        ":statusor",
        ":test",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "statusor",
    hdrs = ["statusor.h"],
    deps = [
        ":errors",
        ":logging",
        ":macros",
        ":platform",
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ] + tf_platform_deps("statusor"),
)

cc_library(
    name = "thread_annotations",
    hdrs = ["thread_annotations.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "test",
    testonly = True,
    srcs = ["test.cc"],
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["test.h"],
    deps = [
        ":logging",
        ":macros",
        ":net",
        ":path",
        ":platform",
        ":types",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        "@com_google_benchmark//:benchmark",
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
        "crash_analysis.h",
        "criticality.h",
        "dynamic_annotations.h",
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
        "prefetch.h",
        "protobuf.h",
        "ram_file_system.h",
        "random.h",
        "resource.h",
        "stack_frame.h",
        "stacktrace.h",
        "stacktrace_handler.h",
        "statusor.h",
        "str_util.h",
        "stringpiece.h",
        "stringprintf.h",
        "subprocess.h",
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
        "byte_order.h",
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
        "byte_order.h",
        "cpu_info.h",
        "dynamic_annotations.h",
        "macros.h",
        "platform.h",
        "prefetch.h",
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
        "byte_order.h",
        "casts.h",
        "coding.cc",
        "coding.h",
        "context.h",
        "cpu_info.cc",
        "cpu_info.h",
        "criticality.h",
        "demangle.h",
        "denormal.cc",
        "denormal.h",
        "dynamic_annotations.h",
        "env.cc",
        "env.h",
        "env_time.h",
        "errors.cc",
        "errors.h",
        "file_statistics.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.h",
        "hash.cc",
        "hash.h",
        "host_info.h",
        "init_main.h",
        "intrusive_ptr.h",
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
        "prefetch.h",
        "protobuf.cc",
        "protobuf.h",
        "protobuf_util.cc",
        "ram_file_system.h",
        "raw_coding.h",
        "refcount.h",
        "regexp.h",
        "resource.h",
        "scanner.cc",
        "scanner.h",
        "setround.cc",
        "setround.h",
        "snappy.h",
        "stacktrace.h",
        "status.cc",
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
        "threadpool.cc",
        "threadpool.h",
        "threadpool_interface.h",
        "tracing.h",
        "@local_xla//xla/tsl/platform:mobile_srcs_no_runtime",
    ] + select({
        "@local_xla//xla/tsl:fuchsia": tf_google_mobile_srcs_no_runtime(),
        "//conditions:default": [
            "file_system_helper.cc",
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
    ] + if_not_fuchsia([
        "subprocess.h",
    ]),
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
        "crash_analysis.h",
        "criticality.h",
        "cuda_root_path.h",
        "demangle.h",
        "env.cc",
        "env.h",
        "env_time.h",
        "error_logging.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.cc",
        "file_system_helper.h",
        "grpc_credentials.h",
        "host_info.h",
        "human_readable_json.h",
        "init_main.h",
        "logging.h",
        "mem.h",
        "net.h",
        "numa.h",
        "ram_file_system.h",
        "resource.h",
        "resource_loader.h",
        "rocm_rocdl_path.h",
        "snappy.h",
        "stacktrace_handler.h",
        "subprocess.h",
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

cc_library(
    name = "intrusive_ptr",
    hdrs = ["intrusive_ptr.h"],
    deps = [],
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
        "blocking_counter.h",
        "demangle.h",
        "denormal.h",
        "host_info.h",
        "intrusive_ptr.h",
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
        "dynamic_annotations.h",
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
        "dynamic_annotations.h",
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
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "dynamic_annotations.h",
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
    deps = tf_windows_aware_platform_deps("platform_port"),
)

cc_library(
    name = "platform_strings",
    srcs = [
        "platform_strings_computed.h",
    ],
)

cc_library(
    name = "protobuf_compiler",
    hdrs = ["protobuf_compiler.h"],
    deps = tf_protobuf_compiler_deps(),
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        ":mutex",
        ":types",
    ],
)

cc_library(
    name = "resource_loader",
    testonly = 1,
    srcs = ["resource_loader.cc"],
    textual_hdrs = ["resource_loader.h"],
    deps = [
        ":logging",
        ":path",
        ":platform",
        ":test",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    textual_hdrs = ["rocm_rocdl_path.h"],
    deps = tf_platform_deps("rocm_rocdl_path"),
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
    tags = ["no_windows"],
    deps = [
        ":logging",
        ":stacktrace",
        ":stacktrace_handler",
        ":test",
        ":test_main",
    ],
)

cc_library(
    name = "str_util",
    srcs = ["str_util.cc"],
    hdrs = ["str_util.h"],
    deps = [
        ":logging",
        ":macros",
        ":stringpiece",
        ":types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "strcat",
    srcs = ["strcat.cc"],
    hdrs = ["strcat.h"],
    deps = [
        ":logging",
        ":macros",
        ":numbers",
        ":stringpiece",
        ":types",
        "@com_google_absl//absl/meta:type_traits",
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
    name = "crash_analysis",
    hdrs = ["crash_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
    ] + tf_platform_deps("crash_analysis"),
)

cc_library(
    name = "stringprintf",
    srcs = ["stringprintf.cc"],
    hdrs = ["stringprintf.h"],
    deps = [
        ":macros",
        ":types",
    ],
)

cc_library(
    name = "subprocess",
    textual_hdrs = [
        "subprocess.h",
    ],
    deps = tf_windows_aware_platform_deps("subprocess"),
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
        ":mutex",
        ":types",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bfloat16",
        ":ml_dtypes",
        ":platform",
        ":tstring",
    ] + tf_platform_deps("types"),
)

cc_library(
    name = "build_test",
    testonly = 1,
    visibility = internal_visibility([
        "//tensorflow/core/platform:__pkg__",
    ]),
    deps = [
        ":byte_order",
        ":fingerprint",
        ":macros",
        ":net",
        ":platform",
        ":platform_port",
        ":platform_strings",
        ":stacktrace_handler",
        ":stringpiece",
        ":threadpool_interface",
    ],
)

cc_library(
    name = "bfloat16",
    hdrs = ["bfloat16.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":byte_order",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "ml_dtypes",
    hdrs = ["ml_dtypes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@ml_dtypes//:float8",
        "@ml_dtypes//:intn",
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
    deps = tf_logging_deps(),
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
    name = "grpc_credentials",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["grpc_credentials.h"],
    deps = tsl_grpc_credentials_deps(),
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/base:prefetch",
    ],
)

cc_library(
    name = "hash",
    srcs = ["hash.cc"],
    hdrs = ["hash.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":macros",
        ":raw_coding",
        ":stringpiece",
        ":types",
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
        ":byte_order",
        ":types",
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
        ":logging",
        ":macros",
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
    tags = ["no_windows"],
    deps = [
        ":logging",
        ":stacktrace",
        ":test",
        ":test_main",
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
        ":types",
    ],
)

cc_library(
    name = "fingerprint",
    hdrs = ["fingerprint.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        ":stringpiece",
        ":types",
    ] + tf_fingerprint_deps(),
)

tsl_cc_test(
    name = "fingerprint_test",
    size = "small",
    srcs = ["fingerprint_test.cc"],
    deps = [
        ":fingerprint",
        ":test",
        ":test_main",
        ":types",
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
        ":macros",
        ":str_util",
        ":stringpiece",
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
        ":test",
        ":test_main",
        ":tstring",
    ],
)

tsl_cc_test(
    name = "hash_test",
    size = "small",
    srcs = ["hash_test.cc"],
    deps = [
        ":hash",
        ":logging",
        ":test",
        ":test_benchmark",
        ":test_main",
    ],
)

tsl_cc_test(
    name = "path_test",
    size = "small",
    srcs = ["path_test.cc"],
    deps = [
        ":env",
        ":env_impl",
        ":path",
        ":stringpiece",
        ":test",
        ":test_main",
    ],
)

tsl_cc_test(
    name = "random_test",
    srcs = ["random_test.cc"],
    deps = [
        ":random",
        ":test",
        ":test_main",
        ":types",
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
        ":test",
        ":test_main",
        ":tstring",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    copts = tsl_copts(),
    linkopts = select({
        "@local_xla//xla/tsl:windows": [],
        "//conditions:default": ["-lm"],
    }),
    deps = [
        ":platform",
        ":stacktrace_handler",
        ":test",
        ":test_benchmark",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tsl_cc_test(
    name = "status_test",
    size = "small",
    srcs = ["status_test.cc"],
    deps = [
        ":errors",
        ":stack_frame",
        ":status",
        ":status_matchers",
        ":status_to_from_proto",
        ":test",
        ":test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/tsl/protobuf:status_proto_cc",
    ],
)

tsl_cc_test(
    name = "statusor_test",
    size = "small",
    srcs = ["statusor_test.cc"],
    deps = [
        ":errors",
        ":macros",
        ":statusor",
        ":test",
        ":test_benchmark",
        ":test_main",
        "@com_google_absl//absl/base:config",
    ],
)

tsl_cc_test(
    name = "status_matchers_test",
    size = "small",
    srcs = ["status_matchers_test.cc"],
    deps = [
        ":errors",
        ":status",
        ":status_matchers",
        ":statusor",
        ":test",
        ":test_main",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
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
        ":threadpool_interface",
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
        ":env",
        ":env_impl",
        ":random",
        ":test",
        ":test_main",
        ":unbounded_work_queue",
        "@com_google_absl//absl/memory",
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
        ":status",
    ] + tf_windows_aware_platform_deps("load_library"),
)

cc_library(
    name = "abi",
    srcs = ["abi.cc"],
    hdrs = ["abi.h"],
    deps = [
        ":types",
    ],
)

cc_library(
    name = "refcount",
    hdrs = ["refcount.h"],
    deps = [
        ":logging",
        ":mutex",
        ":thread_annotations",
    ],
)

cc_library(
    name = "null_file_system",
    hdrs = ["null_file_system.h"],
    deps = [
        ":env",
    ],
)

tsl_cc_test(
    name = "subprocess_test",
    size = "small",
    srcs = ["subprocess_test.cc"],
    data = [
        "//tsl/platform/testdata:test_echo",
        "//tsl/platform/testdata:test_echo_argv_1",
        "//tsl/platform/testdata:test_noop",
        "//tsl/platform/testdata:test_stderr",
    ],
    tags = ["no_oss"],  # TODO(b/327036247): revisit after this moves to XLA
    deps = [
        ":path",
        ":strcat",
        ":subprocess",
        ":test",
        ":test_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tsl_cc_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.cc"],
    deps = [
        ":errors",
        ":test",
        ":test_main",
        "@com_google_absl//absl/status",
    ],
)

tsl_cc_test(
    name = "intrusive_ptr_test",
    size = "small",
    srcs = [
        "intrusive_ptr_test.cc",
    ],
    deps = [
        ":intrusive_ptr",
        ":refcount",
        ":test",
        ":test_main",
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
        ":logging",
        ":test",
        ":test_main",
    ],
)

tsl_cc_test(
    name = "setround_test",
    size = "small",
    srcs = ["setround_test.cc"],
    tags = ["noclang"],
    deps = [
        ":setround",
        ":test",
        ":test_main",
    ],
)

tsl_cc_test(
    name = "refcount_test",
    size = "small",
    srcs = [
        "refcount_test.cc",
    ],
    deps = [
        ":env",
        ":env_impl",
        ":refcount",
        ":test",
        ":test_main",
    ],
)

tsl_cc_test(
    name = "integral_types_test",
    size = "small",
    srcs = [
        "integral_types_test.cc",
    ],
    deps = [
        ":test",
        ":test_main",
        ":types",
    ],
)

tsl_cc_test(
    name = "logging_test",
    size = "small",
    srcs = [
        "logging_test.cc",
    ],
    deps = [
        ":logging",
        ":path",
        ":stacktrace_handler",
        ":statusor",
        ":test",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tsl_cc_test(
    name = "mutex_test",
    size = "small",
    srcs = [
        "mutex_test.cc",
    ],
    deps = [
        ":env",
        ":env_impl",
        ":env_time",
        ":logging",
        ":mutex",
        ":net",
        ":platform_port",
        ":scanner",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":stringprintf",
        ":test",
        ":test_main",
        ":types",
    ],
)

tsl_cc_test(
    name = "net_test",
    size = "small",
    srcs = [
        "net_test.cc",
    ],
    deps = [
        ":logging",
        ":net",
        ":test",
        ":test_main",
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
        ":env",
        ":env_impl",
        ":env_time",
        ":mutex",
        ":platform_port",
        ":test",
        ":test_main",
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
        ":test",
        ":test_main",
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
        ":test",
        ":test_main",
    ],
)

tsl_cc_test(
    name = "strcat_test",
    size = "small",
    srcs = [
        "strcat_test.cc",
    ],
    deps = [
        ":strcat",
        ":stringprintf",
        ":test",
        ":test_main",
        ":types",
        "@com_google_absl//absl/strings",
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
        ":test",
        ":test_main",
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
        ":test",
        ":test_main",
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
        ":test",
        ":test_main",
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
        ":env",
        ":errors",
        ":logging",
        ":random",
        ":status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "retrying_file_system",
    hdrs = [
        "retrying_file_system.h",
    ],
    copts = tsl_copts(),
    deps = [
        ":env",
        ":errors",
        ":random",
        ":retrying_utils",
        ":status",
    ],
)

tsl_cc_test(
    name = "retrying_file_system_test",
    size = "small",
    srcs = ["retrying_file_system_test.cc"],
    deps = [
        ":env_impl",
        ":retrying_file_system",
        ":str_util",
        ":test",
        ":test_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tsl_cc_test(
    name = "retrying_utils_test",
    size = "small",
    srcs = ["retrying_utils_test.cc"],
    deps = [
        ":env",
        ":env_impl",
        ":errors",
        ":retrying_utils",
        ":str_util",
        ":test",
        ":test_main",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)
