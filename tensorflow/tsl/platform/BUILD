# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_cc_test",
    "cuda_library",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "rocm_copts",
)
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/tsl:tsl.bzl",
    "if_not_fuchsia",
    "tsl_copts",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tf_cuda_libdevice_path_deps",
    "tf_fingerprint_deps",
    "tf_google_mobile_srcs_no_runtime",
    "tf_logging_deps",
    "tf_platform_alias",
    "tf_platform_deps",
    "tf_protobuf_compiler_deps",
    "tf_resource_deps",
    "tf_stream_executor_deps",
    "tf_testing_deps",
    "tf_windows_aware_platform_deps",
    "tsl_cc_test",
    "tsl_protobuf_deps",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "@bazel_skylib//:bzl_library.bzl",
    "bzl_library",
)
load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    [
        "mutex.h",
        "load_library.h",
        "stringpiece_test.cc",
    ],
    visibility = [
        "//tensorflow/core/platform:__subpackages__",
        "//tensorflow/tsl:__subpackages__",
    ],
)

cc_library(
    name = "base64",
    srcs = ["base64.cc"],
    hdrs = ["base64.h"],
    deps = [
        ":errors",
        ":status",
        ":stringpiece",
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
        ":types",
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
        ":platform",
    ] + tf_platform_deps("dynamic_annotations"),
)

cc_library(
    name = "gif",
    hdrs = ["gif.h"],
    deps = [
        "@gif",
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
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
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
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "statusor",
    srcs = [
        "statusor.cc",
        "statusor_internals.h",
    ],
    hdrs = ["statusor.h"],
    deps = [
        ":errors",
        ":logging",
        ":macros",
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
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
    deps = tf_testing_deps("test") + [
        ":macros",
        ":platform",
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
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
)

filegroup(
    name = "android_test_srcs",
    testonly = 1,
    srcs = [
        "test.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
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
        ":stringpiece",
    ],
)

filegroup(
    name = "ctstring",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
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
        "dynamic_annotations.h",
        "env.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "fingerprint.h",
        "init_main.h",
        "logger.h",
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
        "//tensorflow/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/tsl/platform/profile_utils:clock_cycle_profiler.h",
        "//tensorflow/tsl/platform/profile_utils:cpu_utils.h",
        "//tensorflow/tsl/platform/profile_utils:i_cpu_utils_helper.h",
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
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "abi.h",
        "abi.cc",
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
        "regexp.h",
        "refcount.h",
        "resource.h",
        "scanner.cc",
        "scanner.h",
        "setround.cc",
        "setround.h",
        "snappy.h",
        "stacktrace.h",
        "status.cc",
        "status.h",
        "statusor.cc",
        "statusor.h",
        "statusor_internals.h",
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
    ] + select({
        "//tensorflow/tsl:fuchsia": tf_google_mobile_srcs_no_runtime(),
        "//conditions:default": [
            "//tensorflow/tsl/platform/default:mobile_srcs_no_runtime",
            "file_system_helper.cc",
            "tracing.cc",
        ],
    }),
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "fingerprint.h",
        "notification.h",
        "png.h",
        "random.cc",
        "random.h",
        "test_benchmark.h",
        "threadpool_options.h",
        "unbounded_work_queue.h",
        "//tensorflow/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.cc",
        "//tensorflow/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/tsl/platform/profile_utils:cpu_utils.cc",
        "//tensorflow/tsl/platform/profile_utils:cpu_utils.h",
        "//tensorflow/tsl/platform/profile_utils:i_cpu_utils_helper.h",
    ] + if_not_fuchsia([
        "subprocess.h",
    ]),
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
)

filegroup(
    name = "gif_hdrs",
    srcs = [
        "gif.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_internal_headers",
    srcs = glob(
        [
            "*.h",
        ],
        exclude = [
            "dynamic_annotations.h",
            "gif.h",
            "png.h",
            "jpeg.h",
        ],
    ) + [
        "//tensorflow/tsl/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/tsl/platform/profile_utils:cpu_utils.h",
        "//tensorflow/tsl/platform/profile_utils:i_cpu_utils_helper.h",
        "//tensorflow/tsl/platform/profile_utils:clock_cycle_profiler.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

exports_files(
    [
        "context.h",
        "host_info.h",
        "human_readable_json.h",
        "bfloat16.h",
        "cpu_info.h",
        "cpu_info.cc",
        "crash_analysis.h",
        "cuda_libdevice_path.h",
        "demangle.h",
        "env.cc",
        "env.h",
        "env_time.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.cc",
        "file_system_helper.h",
        "init_main.h",
        "logging.h",
        "mem.h",
        "net.h",
        "numa.h",
        "ram_file_system.h",
        "resource_loader.h",
        "resource.h",
        "snappy.h",
        "stacktrace_handler.h",
        "cuda_libdevice_path.h",
        "test.h",
        "bfloat16.h",
        "logging.h",
        "rocm_rocdl_path.h",
        "subprocess.h",
        "threadpool.cc",
        "threadpool.h",
        "tracing.h",
        "tracing.cc",
    ],
    visibility = [
        ":__subpackages__",
        "//tensorflow:__subpackages__",
    ],
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
        "init_main.h",
        "macros.h",
        "platform.h",
        "protobuf.h",
        "stringpiece.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
    ],
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
        "tracing.h",
        "unbounded_work_queue.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
)

filegroup(
    name = "jpeg_hdrs",
    srcs = [
        "jpeg.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
    ],
)

cc_library(
    name = "jpeg",
    hdrs = ["jpeg.h"],
    deps = [
        "@libjpeg_turbo//:jpeg",
    ],
)

filegroup(
    name = "tflite_portable_logging_hdrs",
    srcs = [
        "bfloat16.h",
        "macros.h",
        "platform.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
    ],
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
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
)

filegroup(
    name = "gif_internal_hdrs",
    srcs = [
        "bfloat16.h",
        "dynamic_annotations.h",
        "macros.h",
        "mem.h",
        "platform.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
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
    textual_hdrs = ["resource_loader.h"],
    deps = tf_testing_deps("resource_loader"),
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
    ],
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
        ":platform",
        "@com_google_absl//absl/strings:cord",
    ] + tf_platform_deps("cord"),
)

cc_library(
    name = "threadpool_interface",
    hdrs = ["threadpool_interface.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":mutex",
        ":types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        ":bfloat16",
        ":float8",
        ":tstring",
    ] + tf_platform_deps("types"),
)

cc_library(
    name = "build_test",
    testonly = 1,
    visibility = [
        "//tensorflow/core/platform:__pkg__",
    ],
    deps = [
        ":byte_order",
        ":fingerprint",
        ":gif",
        ":jpeg",
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
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "float8",
    hdrs = ["float8.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//third_party/eigen3",
        "@com_google_absl//absl/numeric:bits",
    ],
)

cuda_library(
    name = "float8_test_lib",
    testonly = 1,
    srcs = ["float8_test.cu.cc"],
    copts = rocm_copts(),
    deps = [
        ":float8",
        ":test",
        ":test_main",
        "//third_party/eigen3",
    ],
)

cuda_cc_test(
    name = "float8_test",
    deps = [":float8_test_lib"],
)

cc_library(
    name = "cuda",
    deps = [
        "//tensorflow/tsl/platform/default/build_config:cuda",
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
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        "env",
        ":logging",
        ":protobuf",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
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
    name = "prefetch",
    hdrs = ["prefetch.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
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
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
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
    name = "cuda_libdevice_path",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["cuda_libdevice_path.h"],
    deps = tf_cuda_libdevice_path_deps(),
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
    srcs = ["tensor_float_32_utils.h"],
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
        ":path",
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
        "//tensorflow/tsl:windows": [],
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
        ":status",
        ":test",
        ":test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
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
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
    ] + tf_platform_deps("notification"),
)

cc_library(
    name = "png",
    hdrs = ["png.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        "@png",
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
        "//tensorflow/tsl/platform/testdata:test_echo",
        "//tensorflow/tsl/platform/testdata:test_echo_argv_1",
        "//tensorflow/tsl/platform/testdata:test_noop",
        "//tensorflow/tsl/platform/testdata:test_stderr",
    ],
    deps = [
        ":path",
        ":strcat",
        ":subprocess",
        ":test",
        ":test_main",
        "//tensorflow/tsl/lib/core:status_test_util",
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
        ":test",
        ":test_main",
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

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
    deps = tf_platform_alias("rules_cc_bzl"),
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
        ":random",
        ":status",
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
        "//tensorflow/tsl/lib/core:status_test_util",
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
        "//tensorflow/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "static_threadlocal",
    hdrs = [
        "static_threadlocal.h",
    ],
)
