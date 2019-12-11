# Description:
#   TensorFlow Base libraries.
#   This package contains the following libraries:
#     - Platform dependent libraries that require different implementations
#       across different OSs or environments.
#     - STL replacement libraries rest of TensorFlow should depend on.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to any TensorFlow code outside this package.
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_lib_hdrs",
    "tf_additional_tensor_coding_deps",
    "tf_additional_test_srcs",
    "tf_fingerprint_deps",
    "tf_protobuf_compiler_deps",
    "tf_protobuf_deps",
)
load(
    "//tensorflow/core/platform/default:build_refactor.bzl",
    "tf_instantiate_platform_libraries",
    "tf_legacy_srcs_no_runtime_google",
    "tf_logging_deps",
    "tf_mobile_aware_deps",
    "tf_monitoring_deps",
    "tf_platform_helper_deps",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_not_android",
    "tf_copts",  # @unused
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Note(bmzhao): These are temporary BUILD rule helpers for
# platform-specific library dependencies. These temporarily
# exist to allow us incrementally add granular targets
# to tf/core/platform/BUILD. We will remove these helpers,
# after we finish refactoring tf/core/platform/BUILD.
tf_instantiate_platform_libraries(names = [
    "context",
    "cord",
    "cuda_libdevice_path",
    "dynamic_annotations",
    "env",
    "env_time",
    "human_readable_json",
    "logging",
    "load_library",
    "monitoring",
    "mutex",
    "net",
    "notification",
    "platform_port",
    "rocm_rocdl_path",
    "stacktrace",
    "stacktrace_handler",
    "strong_hash",
    "subprocess",
    "test",
    "test_benchmark",
    "tracing",
    "types",
    "unbounded_work_queue",
    "wide_char",
])

cc_library(
    name = "abi",
    srcs = ["abi.cc"],
    hdrs = ["abi.h"],
    deps = [":types"],
)

cc_library(
    name = "base64",
    srcs = ["base64.cc"],
    hdrs = ["base64.h"],
    deps = [
        ":errors",
        ":status",
    ],
)

cc_library(
    name = "blocking_counter",
    hdrs = ["blocking_counter.h"],
    deps = [
        ":logging",
        ":mutex",
    ],
)

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
)

cc_library(
    name = "coding",
    srcs = ["coding.cc"],
    hdrs = ["coding.h"],
    deps = [
        ":byte_order",
        ":raw_coding",
        ":stringpiece",
        ":types",
    ],
)

cc_library(
    name = "context",
    textual_hdrs = ["context.h"],
    # See call to tf_instantiate_platform_libraries() above.
    deps = tf_mobile_aware_deps("context_impl"),
)

cc_library(
    name = "cord",
    hdrs = ["cord.h"],
    deps = [
        ":platform",
    ] + tf_mobile_aware_deps("cord_impl"),
)

cc_library(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.cc"],
    hdrs = ["cpu_feature_guard.h"],
    deps = [
        ":byte_order",
        ":logging",
        ":platform_port",
    ],
)

cc_library(
    name = "cuda",
    hdrs = ["cuda.h"],
    deps = [
        ":platform",
        "//tensorflow/stream_executor/cuda:cuda_activation_header",
    ],
)

cc_library(
    name = "cuda_libdevice_path",
    textual_hdrs = ["cuda_libdevice_path.h"],
    deps = tf_mobile_aware_deps("cuda_libdevice_path_impl"),
)

cc_library(
    name = "denormal",
    srcs = ["denormal.cc"],
    hdrs = ["denormal.h"],
    deps = [
        ":byte_order",
        ":logging",
        ":macros",
        ":platform",
        ":platform_port",
    ],
)

cc_library(
    name = "dynamic_annotations",
    hdrs = ["dynamic_annotations.h"],
    deps = [
        ":platform",
    ] + tf_mobile_aware_deps("dynamic_annotations_impl"),
)

cc_library(
    name = "env",
    textual_hdrs = [
        "env.h",
        "file_system.h",
        "file_system_helper.h",
        "threadpool.h",
    ],
    deps = tf_platform_helper_deps("env_impl"),
)

cc_library(
    name = "env_time",
    textual_hdrs = ["env_time.h"],
    deps = tf_platform_helper_deps("env_time_impl"),
)

cc_library(
    name = "error",
    srcs = ["error.cc"],
    hdrs = ["error.h"],
    deps = [
        ":platform",
        ":status",
        ":strcat",
        ":types",
    ],
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    deps = [
        ":logging",
        ":macros",
        ":status",
        ":str_util",
        ":strcat",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_statistics",
    hdrs = ["file_statistics.h"],
    deps = [":types"],
)

cc_library(
    name = "fingerprint",
    hdrs = ["fingerprint.h"],
    deps = [
        ":stringpiece",
        ":types",
    ] + tf_fingerprint_deps(),
)

cc_library(
    name = "gif",
    hdrs = ["gif.h"],
    deps = [
        "@gif",
    ],
)

cc_library(
    name = "hash",
    srcs = ["hash.cc"],
    hdrs = ["hash.h"],
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
    deps = tf_mobile_aware_deps("human_readable_json_impl"),
)

cc_library(
    name = "jpeg",
    hdrs = ["jpeg.h"],
    deps = [
        "@libjpeg_turbo//:jpeg",
    ],
)

cc_library(
    name = "load_library",
    textual_hdrs = ["load_library.h"],
    deps = tf_platform_helper_deps("load_library_impl"),
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        ":env",
        ":logging",
        ":protobuf",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_library(
    name = "logging",
    textual_hdrs = ["logging.h"],
    deps = tf_logging_deps(),
)

cc_library(
    name = "monitoring",
    textual_hdrs = ["monitoring.h"],
    deps = tf_monitoring_deps(),
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "mutex",
    textual_hdrs = ["mutex.h"],
    deps = tf_mobile_aware_deps("mutex_impl"),
)

cc_library(
    name = "net",
    textual_hdrs = ["net.h"],
    deps = tf_platform_helper_deps("net_impl"),
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    deps = [
        ":platform",
    ] + tf_mobile_aware_deps("notification_impl"),
)

cc_library(
    name = "null_file_system",
    hdrs = ["null_file_system.h"],
    deps = [
        ":env",
    ],
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
    ],
    alwayslink = True,
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
)

cc_library(
    name = "platform_port",
    textual_hdrs = [
        "cpu_info.h",
        "demangle.h",
        "host_info.h",
        "init_main.h",
        "mem.h",
        "numa.h",
        "snappy.h",
    ],
    deps = tf_platform_helper_deps("platform_port_impl"),
)

cc_library(
    name = "platform_strings",
    srcs = [
        "platform_strings.cc",
        "platform_strings_computed.h",
    ],
    hdrs = ["platform_strings.h"],
)

cc_library(
    name = "png",
    hdrs = ["png.h"],
    deps = [
        ":platform",
        "@png",
    ],
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    deps = [":platform"],
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
    ] + tf_protobuf_deps(),
)

cc_library(
    name = "protobuf_compiler",
    hdrs = ["protobuf_compiler.h"],
    deps = tf_protobuf_compiler_deps(),
)

cc_library(
    name = "protobuf_internal",
    hdrs = ["protobuf_internal.h"],
    deps = [
        ":errors",
        ":platform",
        ":protobuf",
        ":types",
    ] + if_static(["@com_google_protobuf//:protobuf"]),
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
    name = "raw_coding",
    hdrs = ["raw_coding.h"],
    deps = [
        ":byte_order",
        ":types",
    ],
)

cc_library(
    name = "refcount",
    hdrs = ["refcount.h"],
    deps = [
        ":logging",
    ],
)

cc_library(
    name = "regexp",
    hdrs = ["regexp.h"],
    deps = [
        ":platform",
        ":types",
        "@com_google_absl//absl/strings",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    textual_hdrs = ["rocm_rocdl_path.h"],
    deps = tf_mobile_aware_deps("rocm_rocdl_path_impl"),
)

cc_library(
    name = "rocm",
    hdrs = if_rocm_is_configured(["rocm.h"]),
    deps = if_rocm_is_configured([
        ":platform",
        "//tensorflow/stream_executor/rocm:rocm_activation",
    ]),
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
    deps = [
        ":platform",
    ] + tf_platform_helper_deps("stacktrace_impl"),
)

filegroup(
    name = "stacktrace_handler_hdrs",
    srcs = [
        "stacktrace_handler.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "stacktrace_handler",
    textual_hdrs = ["stacktrace_handler.h"],
    deps = tf_platform_helper_deps("stacktrace_handler_impl"),
)

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        ":logging",
        ":macros",
        ":mutex",
        ":stacktrace",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":stringprintf",
        ":types",
        "//tensorflow/core:error_codes_proto_impl_cc",
        "@com_google_absl//absl/base",
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
    deps = [
        "@com_google_absl//absl/strings",
    ],
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
    name = "strong_hash",
    hdrs = ["strong_hash.h"],
    deps = [
        ":platform",
        ":types",
    ] + tf_mobile_aware_deps("strong_hash_impl"),
)

cc_library(
    name = "subprocess",
    textual_hdrs = [
        "subprocess.h",
    ],
    deps = tf_platform_helper_deps("subprocess_impl"),
)

cc_library(
    name = "tensor_coding",
    srcs = ["tensor_coding.cc"],
    hdrs = ["tensor_coding.h"],
    deps = [
        ":coding",
        ":platform",
        ":protobuf",
        ":refcount",
        ":stringpiece",
        ":strcat",
        ":types",
    ] + tf_additional_tensor_coding_deps(),
)

cc_library(
    name = "test",
    testonly = True,
    textual_hdrs = ["test.h"],
    deps = tf_mobile_aware_deps("test_impl"),
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    deps = [
        ":platform",
        ":test_benchmark_impl",
    ],
)

cc_library(
    name = "thread_annotations",
    hdrs = ["thread_annotations.h"],
)

cc_library(
    name = "threadpool_interface",
    hdrs = ["threadpool_interface.h"],
    deps = [
        ":mutex",
        ":types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "threadpool_options",
    hdrs = ["threadpool_options.h"],
    deps = [
        ":threadpool_interface",
    ],
)

cc_library(
    name = "tracing",
    textual_hdrs = ["tracing.h"],
    deps = tf_mobile_aware_deps("tracing_impl"),
)

cc_library(
    name = "tstring",
    hdrs = ["tstring.h"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        ":platform",
        ":tstring",
    ] + tf_mobile_aware_deps("types_impl"),
)

cc_library(
    name = "unbounded_work_queue",
    hdrs = ["unbounded_work_queue.h"],
    deps = [
        ":platform",
    ] + tf_mobile_aware_deps("unbounded_work_queue_impl"),
)

# This is a hacky, do-nothing, binary that makes it easy to verify ability to
# build, link, and in some cases run all of the libraries under platform.
# Realistically, most of this would be covered by tests but at this point
# keeping things building correctly is step one.
cc_binary(
    name = "build_test",
    testonly = 1,
    srcs = ["build_test.cc"],
    visibility = ["//visibility:private"],
    deps = [
        ":abi",
        ":byte_order",
        ":cord",
        ":cpu_feature_guard",
        ":denormal",
        ":dynamic_annotations",
        ":env_time",
        ":file_statistics",
        ":fingerprint",
        ":gif",
        ":jpeg",
        ":macros",
        ":mutex",
        ":net",
        ":numbers",
        ":platform",
        ":platform_port",
        ":platform_strings",
        ":png",
        ":prefetch",
        ":protobuf",
        ":scanner",
        ":setround",
        ":stacktrace",
        ":stacktrace_handler",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":stringprintf",
        ":subprocess",
        ":tensor_coding",
        ":test",
        ":thread_annotations",
        ":threadpool_interface",
        ":threadpool_options",
        ":tstring",
        ":types",
    ] + if_not_android([":rocm_rocdl_path"]),
)

# --------------------------------------------------------------------------
#     Below libraries are here only to make sure the legacy build rules
#     in tensorflow/core/BUILD are working!
#
#     DO NOT add any new dependencies on these rules!
#
# --------------------------------------------------------------------------

filegroup(
    name = "legacy_platform_lib_hdrs",
    srcs = tf_additional_lib_hdrs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

# These are the files in common between :legacy_srcs_no_runtime
# and :legacy_srcs_no_runtime_google
# These files as basically all the headers + cc files under tensorflow/core/platform,
# excluding any test sources, testing utilities, cuda, rocm, stream_executor,
# image headers (gif.h, jpeg.h, png.h), and certain translation units (
# env_time.cc, logging.cc, logger.cc, mutex.cc) that would cause collisions
# with :platform_base, a common dependency for downstream targets.
filegroup(
    name = "legacy_srcs_common",
    srcs = [
        "//tensorflow/core/platform:abi.cc",
        "//tensorflow/core/platform:abi.h",
        "//tensorflow/core/platform:base64.cc",
        "//tensorflow/core/platform:base64.h",
        "//tensorflow/core/platform:blocking_counter.h",
        "//tensorflow/core/platform:byte_order.h",
        "//tensorflow/core/platform:coding.cc",
        "//tensorflow/core/platform:coding.h",
        "//tensorflow/core/platform:context.h",
        "//tensorflow/core/platform:cord.h",
        "//tensorflow/core/platform:cpu_feature_guard.cc",
        "//tensorflow/core/platform:cpu_feature_guard.h",
        "//tensorflow/core/platform:cpu_info.cc",
        "//tensorflow/core/platform:cpu_info.h",
        "//tensorflow/core/platform:demangle.h",
        "//tensorflow/core/platform:denormal.cc",
        "//tensorflow/core/platform:denormal.h",
        "//tensorflow/core/platform:dynamic_annotations.h",
        "//tensorflow/core/platform:env.cc",
        "//tensorflow/core/platform:env.h",
        "//tensorflow/core/platform:env_time.h",
        "//tensorflow/core/platform:error.cc",
        "//tensorflow/core/platform:error.h",
        "//tensorflow/core/platform:errors.h",
        "//tensorflow/core/platform:file_statistics.h",
        "//tensorflow/core/platform:file_system.cc",
        "//tensorflow/core/platform:file_system.h",
        "//tensorflow/core/platform:file_system_helper.cc",
        "//tensorflow/core/platform:file_system_helper.h",
        "//tensorflow/core/platform:fingerprint.h",
        "//tensorflow/core/platform:hash.cc",
        "//tensorflow/core/platform:hash.h",
        "//tensorflow/core/platform:host_info.h",
        "//tensorflow/core/platform:human_readable_json.h",
        "//tensorflow/core/platform:init_main.h",
        "//tensorflow/core/platform:load_library.h",
        "//tensorflow/core/platform:logger.h",
        "//tensorflow/core/platform:logging.h",
        "//tensorflow/core/platform:macros.h",
        "//tensorflow/core/platform:mem.h",
        "//tensorflow/core/platform:monitoring.h",
        "//tensorflow/core/platform:mutex.h",
        "//tensorflow/core/platform:net.h",
        "//tensorflow/core/platform:notification.h",
        "//tensorflow/core/platform:null_file_system.h",
        "//tensorflow/core/platform:numa.h",
        "//tensorflow/core/platform:numbers.cc",
        "//tensorflow/core/platform:numbers.h",
        "//tensorflow/core/platform:path.cc",
        "//tensorflow/core/platform:path.h",
        "//tensorflow/core/platform:platform.h",
        "//tensorflow/core/platform:platform_strings.cc",
        "//tensorflow/core/platform:platform_strings.h",
        "//tensorflow/core/platform:platform_strings_computed.h",
        "//tensorflow/core/platform:prefetch.h",
        "//tensorflow/core/platform:profile_utils/android_armv7a_cpu_utils_helper.cc",
        "//tensorflow/core/platform:profile_utils/android_armv7a_cpu_utils_helper.h",
        "//tensorflow/core/platform:profile_utils/clock_cycle_profiler.cc",
        "//tensorflow/core/platform:profile_utils/clock_cycle_profiler.h",
        "//tensorflow/core/platform:profile_utils/cpu_utils.cc",
        "//tensorflow/core/platform:profile_utils/cpu_utils.h",
        "//tensorflow/core/platform:profile_utils/i_cpu_utils_helper.h",
        "//tensorflow/core/platform:protobuf.cc",
        "//tensorflow/core/platform:protobuf.h",
        "//tensorflow/core/platform:protobuf_compiler.h",
        "//tensorflow/core/platform:protobuf_internal.h",
        "//tensorflow/core/platform:protobuf_util.cc",
        "//tensorflow/core/platform:random.cc",
        "//tensorflow/core/platform:random.h",
        "//tensorflow/core/platform:raw_coding.h",
        "//tensorflow/core/platform:refcount.h",
        "//tensorflow/core/platform:regexp.h",
        "//tensorflow/core/platform:scanner.cc",
        "//tensorflow/core/platform:scanner.h",
        "//tensorflow/core/platform:setround.cc",
        "//tensorflow/core/platform:setround.h",
        "//tensorflow/core/platform:snappy.h",
        "//tensorflow/core/platform:stacktrace.h",
        "//tensorflow/core/platform:stacktrace_handler.h",
        "//tensorflow/core/platform:status.cc",
        "//tensorflow/core/platform:status.h",
        "//tensorflow/core/platform:str_util.cc",
        "//tensorflow/core/platform:str_util.h",
        "//tensorflow/core/platform:strcat.cc",
        "//tensorflow/core/platform:strcat.h",
        "//tensorflow/core/platform:stream_executor_no_cuda.h",
        "//tensorflow/core/platform:stringpiece.h",
        "//tensorflow/core/platform:stringprintf.cc",
        "//tensorflow/core/platform:stringprintf.h",
        "//tensorflow/core/platform:strong_hash.h",
        "//tensorflow/core/platform:subprocess.h",
        "//tensorflow/core/platform:tensor_coding.cc",
        "//tensorflow/core/platform:tensor_coding.h",
        "//tensorflow/core/platform:test_benchmark.h",
        "//tensorflow/core/platform:thread_annotations.h",
        "//tensorflow/core/platform:threadpool.cc",
        "//tensorflow/core/platform:threadpool.h",
        "//tensorflow/core/platform:threadpool_interface.h",
        "//tensorflow/core/platform:threadpool_options.h",
        "//tensorflow/core/platform:tracing.cc",
        "//tensorflow/core/platform:tracing.h",
        "//tensorflow/core/platform:tstring.h",
        "//tensorflow/core/platform:types.h",
        "//tensorflow/core/platform:unbounded_work_queue.h",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "legacy_srcs_no_runtime",
    srcs = [
        ":legacy_srcs_common",
        "//tensorflow/core/platform/default:context.h",
        "//tensorflow/core/platform/default:cord.h",
        "//tensorflow/core/platform/default:dynamic_annotations.h",
        "//tensorflow/core/platform/default:env.cc",
        "//tensorflow/core/platform/default:human_readable_json.cc",
        "//tensorflow/core/platform/default:integral_types.h",
        "//tensorflow/core/platform/default:load_library.cc",
        "//tensorflow/core/platform/default:logging.h",
        "//tensorflow/core/platform/default:monitoring.cc",
        "//tensorflow/core/platform/default:mutex.h",
        "//tensorflow/core/platform/default:mutex_data.h",
        "//tensorflow/core/platform/default:net.cc",
        "//tensorflow/core/platform/default:notification.h",
        "//tensorflow/core/platform/default:port.cc",
        "//tensorflow/core/platform/default:posix_file_system.cc",
        "//tensorflow/core/platform/default:posix_file_system.h",
        "//tensorflow/core/platform/default:stacktrace.h",
        "//tensorflow/core/platform/default:stacktrace_handler.cc",
        "//tensorflow/core/platform/default:strong_hash.h",
        "//tensorflow/core/platform/default:subprocess.cc",
        "//tensorflow/core/platform/default:subprocess.h",
        "//tensorflow/core/platform/default:tracing.cc",
        "//tensorflow/core/platform/default:tracing_impl.h",
        "//tensorflow/core/platform/default:unbounded_work_queue.cc",
        "//tensorflow/core/platform/default:unbounded_work_queue.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_srcs_no_runtime_google",
    srcs = [":legacy_srcs_common"] + tf_legacy_srcs_no_runtime_google(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_headers",
    srcs = glob(
        [
            "*.h",
            "profile_utils/**/*.h",
        ],
        exclude = [
            "dynamic_annotations.h",
            "denormal.h",
            "gif.h",
            "jpeg.h",
            "png.h",
            "setround.h",
            "stringprintf.h",
            "strong_hash.h",
            "str_util.h",
            "thread_annotations.h",
            "**/cuda.h",
            "**/rocm.h",
            "**/stream_executor.h",
        ],
    ),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_srcs",
    srcs = [
        "//tensorflow/core/platform:profile_utils/android_armv7a_cpu_utils_helper.cc",
        "//tensorflow/core/platform:profile_utils/clock_cycle_profiler.cc",
        "//tensorflow/core/platform:profile_utils/cpu_utils.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_test_srcs",
    srcs = tf_additional_test_srcs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

bzl_library(
    name = "build_config_root_bzl",
    srcs = [
        "build_config_root.bzl",
        "//tensorflow/core/platform/default:build_config_root.bzl",
    ],
)

# TODO(gunan): Remove the following once references in core/BUILD is removed.
exports_files(
    glob(
        [
            "*",
            "**",
        ],
        exclude = [
            "abi.h",
            "byte_order.h",
            "cpu_info.cc",
            "cpu_info.h",
            "logging.h",
            "macros.h",
            "platform.h",
            "types.h",
            "stacktrace.h",
        ],
    ),
)

exports_files(
    [
        "stacktrace.h",
        "cpu_info.h",
        "types.h",
        "macros.h",
        "abi.h",
        "logging.h",
        "platform.h",
        "byte_order.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)
