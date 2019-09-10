# Description:
#   TensorFlow Base libraries.
#   This package contains the following libraries:
#     - Platform dependent libraries that require different implementations
#       across different OSs or environments.
#     - STL replacement libraries rest of TensorFlow should depend on.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to any TensorFlow code outside this package.

load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_additional_lib_hdrs",
    "tf_additional_lib_srcs",
    "tf_additional_libdevice_srcs",
    "tf_additional_minimal_lib_srcs",
    "tf_additional_monitoring_srcs",
    "tf_additional_proto_hdrs",
    "tf_additional_rocdl_deps",
    "tf_additional_rocdl_srcs",
    "tf_additional_test_srcs",
    "tf_logging_absl_deps",
    "tf_platform_hdrs",
    "tf_platform_srcs",
    "tf_protobuf_compiler_deps",
    "tf_protobuf_deps",
)
load(
    "//tensorflow/core/platform:default/build_refactor.bzl",
    "tf_instantiate_platform_libraries",
    "tf_platform_helper_deps",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

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
    "dynamic_annotations",
    "env_time",
    "fingerprint",
    "human_readable_json",
    "mutex",
    "notification",
    "strong_hash",
])

cc_library(
    name = "abi",
    srcs = ["abi.cc"],
    hdrs = ["abi.h"],
    deps = [":types"],
)

cc_library(
    name = "annotation",
    srcs = ["annotation.cc"],
    hdrs = ["annotation.h"],
    visibility = [
        "//perftools/accelerators/xprof:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    deps = [
        ":macros",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
)

cc_library(
    name = "context",
    textual_hdrs = ["context.h"],
    deps = [
        # See call to tf_instantiate_platform_libraries() above.
        ":context_impl",
    ],
)

cc_library(
    name = "cord",
    hdrs = ["cord.h"],
    deps = [
        ":cord_impl",
        ":platform",
    ],
)

cc_library(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.cc"],
    hdrs = ["cpu_feature_guard.h"],
    deps = [
        ":byte_order",
        ":cpu_info",
        ":logging",
    ],
)

cc_library(
    name = "cpu_info",
    srcs = ["cpu_info.cc"],
    hdrs = ["cpu_info.h"],
    copts = tf_copts(),
    deps = [
        ":byte_order",
        ":logging",
        ":platform",
        ":types",
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
    name = "denormal",
    srcs = ["denormal.cc"],
    hdrs = ["denormal.h"],
    deps = [
        ":byte_order",
        ":cpu_info",
        ":logging",
        ":macros",
        ":platform",
    ],
)

cc_library(
    name = "dynamic_annotations",
    hdrs = ["dynamic_annotations.h"],
    deps = [
        ":dynamic_annotations_impl",
        ":platform",
    ],
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
        ":types",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/strings:string_utils",
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
        ":fingerprint_impl",
        ":platform",
        ":stringpiece",
        ":types",
    ],
)

cc_library(
    name = "grpc_services",
    hdrs = ["grpc_services.h"],
    deps = [
        ":platform",
        "//tensorflow/core/profiler:profiler_analysis_proto_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
    ],
)

cc_library(
    name = "host_info",
    hdrs = ["host_info.h"],
    deps = [":types"],
)

cc_library(
    name = "human_readable_json",
    textual_hdrs = ["human_readable_json.h"],
    deps = [
        ":human_readable_json_impl",
    ],
)

cc_library(
    name = "logging",
    srcs = tf_platform_hdrs(["logging.h"]) + tf_platform_srcs(["logging.cc"]),
    hdrs = ["logging.h"],
    deps = [
        ":env_time",
        ":macros",
        ":platform",
        ":types",
        "//tensorflow/core/platform/default/build_config:base",
    ] + tf_logging_absl_deps(),
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "mutex",
    textual_hdrs = ["mutex.h"],
    deps = [
        ":mutex_impl",
    ],
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    deps = [
        ":notification_impl",
        ":platform",
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
    name = "rocm_rocdl_path",
    srcs = ["rocm_rocdl_path.cc"] + tf_additional_rocdl_srcs(),
    hdrs = ["rocm_rocdl_path.h"],
    deps = [
        ":types",
        "//tensorflow/core:lib",
    ] + tf_additional_rocdl_deps(),
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
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
    name = "stacktrace",
    srcs = tf_platform_srcs([
        "stacktrace.h",
        "stacktrace.cc",
    ]),
    hdrs = ["stacktrace.h"],
    deps = [
        ":abi",
        ":platform",
        "//tensorflow/core/platform/default/build_config:stacktrace",
    ],
)

cc_library(
    name = "stacktrace_handler",
    srcs = tf_platform_srcs(["stacktrace_handler.cc"]),
    hdrs = ["stacktrace_handler.h"],
    deps = [
        ":platform",
        ":stacktrace",
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
    name = "str_util",
    srcs = ["str_util.cc"],
    hdrs = ["str_util.h"],
    deps = [
        ":logging",
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
        ":strong_hash_impl",
        ":types",
    ],
)

cc_library(
    name = "thread_annotations",
    hdrs = ["thread_annotations.h"],
)

cc_library(
    name = "tstring",
    hdrs = ["tstring.h"],
)

cc_library(
    name = "types",
    srcs = tf_platform_hdrs(["integral_types.h"]),
    hdrs = ["types.h"],
    deps = [
        ":platform",
        ":tstring",
        "//tensorflow/core/platform/default/build_config:base",
    ],
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

filegroup(
    name = "legacy_platform_lib_srcs",
    srcs = tf_additional_lib_srcs(
        exclude = [
            "*test*",
            "**/*test*",
            "**/cuda.h",
            "**/cuda_libdevice_path.cc",
            "**/rocm.h",
            "**/monitoring.cc",
            "**/stream_executor.h",
            "**/env_time.cc",
            "**/logger.cc",
            "**/logging.cc",
            "**/human_readable_json.cc",
            "**/rocm.h",
            "**/rocm_rocdl_path.cc",
            "abi.cc",
            "cpu_info.cc",
            "numbers.cc",
            "platform_strings.cc",
            "protobuf.cc",
            "scanner.cc",
            "strcat.cc",
            "stringprintf.cc",
        ],
    ),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_proto_hdrs",
    srcs = tf_additional_proto_hdrs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_srcs_no_runtime",
    srcs = glob(
        [
            "**/*.h",
            "**/*.cc",
        ],
        exclude = [
            "*test.*",
            "*testutil*",
            "*testlib*",
            "*main.cc",
            "**/*test.*",
            "**/*testutil*",
            "**/*testlib*",
            "**/*main.cc",
            "**/cuda_libdevice_path.*",
            "**/logger.cc",
            # Exclude env_time and logging to avoid collisions with
            # :platform_base, a common dependency for downstream targets.
            "**/env_time.cc",
            "**/logging.cc",
            "**/rocm_rocdl_path.*",
            "default/test_benchmark.*",
            "cuda.h",
            "rocm.h",
            "google/**/*",
            "hadoop/**/*",
            "gif.h",
            "jpeg.h",
            "png.h",
            "stream_executor.*",
            "windows/**/*",
        ],
    ),
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
            "annotation.h",
            "gif.h",
            "jpeg.h",
            "png.h",
            "stringprintf.h",
            "str_util.h",
            "**/cuda.h",
            "**/rocm.h",
            "**/stream_executor.h",
        ],
    ),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_srcs",
    srcs = glob(
        [
            "*.cc",
            "profile_utils/**/*.cc",
        ],
        exclude = [
            "*test*",
            "**/*test*",
            "**/env_time.cc",
            "**/monitoring.cc",
            "**/cuda_libdevice_path.cc",
            "**/logger.cc",
            "**/logging.cc",
            "**/human_readable_json.cc",
            "**/rocm_rocdl_path.cc",
            "abi.cc",
            "annotation.cc",
            "cpu_info.cc",
            "numbers.cc",
            "platform_strings.cc",
            "protobuf.cc",
            "scanner.cc",
            "strcat.cc",
            "stringprintf.cc",
            "str_util.cc",
        ],
    ),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_test_srcs",
    srcs = tf_additional_test_srcs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_minimal_lib_srcs",
    srcs = tf_additional_minimal_lib_srcs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_libdevice_srcs",
    srcs = tf_additional_libdevice_srcs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_monitoring_srcs",
    srcs = tf_additional_monitoring_srcs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_platform_port_srcs",
    srcs = tf_platform_hdrs([
        "cpu_info.h",
        "dynamic_annotations.h",
        "thread_annotations.h",
        "mutex.h",
    ]) + tf_platform_srcs([
        "port.cc",
    ]),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_platform_env_srcs",
    srcs = tf_platform_srcs([
        "env.cc",
        "load_library.cc",
    ]) + tf_platform_hdrs([
        "wide_char.h",
    ]),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_file_system_hdrs",
    srcs = tf_platform_hdrs([
        "windows_file_system.h",
    ]),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_platform_other_srcs",
    srcs = tf_platform_srcs([
        "subprocess.cc",
        "net.cc",
        "tracing.cc",
    ]) + tf_platform_hdrs([
        "tracing.h",
        "error.h",
        "context.h",
        "fingerprint.h",
        "notification.h",
        "strong_hash.h",
        "subprocess.h",
        "tracing_impl.h",
    ]),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_human_readable_json_src",
    srcs = tf_platform_srcs(["human_readable_json.cc"]),
    visibility = ["//tensorflow/core:__pkg__"],
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
