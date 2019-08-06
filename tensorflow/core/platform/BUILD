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
    "tf_additional_device_tracer_srcs",
    "tf_additional_lib_hdrs",
    "tf_additional_lib_srcs",
    "tf_additional_libdevice_srcs",
    "tf_additional_minimal_lib_srcs",
    "tf_additional_monitoring_srcs",
    "tf_additional_proto_hdrs",
    "tf_additional_test_srcs",
    "tf_env_time_srcs",
    "tf_logging_absl_deps",
    "tf_platform_hdrs",
    "tf_platform_srcs",
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

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
)

cc_library(
    name = "abi",
    srcs = ["abi.cc"],
    hdrs = ["abi.h"],
    deps = [":types"],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
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

cc_library(
    name = "env_time",
    srcs = ["env_time.cc"] + tf_env_time_srcs(),
    hdrs = ["env_time.h"],
    deps = [
        ":types",
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
    name = "cpu_info",
    srcs = ["cpu_info.cc"] + tf_platform_srcs([
        "cpu_info.h",
    ]),
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
    name = "stacktrace",
    srcs = glob(["*/stacktrace.h"]),
    hdrs = ["stacktrace.h"],
    deps = [
        ":abi",
        ":platform",
        "//tensorflow/core/platform/default/build_config:stacktrace",
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
            "**/device_tracer.cc",
            "**/logger.cc",
            "**/logging.cc",
            "**/human_readable_json.cc",
            "abi.cc",
            "cpu_info.cc",
            "platform_strings.cc",
            "protobuf.cc",
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
            "gif.h",
            "jpeg.h",
            "png.h",
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
            "**/device_tracer.cc",
            "**/logger.cc",
            "**/logging.cc",
            "**/human_readable_json.cc",
            "abi.cc",
            "cpu_info.cc",
            "platform_strings.cc",
            "protobuf.cc",
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
    name = "legacy_device_tracer_srcs",
    srcs = tf_additional_device_tracer_srcs(),
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
