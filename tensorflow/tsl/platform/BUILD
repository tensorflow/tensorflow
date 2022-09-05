# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_cuda_libdevice_path_deps",
    "tf_fingerprint_deps",
    "tf_logging_deps",
    "tf_platform_deps",
    "tf_protobuf_compiler_deps",
    "tf_resource_deps",
    "tf_testing_deps",
    "tf_windows_aware_platform_deps",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
    compatible_with = get_compatible_with_portable(),
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
        "bfloat16.h",
        "casts.h",
        "cpu_info.h",
        "crash_analysis.h",
        "dynamic_annotations.h",
        "fingerprint.h",
        "init_main.h",
        "mem.h",
        "net.h",
        "numa.h",
        "prefetch.h",
        "resource.h",
        "stack_frame.h",
        "stacktrace.h",
        "stacktrace_handler.h",
        "str_util.h",
        "stringpiece.h",
        "stringprintf.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Header files for tensorflow/core:platform_base.
filegroup(
    name = "base_hdrs",
    srcs = [
        "byte_order.h",
        "macros.h",
        "threadpool_interface.h",
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
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "bfloat16.h",
        "byte_order.h",
        "casts.h",
        "cpu_info.cc",
        "cpu_info.h",
        "demangle.h",
        "dynamic_annotations.h",
        "fingerprint.h",
        "hash.cc",
        "hash.h",
        "host_info.h",
        "init_main.h",
        "intrusive_ptr.h",
        "macros.h",
        "mem.h",
        "numa.h",
        "platform.h",
        "platform_strings_computed.h",
        "prefetch.h",
        "raw_coding.h",
        "regexp.h",
        "resource.h",
        "snappy.h",
        "stacktrace.h",
        "str_util.cc",
        "str_util.h",
        "stringpiece.h",
        "stringprintf.cc",
        "stringprintf.h",
        "threadpool_interface.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "fingerprint.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core/platform:__pkg__"],
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
            "jpeg.h",
        ],
    ),
    compatible_with = get_compatible_with_portable(),
)

exports_files(
    [
        "host_info.h",
        "bfloat16.h",
        "cpu_info.h",
        "cpu_info.cc",
        "crash_analysis.h",
        "cuda_libdevice_path.h",
        "demangle.h",
        "init_main.h",
        "logging.h",
        "mem.h",
        "net.h",
        "numa.h",
        "resource_loader.h",
        "resource.h",
        "snappy.h",
        "stacktrace_handler.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
        "init_main.h",
        "macros.h",
        "platform.h",
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
        "demangle.h",
        "host_info.h",
        "intrusive_ptr.h",
        "platform.h",
        "snappy.h",
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
    name = "resource_loader",
    testonly = 1,
    textual_hdrs = ["resource_loader.h"],
    deps = tf_testing_deps("resource_loader"),
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
        "//third_party/eigen3",
        # TODO(jakeharmon): Remove deps on core once mutex and types are migrated
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        ":bfloat16",
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
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
    ] + tf_windows_aware_platform_deps("stacktrace"),
)

cc_library(
    name = "cuda_libdevice_path",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["cuda_libdevice_path.h"],
    deps = tf_cuda_libdevice_path_deps(),
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

tf_cc_test(
    name = "fingerprint_test",
    size = "small",
    srcs = ["fingerprint_test.cc"],
    deps = [
        ":fingerprint",
        ":types",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "cuda",
    hdrs = ["cuda.h"],
    deps = [
        ":platform",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation_header",
    ],
)

tf_cuda_library(
    name = "stream_executor",
    srcs = ["stream_executor.h"],
    hdrs = [
        "cuda.h",
    ],
    features = ["-parse_headers"],
    deps = [
        "//tensorflow/tsl/platform/default/build_config:stream_executor",
    ],
)
