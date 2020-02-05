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
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_gpu_tests_tags",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_env_hdrs",
    "tf_additional_lib_hdrs",
    "tf_additional_monitoring_hdrs",
    "tf_additional_tensor_coding_deps",
    "tf_additional_test_srcs",
    "tf_fingerprint_deps",
    "tf_google_mobile_srcs_no_runtime",
    "tf_google_mobile_srcs_only_runtime",
    "tf_kernel_tests_linkstatic",
    "tf_logging_deps",
    "tf_monitoring_deps",
    "tf_platform_alias",
    "tf_platform_deps",
    "tf_protobuf_compiler_deps",
    "tf_protobuf_deps",
    "tf_windows_aware_platform_deps",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_chromiumos",
    "if_not_android",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",  # @unused
    "tf_cuda_library",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(
    [
        "context.h",
        "cpu_info.cc",
        "cpu_info.h",
        "cuda_libdevice_path.h",
        "demangle.h",
        "env.cc",
        "env.h",
        "env_time.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.cc",
        "file_system_helper.h",
        "host_info.h",
        "human_readable_json.h",
        "init_main.h",
        "load_library.h",
        "logging.h",
        "mem.h",
        "monitoring.h",
        "mutex.h",
        "net.h",
        "numa.h",
        "resource_loader.h",
        "snappy.h",
        "stacktrace_handler.h",
        "subprocess.h",
        "test.h",
        "threadpool.cc",
        "threadpool.h",
        "tracing.cc",
        "tracing.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

exports_files(["rocm_rocdl_path.h"])

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
    deps = tf_platform_deps("context"),
)

cc_library(
    name = "cord",
    hdrs = ["cord.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("cord"),
)

cc_library(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.cc"],
    hdrs = ["cpu_feature_guard.h"],
    deps = [
        ":byte_order",
        ":logging",
        ":platform_port",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("casts"),
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
    deps = tf_platform_deps("cuda_libdevice_path"),
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
    ] + tf_platform_deps("dynamic_annotations"),
)

cc_library(
    name = "env",
    textual_hdrs = [
        "env.h",
        "file_system.h",
        "file_system_helper.h",
        "threadpool.h",
    ],
    deps = tf_windows_aware_platform_deps("env"),
)

cc_library(
    name = "env_time",
    textual_hdrs = ["env_time.h"],
    deps = tf_windows_aware_platform_deps("env_time"),
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
    deps = tf_platform_deps("human_readable_json"),
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
    deps = tf_windows_aware_platform_deps("load_library"),
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
    deps = tf_platform_deps("mutex"),
)

cc_library(
    name = "net",
    textual_hdrs = ["net.h"],
    deps = tf_windows_aware_platform_deps("net"),
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("notification"),
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
    deps = tf_windows_aware_platform_deps("platform_port"),
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
    defines = select({
        "//tensorflow:android": ["TENSORFLOW_LITE_PROTOS"],
        "//conditions:default": [],
    }),
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
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "resource_loader",
    testonly = 1,
    textual_hdrs = ["resource_loader.h"],
    deps = tf_platform_deps("resource_loader"),
)

cc_library(
    name = "rocm_rocdl_path",
    textual_hdrs = ["rocm_rocdl_path.h"],
    deps = tf_platform_deps("rocm_rocdl_path"),
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
    ] + tf_windows_aware_platform_deps("stacktrace"),
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
    deps = tf_windows_aware_platform_deps("stacktrace_handler"),
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
    ] + tf_platform_deps("strong_hash"),
)

cc_library(
    name = "subprocess",
    textual_hdrs = [
        "subprocess.h",
    ],
    deps = tf_windows_aware_platform_deps("subprocess"),
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
    deps = tf_platform_deps("test"),
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("test_benchmark"),
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
    deps = tf_platform_deps("tracing"),
)

cc_library(
    name = "tstring",
    hdrs = [
        "ctstring.h",
        "ctstring_internal.h",
        "tstring.h",
    ],
    deps = [
        ":cord",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        ":platform",
        ":tstring",
    ] + tf_platform_deps("types"),
)

cc_library(
    name = "unbounded_work_queue",
    hdrs = ["unbounded_work_queue.h"],
    deps = [
        ":platform",
    ] + tf_platform_deps("unbounded_work_queue"),
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

tf_cuda_library(
    name = "stream_executor",
    srcs = ["stream_executor.h"],
    hdrs = [
        "cuda.h",
        "rocm.h",
        "stream_executor.h",
    ],
    features = ["-parse_headers"],
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor",
    ],
)

# Like stream_executor library, but compiles without --config=cuda
# and does not include any cuda dependencies.
cc_library(
    name = "stream_executor_no_cuda",
    srcs = ["stream_executor.h"],
    hdrs = [
        "stream_executor_no_cuda.h",
    ],
    features = ["-parse_headers"],
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        "//tensorflow/core/platform/default/build_config:stream_executor_no_cuda",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//tensorflow/core:__pkg__"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",  # buildcleaner: keep
        "//tensorflow/core/platform/default/build_config:test_main",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "fingerprint_test.cc",
        "integral_types_test.cc",
        "logging_test.cc",
        "mutex_test.cc",
        "net_test.cc",
        "port_test.cc",
        "profile_utils/cpu_utils_test.cc",
        "scanner_test.cc",
        "stacktrace_handler_test.cc",
        "stacktrace_test.cc",
        "str_util_test.cc",
        "strcat_test.cc",
        "stringpiece_test.cc",
        "stringprintf_test.cc",
        "vmodule_benchmark_test.cc",
    ],
    create_named_test_suite = True,
    deps = [
        ":scanner",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":stringprintf",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@zlib_archive//:zlib",
    ],
)

tf_cc_test(
    name = "subprocess_test",
    size = "small",
    srcs = ["subprocess_test.cc"],
    data = [
        "//tensorflow/core/platform/testdata:test_echo",
        "//tensorflow/core/platform/testdata:test_echo_argv_1",
        "//tensorflow/core/platform/testdata:test_noop",
        "//tensorflow/core/platform/testdata:test_stderr",
    ],
    deps = [
        ":resource_loader",
        ":strcat",
        ":subprocess",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "resource_loader_test",
    size = "small",
    srcs = ["resource_loader_test.cc"],
    data = [
        ":resource_loader.h",
    ],
    deps = [
        ":resource_loader",
        ":status",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "ctstring_test",
    size = "small",
    srcs = ["ctstring_test.cc"],
    deps = [
        ":tstring",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "tstring_test",
    size = "small",
    srcs = ["tstring_test.cc"],
    deps = [
        ":cord",
        ":tstring",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "platform_strings_test",
    size = "small",
    srcs = ["platform_strings_test.cc"],
    features = ["-dynamic_link_test_srcs"],  # see go/dynamic_link_test_srcs
    deps = [
        ":platform_strings",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "env_test",
    size = "small",
    srcs = ["env_test.cc"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "fake_python_env_test",
    size = "small",
    srcs = ["fake_python_env_test.cc"],
    args = [
        "/some/path/to/pythontest.runfiles/org_tensorflow/stuff/to/run.py",
    ],
    tags = [
        "local",
        "no_gpu",
        "no_windows",
        "nomac",
        "notap",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "abi_test",
    size = "small",
    srcs = ["abi_test.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "setround_test",
    size = "small",
    srcs = ["setround_test.cc"],
    tags = [
        "noasan",
        "noclang",
        "nomsan",
        "notsan",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "file_system_test",
    size = "small",
    srcs = ["file_system_test.cc"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "unbounded_work_queue_test",
    srcs = ["unbounded_work_queue_test.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "vmodule_test",
    srcs = ["vmodule_test.cc"],
    tags = ["optonly"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test_gpu(
    name = "rocm_rocdl_path_test",
    size = "small",
    srcs = ["rocm_rocdl_path_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_gpu_tests_tags(),
    deps = [
        ":rocm_rocdl_path",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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

# Header files for tensorflow/core:platform_base.
filegroup(
    name = "base_hdrs",
    srcs = [
        "byte_order.h",
        "cord.h",
        "ctstring.h",
        "ctstring_internal.h",
        "env_time.h",
        "logging.h",
        "macros.h",
        "platform_strings.h",
        "threadpool.h",
        "threadpool_interface.h",
        "threadpool_options.h",
        "tstring.h",
        "types.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "abi.h",
        "casts.h",
        "context.h",
        "cpu_feature_guard.h",
        "cpu_info.h",
        "dynamic_annotations.h",
        "env.h",
        "error.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "fingerprint.h",
        "init_main.h",
        "logger.h",
        "mem.h",
        "monitoring.h",
        "mutex.h",
        "net.h",
        "notification.h",
        "null_file_system.h",
        "numa.h",
        "path.h",
        "prefetch.h",
        "profile_utils/android_armv7a_cpu_utils_helper.h",
        "profile_utils/clock_cycle_profiler.h",
        "profile_utils/cpu_utils.h",
        "profile_utils/i_cpu_utils_helper.h",
        "protobuf.h",
        "stacktrace.h",
        "stacktrace_handler.h",
        "status.h",
        "str_util.h",
        "strcat.h",
        "stringpiece.h",
        "stringprintf.h",
        "strong_hash.h",
        "subprocess.h",
        "thread_annotations.h",
        ":base_hdrs",
    ] + tf_additional_monitoring_hdrs() + tf_additional_env_hdrs(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
        "init_main.h",
        "logging.h",
        "macros.h",
        "platform.h",
        "protobuf.h",
        "stringpiece.h",
        "tstring.h",
        "types.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "test_hdrs",
    srcs = [
        "test.h",
        "test_benchmark.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "android_test_srcs",
    srcs = [
        "test.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
        "byte_order.h",
        "cpu_info.h",
        "ctstring.h",
        "ctstring_internal.h",
        "dynamic_annotations.h",
        "macros.h",
        "mutex.h",
        "platform.h",
        "prefetch.h",
        "protobuf.h",
        "thread_annotations.h",
        "tstring.h",
        "types.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_internal_private_hdrs",
    srcs = [
        "raw_coding.h",
        "scanner.h",
        "str_util.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
        "blocking_counter.h",
        "demangle.h",
        "denormal.h",
        "host_info.h",
        "monitoring.h",
        "platform.h",
        "protobuf_internal.h",
        "refcount.h",
        "setround.h",
        "snappy.h",
        "tensor_coding.h",
        "tracing.h",
        "unbounded_work_queue.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "jpeg_hdrs",
    srcs = [
        "jpeg.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "gif_hdrs",
    srcs = [
        "gif.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "tflite_portable_logging_hdrs",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
        "logging.h",
        "macros.h",
        "platform.h",
        "tstring.h",
        "types.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "jpeg_internal_hdrs",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
        "dynamic_annotations.h",
        "logging.h",
        "macros.h",
        "mem.h",
        "platform.h",
        "stringpiece.h",
        "tstring.h",
        "types.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "gif_internal_hdrs",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
        "dynamic_annotations.h",
        "logging.h",
        "macros.h",
        "mem.h",
        "platform.h",
        "tstring.h",
        "types.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "abi.cc",
        "abi.h",
        "blocking_counter.h",
        "byte_order.h",
        "coding.cc",
        "coding.h",
        "context.h",
        "cord.h",
        "cpu_info.cc",
        "cpu_info.h",
        "ctstring.h",
        "ctstring_internal.h",
        "demangle.h",
        "denormal.cc",
        "denormal.h",
        "dynamic_annotations.h",
        "env.cc",
        "env.h",
        "env_time.h",
        "error.cc",
        "error.h",
        "errors.h",
        "file_statistics.h",
        "file_system.cc",
        "file_system.h",
        "file_system_helper.cc",
        "file_system_helper.h",
        "hash.cc",
        "hash.h",
        "host_info.h",
        "init_main.h",
        "load_library.h",
        "logging.h",
        "macros.h",
        "mem.h",
        "mutex.h",
        "numa.h",
        "numbers.cc",
        "numbers.h",
        "path.cc",
        "path.h",
        "platform.h",
        "prefetch.h",
        "protobuf.cc",
        "protobuf.h",
        "protobuf_util.cc",
        "raw_coding.h",
        "refcount.h",
        "scanner.cc",
        "scanner.h",
        "setround.cc",
        "setround.h",
        "snappy.h",
        "stacktrace.h",
        "status.cc",
        "status.h",
        "str_util.cc",
        "str_util.h",
        "strcat.cc",
        "strcat.h",
        "stringpiece.h",
        "stringprintf.cc",
        "stringprintf.h",
        "tensor_coding.cc",
        "tensor_coding.h",
        "thread_annotations.h",
        "threadpool.cc",
        "threadpool.h",
        "threadpool_interface.h",
        "tracing.cc",
        "tracing.h",
        "tstring.h",
        "types.h",
    ] + if_chromiumos(
        tf_google_mobile_srcs_no_runtime(),
        otherwise = [
            "//tensorflow/core/platform/default:mobile_srcs_no_runtime",
        ],
    ) + tf_platform_alias("additional_mobile_srcs_no_runtime"),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "base64.cc",
        "base64.h",
        "casts.h",
        "cpu_feature_guard.cc",
        "cpu_feature_guard.h",
        "fingerprint.h",
        "monitoring.h",
        "notification.h",
        "platform_strings.cc",
        "platform_strings.h",
        "platform_strings_computed.h",
        "profile_utils/android_armv7a_cpu_utils_helper.cc",
        "profile_utils/android_armv7a_cpu_utils_helper.h",
        "profile_utils/cpu_utils.cc",
        "profile_utils/cpu_utils.h",
        "profile_utils/i_cpu_utils_helper.h",
        "protobuf_internal.h",
        "random.cc",
        "random.h",
        "subprocess.h",
        "test_benchmark.h",
        "threadpool_options.h",
        "unbounded_work_queue.h",
        "//tensorflow/core/platform/default:mobile_srcs_only_runtime",
    ] + tf_google_mobile_srcs_only_runtime(),
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
        "profile_utils/android_armv7a_cpu_utils_helper.cc",
        "profile_utils/clock_cycle_profiler.cc",
        "profile_utils/cpu_utils.cc",
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
    ] + tf_platform_alias("build_config_root.bzl"),
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = [
        "rules_cc.bzl",
    ] + tf_platform_alias("rules_cc.bzl"),
)
