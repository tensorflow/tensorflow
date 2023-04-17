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
    "tf_additional_lib_hdrs",
    "tf_additional_tensor_coding_deps",
    "tf_google_mobile_srcs_only_runtime",
    "tf_platform_alias",
    "tf_platform_deps",
    "tf_windows_aware_platform_deps",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_binary",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
    "if_not_android",
    "if_not_fuchsia",
    "if_windows",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_copts",  # @unused
    "tf_cuda_library",
)
load(
    "//tensorflow/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_portable", "tf_cuda_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Authorized users go here.
package_group(
    name = "friends",
    packages = if_google([
        "@tf_runtime//...",
        "//third_party/tf_runtime_google/...",
    ]),
)

exports_files(
    [
        "context.h",
        "cpu_info.h",
        "bfloat16.h",
        "demangle.h",
        "env.h",
        "env_time.h",
        "file_system.h",
        "file_system_helper.h",
        "float8.h",
        "host_info.h",
        "human_readable_json.h",
        "init_main.h",
        "load_library.h",
        "logging.h",
        "mem.h",
        "mutex.h",
        "crash_analysis.h",
        "net.h",
        "numa.h",
        "ram_file_system.h",
        "resource_loader.h",
        "resource.h",
        "snappy.h",
        "stacktrace_handler.h",
        "subprocess.h",
        "test.h",
        "threadpool.h",
        "tracing.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

exports_files(["rocm_rocdl_path.h"])

cc_library(
    name = "abi",
    hdrs = ["abi.h"],
    deps = [
        "//tensorflow/tsl/platform:abi",
    ],
)

cc_library(
    name = "base64",
    hdrs = ["base64.h"],
    deps = [
        ":errors",
        ":status",
        ":stringpiece",
        "//tensorflow/tsl/platform:base64",
    ],
)

cc_library(
    name = "bfloat16",
    hdrs = ["bfloat16.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core/platform:byte_order",
        "//tensorflow/tsl/platform:bfloat16",
    ],
)

cc_library(
    name = "float8",
    hdrs = ["float8.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:float8",
    ],
)

cc_library(
    name = "blocking_counter",
    hdrs = ["blocking_counter.h"],
    deps = [
        ":logging",
        ":mutex",
        "//tensorflow/tsl/platform:blocking_counter",
    ],
)

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["//tensorflow/tsl/platform:byte_order"],
)

cc_library(
    name = "coding",
    hdrs = ["coding.h"],
    deps = [
        ":raw_coding",
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:coding",
    ],
)

cc_library(
    name = "context",
    textual_hdrs = ["context.h"],
    deps = [
        "//tensorflow/tsl/platform:context",
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
    name = "crash_analysis",
    hdrs = ["crash_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:crash_analysis",
    ],
)

cc_library(
    name = "cpu_feature_guard",
    srcs = ["cpu_feature_guard.cc"],
    hdrs = ["cpu_feature_guard.h"],
    deps = [
        ":byte_order",
        ":logging",
        ":platform_port",
        "//tensorflow/tsl/platform:platform_port",
        "@com_google_absl//absl/base",
    ],
)

# Required to avoid ODR issues in the pybind wrapper.
cc_library(
    name = "cpu_feature_guard_hdr",
    hdrs = ["cpu_feature_guard.h"],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:casts",
    ] + tf_platform_deps("casts"),
)

cc_library(
    name = "cuda",
    hdrs = ["cuda.h"],
    compatible_with = [],
    deps = [
        ":platform",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation_header",
    ],
)

cc_library(
    name = "denormal",
    hdrs = ["denormal.h"],
    deps = [
        ":macros",
        "//tensorflow/tsl/platform:denormal",
    ],
)

cc_library(
    name = "dynamic_annotations",
    hdrs = ["dynamic_annotations.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:dynamic_annotations",
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
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:context",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:denormal",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:file_statistics",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/platform:scanner",
        "//tensorflow/core/platform:setround",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:stringprintf",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/platform:tracing",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/platform:env",
    ] + tf_windows_aware_platform_deps("env"),
)

tf_cc_test(
    name = "env_test",
    size = "small",
    srcs = ["env_test.cc"],
    deps = [
        ":cord",
        ":env",
        ":env_impl",
        ":null_file_system",
        ":path",
        ":protobuf",
        ":str_util",
        ":strcat",
        ":stringpiece",
        ":test",
        ":test_main",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//third_party/eigen3",
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
    deps = [
        "//tensorflow/tsl/platform:env_time",
    ] + tf_windows_aware_platform_deps("env_time"),
)

cc_library(
    name = "errors",
    hdrs = ["errors.h"],
    deps = [
        ":logging",
        ":macros",
        ":platform",
        ":status",
        ":str_util",
        ":strcat",
        ":types",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "file_statistics",
    hdrs = ["file_statistics.h"],
    deps = [
        ":types",
        "//tensorflow/tsl/platform:file_statistics",
    ],
)

cc_library(
    name = "fingerprint",
    hdrs = ["fingerprint.h"],
    deps = [
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:fingerprint",
    ],
)

cc_library(
    name = "gif",
    hdrs = ["gif.h"],
    deps = [
        "//tensorflow/tsl/platform:gif",
    ],
)

cc_library(
    name = "hash",
    hdrs = ["hash.h"],
    deps = [
        ":macros",
        ":raw_coding",
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:hash",
    ],
)

cc_library(
    name = "human_readable_json",
    textual_hdrs = ["human_readable_json.h"],
    deps = [
        "//tensorflow/tsl/platform:human_readable_json",
    ],
)

cc_library(
    name = "jpeg",
    hdrs = ["jpeg.h"],
    deps = [
        "//tensorflow/tsl/platform:jpeg",
    ],
)

cc_library(
    name = "load_library",
    textual_hdrs = ["load_library.h"],
    deps = [
        "//tensorflow/tsl/platform:load_library",
    ],
)

cc_library(
    name = "logger",
    hdrs = ["logger.h"],
    deps = [
        ":env",
        ":logging",
        ":protobuf",
        "//tensorflow/tsl/platform:logger",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_library(
    name = "logging",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["logging.h"],
    deps = [
        ":types",
        "//tensorflow/tsl/platform:logging",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:macros",
    ],
)

filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = tf_platform_deps("xla_cpu_runtime_srcs") + [
        "cord.h",
        "dynamic_annotations.h",
        "env_time.h",
        "macros.h",
        "platform.h",
        "tstring.h",
        "types.h",
        ":ctstring",
    ] + if_windows([
        "//tensorflow/tsl/platform/windows:xla_cpu_runtime_srcs",
    ]),
)

cc_library(
    name = "mutex",
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["mutex.h"],
    deps = [
        ":platform",
        ":thread_annotations",
        ":types",
        "//tensorflow/tsl/platform:mutex",
    ],
)

cc_library(
    name = "net",
    textual_hdrs = ["net.h"],
    deps = [
        "//tensorflow/tsl/platform:net",
    ] + tf_windows_aware_platform_deps("net"),
)

cc_library(
    name = "notification",
    hdrs = ["notification.h"],
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:notification",
    ],
)

cc_library(
    name = "null_file_system",
    hdrs = ["null_file_system.h"],
    deps = [
        "//tensorflow/tsl/platform:null_file_system",
    ],
)

py_test(
    name = "ram_file_system_test",
    srcs = ["ram_file_system_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "numbers",
    hdrs = ["numbers.h"],
    deps = [
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:numbers",
    ],
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    deps = [
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:path",
    ],
    alwayslink = True,
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform",
    ],
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
    deps = [
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/platform:platform_port",
    ],
)

cc_library(
    name = "platform_strings",
    srcs = [
        "platform_strings.cc",
        "platform_strings_computed.h",
    ],
    hdrs = ["platform_strings.h"],
    deps = [
        "//tensorflow/tsl/platform:platform_strings",
    ],
)

cc_library(
    name = "png",
    hdrs = ["png.h"],
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:png",
    ],
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    deps = ["//tensorflow/tsl/platform:prefetch"],
)

cc_library(
    name = "protobuf",
    hdrs = ["protobuf.h"],
    deps = [
        ":platform",
        ":types",
        "//tensorflow/tsl/platform:protobuf",
    ],
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
    hdrs = ["random.h"],
    deps = [
        ":mutex",
        ":types",
        "//tensorflow/tsl/platform:random",
    ],
)

cc_library(
    name = "raw_coding",
    hdrs = ["raw_coding.h"],
    deps = [
        ":byte_order",
        ":types",
        "//tensorflow/tsl/platform:raw_coding",
    ],
)

cc_library(
    name = "refcount",
    hdrs = ["refcount.h"],
    deps = [
        ":mutex",
        "//tensorflow/tsl/platform:refcount",
    ],
)

cc_library(
    name = "intrusive_ptr",
    hdrs = ["intrusive_ptr.h"],
    deps = ["//tensorflow/tsl/platform:intrusive_ptr"],
)

cc_library(
    name = "regexp",
    hdrs = ["regexp.h"],
    deps = [
        "//tensorflow/tsl/platform:regexp",
    ],
)

cc_library(
    name = "resource",
    textual_hdrs = ["resource.h"],
    deps = [
        ":stringpiece",
        "//tensorflow/tsl/platform:resource",
    ],
)

cc_library(
    name = "resource_loader",
    testonly = 1,
    textual_hdrs = ["resource_loader.h"],
    deps = [
        "//tensorflow/tsl/platform:resource_loader",
    ],
)

cc_library(
    name = "rocm_rocdl_path",
    textual_hdrs = ["rocm_rocdl_path.h"],
    deps = [
        "//tensorflow/tsl/platform:rocm_rocdl_path",
    ],
)

cc_library(
    name = "rocm",
    hdrs = if_rocm_is_configured(["rocm.h"]),
    deps = if_rocm_is_configured([
        ":platform",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_activation",
    ]),
)

cc_library(
    name = "scanner",
    hdrs = ["scanner.h"],
    deps = [
        "//tensorflow/tsl/platform:scanner",
    ],
)

cc_library(
    name = "setround",
    hdrs = ["setround.h"],
    deps = [
        ":logging",
        ":macros",
        "//tensorflow/tsl/platform:setround",
    ],
)

cc_library(
    name = "stacktrace",
    hdrs = ["stacktrace.h"],
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:stacktrace",
    ],
)

filegroup(
    name = "stacktrace_handler_hdrs",
    srcs = [
        "stacktrace_handler.h",
        "//tensorflow/tsl/platform:stacktrace_handler_hdrs",
    ],
)

cc_library(
    name = "stacktrace_handler",
    textual_hdrs = ["stacktrace_handler.h"],
    deps = [
        "//tensorflow/tsl/platform:stacktrace_handler",
    ],
    alwayslink = 1,
)

cc_library(
    name = "stack_frame",
    hdrs = ["stack_frame.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["//tensorflow/tsl/platform:stack_frame"],
)

cc_library(
    name = "status",
    hdrs = ["status.h"],
    deps = [
        ":logging",
        ":macros",
        ":mutex",
        ":stack_frame",
        ":stacktrace",
        ":str_util",
        ":strcat",
        ":stringprintf",
        ":types",
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "status_matchers",
    testonly = 1,
    hdrs = ["status_matchers.h"],
    deps = [
        ":status",
        ":statusor",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "statusor",
    hdrs = ["statusor.h"],
    deps = [
        ":macros",
        ":status",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "str_util",
    hdrs = ["str_util.h"],
    deps = [
        ":macros",
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:str_util",
    ],
)

cc_library(
    name = "strcat",
    hdrs = ["strcat.h"],
    deps = [
        ":logging",
        ":macros",
        ":numbers",
        ":stringpiece",
        ":types",
        "//tensorflow/tsl/platform:strcat",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

cc_library(
    name = "stringpiece",
    hdrs = ["stringpiece.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "stringprintf",
    hdrs = ["stringprintf.h"],
    deps = [
        ":macros",
        ":types",
        "//tensorflow/tsl/platform:stringprintf",
    ],
)

cc_library(
    name = "strong_hash",
    textual_hdrs = ["strong_hash.h"],
    deps = [
        ":platform",
        ":types",
        "@highwayhash//:sip_hash",
    ],
)

cc_library(
    name = "subprocess",
    textual_hdrs = [
        "subprocess.h",
    ],
    deps = [
        "//tensorflow/tsl/platform:subprocess",
    ],
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
        ":strcat",
        ":stringpiece",
        ":types",
    ] + tf_additional_tensor_coding_deps(),
)

cc_library(
    name = "test",
    testonly = True,
    compatible_with = get_compatible_with_portable(),
    textual_hdrs = ["test.h"],
    deps = [
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "test_benchmark",
    testonly = True,
    hdrs = ["test_benchmark.h"],
    deps = [
        "//tensorflow/tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "thread_annotations",
    hdrs = ["thread_annotations.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "threadpool_interface",
    hdrs = ["threadpool_interface.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":mutex",
        ":types",
        "//tensorflow/tsl/platform:threadpool_interface",
    ],
)

cc_library(
    name = "threadpool_options",
    hdrs = ["threadpool_options.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":threadpool_interface",
        "//tensorflow/tsl/platform:threadpool_options",
    ],
)

cc_library(
    name = "tracing",
    textual_hdrs = ["tracing.h"],
    deps = [
        "//tensorflow/tsl/platform:tracing",
    ],
)

cc_library(
    name = "tstring",
    hdrs = [
        "ctstring.h",  #TODO(ddunleavy) make this cleaner later. only depend on tstring.h
        "ctstring_internal.h",
        "tstring.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/platform:tstring",
    ],
)

filegroup(
    name = "ctstring",
    srcs = [
        "ctstring.h",
        "ctstring_internal.h",
        "//tensorflow/tsl/platform:ctstring",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bfloat16",
        ":float8",
        ":platform",
        ":tstring",
        "//tensorflow/tsl/framework:device_type",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "unbounded_work_queue",
    hdrs = ["unbounded_work_queue.h"],
    deps = [
        ":platform",
        "//tensorflow/tsl/platform:unbounded_work_queue",
    ],
)

cc_library(
    name = "retrying_utils",
    hdrs = [
        "retrying_utils.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/tsl/platform:retrying_utils",
    ],
)

cc_library(
    name = "retrying_file_system",
    hdrs = [
        "retrying_file_system.h",
    ],
    copts = tf_copts(),
    deps = [
        ":retrying_utils",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/tsl/platform:retrying_file_system",
    ],
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
        ":crash_analysis",
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
        ":retrying_file_system",
        ":retrying_utils",
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
        "//tensorflow/tsl/platform:build_test",
    ] + if_not_android([":rocm_rocdl_path"]),
)

cc_library(
    name = "gpu_all_runtime",
    visibility = ["//visibility:private"],
    deps = if_cuda_is_configured(
        ["//tensorflow/compiler/xla/stream_executor/cuda:all_runtime"],
    ) + if_rocm_is_configured(
        ["//tensorflow/compiler/xla/stream_executor/rocm:all_runtime"],
    ),
)

# This do-nothing lib should only be used as a dep in cuda_deps in :stream_executor cuda lib because
# the "+" operator doesn't work for bazel in stream_executor cuda_deps.
cc_library(
    name = "private_static_dep",
    visibility = ["//visibility:private"],
    deps = if_static(
        [":gpu_all_runtime"],
    ),
)

tf_cuda_library(
    name = "stream_executor",
    srcs = ["stream_executor.h"],
    hdrs = [
        "cuda.h",
        "rocm.h",
        "stream_executor.h",
    ],
    cuda_deps =
        if_google(
            ["//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform"],
            [
                ":private_static_dep",
                "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation",
                "//tensorflow/compiler/xla/stream_executor/rocm:rocm_activation",
            ],
        ),
    features = ["-parse_headers"],
    deps = [
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_activation_header",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/platform",
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
    deps = [
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:dnn",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:multi_platform_manager",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_platform_id",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform_id",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
        "//tensorflow/compiler/xla/stream_executor/rocm:rocm_platform_id",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    deps = [
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "tensor_float_32_utils",
    hdrs = ["tensor_float_32_utils.h"],
    deps = [
        "//tensorflow/tsl/platform:tensor_float_32_utils",
    ],
)

cc_library(
    name = "enable_tf2_utils",
    srcs = ["enable_tf2_utils.cc"],
    hdrs = ["enable_tf2_utils.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core/util:env_var",
    ],
    alwayslink = 1,
)

cc_library(
    name = "error_payloads",
    srcs = ["error_payloads.cc"],
    hdrs = ["error_payloads.h"],
    deps = [
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

alias(
    name = "profile_utils_cpu_utils",
    actual = "//tensorflow/core/platform/profile_utils:profile_utils_cpu_utils",
)

filegroup(
    name = "tensor_float_32_hdr",
    srcs = [
        "tensor_float_32_utils.h",
        "//tensorflow/tsl/platform:tensor_float_32_hdr",
    ],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "tensor_float_32_hdr_lib",
    hdrs = [":tensor_float_32_hdr"],
    compatible_with = get_compatible_with_portable(),
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = [
        "rocm.h",
        "stream_executor.h",
    ],
    visibility = [
        "//tensorflow/contrib/fused_conv:__pkg__",
        "//tensorflow/contrib/image:__pkg__",
        "//tensorflow/core/kernels:__pkg__",
        "//tensorflow/core/kernels/image:__pkg__",
    ],
)

filegroup(
    name = "enable_tf2_hdr",
    srcs = ["enable_tf2_utils.h"],
    compatible_with = get_compatible_with_portable(),
)

tf_cc_tests(
    name = "low_level_library_tests",
    size = "small",
    srcs = [
        "vmodule_benchmark_test.cc",
        "//tensorflow/tsl/platform/profile_utils:cpu_utils_test.cc",
    ],
    create_named_test_suite = True,
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tsl/platform:logging",
        "@zlib",
    ],
)

tf_cc_test(
    name = "enable_tf2_utils_test",
    size = "small",
    srcs = [
        "enable_tf2_utils_test.cc",
    ],
    deps = [
        ":enable_tf2_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/util:env_var",
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
        "//tensorflow/tsl/platform:resource_loader",
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

tf_cuda_cc_test(
    name = "rocm_rocdl_path_test",
    size = "small",
    srcs = ["rocm_rocdl_path_test.cc"],
    tags = tf_gpu_tests_tags(),
    deps = [
        ":rocm_rocdl_path",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
    ]),
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
        "env_time.h",
        "logging.h",
        "macros.h",
        "platform_strings.h",
        "threadpool.h",
        "threadpool_interface.h",
        "threadpool_options.h",
        "tstring.h",
        "types.h",
        ":ctstring",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "abi.h",
        "base64.h",
        "bfloat16.h",
        "casts.h",
        "coding.h",
        "context.h",
        "cord.h",
        "cpu_feature_guard.h",
        "cpu_info.h",
        "crash_analysis.h",
        "dynamic_annotations.h",
        "env.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "fingerprint.h",
        "float8.h",
        "init_main.h",
        "logger.h",
        "mem.h",
        "mutex.h",
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
        "status.h",
        "statusor.h",
        "str_util.h",
        "strcat.h",
        "stringpiece.h",
        "stringprintf.h",
        "strong_hash.h",
        "subprocess.h",
        "thread_annotations.h",
        ":base_hdrs",
        "//tensorflow/core/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/core/platform/profile_utils:clock_cycle_profiler.h",
        "//tensorflow/core/platform/profile_utils:cpu_utils.h",
        "//tensorflow/core/platform/profile_utils:i_cpu_utils_helper.h",
        "//tensorflow/tsl/platform:base_hdrs",
        "//tensorflow/tsl/platform:lib_hdrs",
        # copybara:uncomment(GoogleFileSystem) "//tensorflow/tsl/platform/google:env.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_proto_parsing_hdrs",
    srcs = [
        "init_main.h",
        "logging.h",
        "macros.h",
        "platform.h",
        "protobuf.h",
        "stringpiece.h",
        "tstring.h",
        "types.h",
        ":ctstring",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "test_hdrs",
    testonly = 1,
    srcs = [
        "test.h",
        "test_benchmark.h",
        "//tensorflow/tsl/platform:test_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "android_test_srcs",
    testonly = 1,
    srcs = [
        "test.h",
        "//tensorflow/tsl/platform:android_test_srcs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
        "bfloat16.h",
        "byte_order.h",
        "cpu_info.h",
        "dynamic_annotations.h",
        "float8.h",
        "macros.h",
        "mutex.h",
        "platform.h",
        "prefetch.h",
        "protobuf.h",
        "thread_annotations.h",
        "tstring.h",
        "types.h",
        ":ctstring",
        "//tensorflow/tsl/platform:framework_lite_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "lib_internal_private_hdrs",
    srcs = [
        "bfloat16.h",
        "float8.h",
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
        "intrusive_ptr.h",
        "platform.h",
        "protobuf_internal.h",
        "refcount.h",
        "setround.h",
        "snappy.h",
        "tensor_coding.h",
        "tracing.h",
        "unbounded_work_queue.h",
        "//tensorflow/tsl/platform:lib_internal_public_hdrs",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "jpeg_hdrs",
    srcs = [
        "jpeg.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
    ],
)

filegroup(
    name = "gif_hdrs",
    srcs = [
        "gif.h",
    ],
    visibility = ["//tensorflow/core/lib/gif:__pkg__"],
)

filegroup(
    name = "tflite_portable_logging_hdrs",
    srcs = [
        "bfloat16.h",
        "float8.h",
        "logging.h",
        "macros.h",
        "platform.h",
        "tstring.h",
        "types.h",
        ":ctstring",
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
        "logging.h",
        "macros.h",
        "mem.h",
        "platform.h",
        "stringpiece.h",
        "tstring.h",
        "types.h",
        ":ctstring",
        "//tensorflow/tsl/platform:jpeg_internal_hdrs",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/jpeg:__pkg__",
    ],
)

filegroup(
    name = "gif_internal_hdrs",
    srcs = [
        "bfloat16.h",
        "dynamic_annotations.h",
        "logging.h",
        "macros.h",
        "mem.h",
        "platform.h",
        "tstring.h",
        "types.h",
        ":ctstring",
        "//tensorflow/tsl/platform:gif_internal_hdrs",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "abi.h",
        "bfloat16.h",
        "blocking_counter.h",
        "byte_order.h",
        "casts.h",
        "coding.h",
        "context.h",
        "cord.h",
        "cpu_info.h",
        "demangle.h",
        "denormal.h",
        "dynamic_annotations.h",
        "env.h",
        "env_time.h",
        "errors.h",
        "file_statistics.h",
        "file_system.h",
        "file_system_helper.h",
        "float8.h",
        "hash.h",
        "host_info.h",
        "init_main.h",
        "intrusive_ptr.h",
        "load_library.h",
        "logging.h",
        "macros.h",
        "mem.h",
        "mutex.h",
        "numa.h",
        "numbers.h",
        "path.h",
        "platform.h",
        "prefetch.h",
        "protobuf.h",
        "ram_file_system.h",
        "raw_coding.h",
        "refcount.h",
        "regexp.h",
        "resource.h",
        "scanner.h",
        "setround.h",
        "snappy.h",
        "stack_frame.h",
        "stacktrace.h",
        "status.h",
        "statusor.h",
        "str_util.h",
        "strcat.h",
        "stringpiece.h",
        "stringprintf.h",
        "tensor_coding.cc",
        "tensor_coding.h",
        "thread_annotations.h",
        "threadpool.h",
        "threadpool_interface.h",
        "tracing.h",
        "tstring.h",
        "types.h",
        ":ctstring",
        "//tensorflow/tsl/platform:mobile_srcs_no_runtime",
    ] + tf_platform_alias("additional_mobile_srcs_no_runtime"),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "base64.h",
        "cpu_feature_guard.cc",
        "cpu_feature_guard.h",
        "fingerprint.h",
        "notification.h",
        "platform_strings.cc",
        "platform_strings.h",
        "platform_strings_computed.h",
        "png.h",
        "protobuf_internal.h",
        "random.h",
        "test_benchmark.h",
        "threadpool_options.h",
        "unbounded_work_queue.h",
        "//tensorflow/core/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/core/platform/profile_utils:cpu_utils.h",
        "//tensorflow/core/platform/profile_utils:i_cpu_utils_helper.h",
        "//tensorflow/tsl/platform:mobile_srcs_only_runtime",
        "//tensorflow/tsl/platform/default:mobile_srcs_only_runtime",
    ] + if_not_fuchsia([
        "subprocess.h",
    ]) + tf_google_mobile_srcs_only_runtime(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_headers",
    srcs = glob(
        [
            "*.h",
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
    ) + [
        "//tensorflow/core/platform/profile_utils:android_armv7a_cpu_utils_helper.h",
        "//tensorflow/core/platform/profile_utils:clock_cycle_profiler.h",
        "//tensorflow/core/platform/profile_utils:cpu_utils.h",
        "//tensorflow/core/platform/profile_utils:i_cpu_utils_helper.h",
        "//tensorflow/tsl/platform:legacy_lib_internal_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

alias(
    name = "legacy_lib_internal_srcs",
    actual = "//tensorflow/core/platform/profile_utils:legacy_lib_internal_srcs",
    visibility = ["//tensorflow/core:__pkg__"],
)

bzl_library(
    name = "build_config_root_bzl",
    srcs = [
        "build_config_root.bzl",
        "build_config_root.default.bzl",
    ] + tf_platform_alias("build_config_root.bzl"),
    visibility = [
        "//learning/brain/google/xla/tests:__pkg__",
        "//learning/brain/tfrc/runtime/tpu_driver:__subpackages__",
        "//nlp/deleuze:__pkg__",
        "//tensorflow:__subpackages__",
    ],
)

bzl_library(
    name = "rules_cc_bzl",
    srcs = ["rules_cc.bzl"],
    deps = tf_platform_alias("rules_cc_bzl"),
)
