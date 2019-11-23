"""
Build targets for default implementations of tf/core/platform libraries.
"""
# This is a temporary hack to mimic the presence of a BUILD file under
# tensorflow/core/platform/default. This is part of a large refactoring
# of BUILD rules under tensorflow/core/platform. We will remove this file
# and add real BUILD files under tensorflow/core/platform/default and
# tensorflow/core/platform/windows after the refactoring is complete.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

TF_DEFAULT_PLATFORM_LIBRARIES = {
    "context": {
        "name": "context_impl",
        "hdrs": ["//tensorflow/core/platform:context.h"],
        "textual_hdrs": ["//tensorflow/core/platform:default/context.h"],
        "deps": [
            "//tensorflow/core/platform",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "cord": {
        "name": "cord_impl",
        "hdrs": ["//tensorflow/core/platform:default/cord.h"],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "cuda_libdevice_path": {
        "name": "cuda_libdevice_path_impl",
        "hdrs": [
            "//tensorflow/core/platform:cuda_libdevice_path.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/cuda_libdevice_path.cc",
        ],
        "deps": [
            "@local_config_cuda//cuda:cuda_headers",
            "//tensorflow/core:lib",
            # TODO(bmzhao): When bazel gains cc_shared_library support, the targets below are
            # the actual granular targets we should depend on, instead of tf/core:lib.
            # "//tensorflow/core/platform:logging",
            # "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "dynamic_annotations": {
        "name": "dynamic_annotations_impl",
        "hdrs": [
            "//tensorflow/core/platform:default/dynamic_annotations.h",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "env": {
        "name": "env_impl",
        "hdrs": [
            "//tensorflow/core/platform:env.h",
            "//tensorflow/core/platform:file_system.h",
            "//tensorflow/core/platform:file_system_helper.h",
            "//tensorflow/core/platform:threadpool.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:env.cc",
            "//tensorflow/core/platform:file_system.cc",
            "//tensorflow/core/platform:file_system_helper.cc",
            "//tensorflow/core/platform:threadpool.cc",
            "//tensorflow/core/platform:default/env.cc",
            "//tensorflow/core/platform:default/posix_file_system.h",
            "//tensorflow/core/platform:default/posix_file_system.cc",
        ],
        "deps": [
            "@com_google_absl//absl/time",
            "@com_google_absl//absl/types:optional",
            "//third_party/eigen3",
            "//tensorflow/core/lib/core:blocking_counter",
            "//tensorflow/core/lib/core:error_codes_proto_cc",
            "//tensorflow/core/lib/core:errors",
            "//tensorflow/core/lib/core:status",
            "//tensorflow/core/lib/core:stringpiece",
            "//tensorflow/core/lib/io:path",
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:context",
            "//tensorflow/core/platform:cord",
            "//tensorflow/core/platform:denormal",
            "//tensorflow/core/platform:error",
            "//tensorflow/core/platform:env_time",
            "//tensorflow/core/platform:file_statistics",
            "//tensorflow/core/platform:load_library",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform:platform_port",
            "//tensorflow/core/platform:protobuf",
            "//tensorflow/core/platform:setround",
            "//tensorflow/core/platform:stringpiece",
            "//tensorflow/core/platform:stringprintf",
            "//tensorflow/core/platform:strcat",
            "//tensorflow/core/platform:str_util",
            "//tensorflow/core/platform:threadpool_interface",
            "//tensorflow/core/platform:tracing",
            "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "env_time": {
        "name": "env_time_impl",
        "hdrs": [
            "//tensorflow/core/platform:env_time.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/env_time.cc",
        ],
        "deps": [
            "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "human_readable_json": {
        "name": "human_readable_json_impl",
        "hdrs": [
            "//tensorflow/core/platform:human_readable_json.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/human_readable_json.cc",
        ],
        "deps": [
            "//tensorflow/core/lib/core:errors",
            "//tensorflow/core/lib/core:status",
            "//tensorflow/core/platform:strcat",
            "//tensorflow/core/platform:protobuf",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "load_library": {
        "name": "load_library_impl",
        "hdrs": [
            "//tensorflow/core/platform:load_library.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/load_library.cc",
        ],
        "deps": [
            "//tensorflow/core/lib/core:errors",
            "//tensorflow/core/lib/core:status",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "logging": {
        "name": "logging_impl",
        "hdrs": [
            "//tensorflow/core/platform:logging.h",
        ],
        "textual_hdrs": [
            "//tensorflow/core/platform:default/logging.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/logging.cc",
        ],
        "deps": [
            "@com_google_absl//absl/base",
            "@com_google_absl//absl/strings",
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:env_time",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "mutex": {
        "name": "mutex_impl",
        "hdrs": [
            "//tensorflow/core/platform:mutex.h",
        ],
        "textual_hdrs": [
            "//tensorflow/core/platform:default/mutex.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/mutex.cc",
            "//tensorflow/core/platform:default/mutex_data.h",
        ],
        "deps": [
            "@nsync//:nsync_cpp",
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:thread_annotations",
            "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "net": {
        "name": "net_impl",
        "hdrs": [
            "//tensorflow/core/platform:net.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/net.cc",
        ],
        "deps": [
            "//tensorflow/core/platform:strcat",
            "//tensorflow/core/platform:logging",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
        "alwayslink": 1,
    },
    "notification": {
        "name": "notification_impl",
        "hdrs": [
            "//tensorflow/core/platform:default/notification.h",
        ],
        "deps": [
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "rocm_rocdl_path": {
        "name": "rocm_rocdl_path_impl",
        "hdrs": [
            "//tensorflow/core/platform:rocm_rocdl_path.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/rocm_rocdl_path.cc",
        ],
        "deps": [
            "@local_config_rocm//rocm:rocm_headers",
            "//tensorflow/core:lib",
            # TODO(bmzhao): When bazel gains cc_shared_library support, the targets below are
            # the actual granular targets we should depend on, instead of tf/core:lib.
            # "//tensorflow/core/lib/io:path",
            # "//tensorflow/core/platform:logging",
            # "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "stacktrace": {
        "name": "stacktrace_impl",
        "hdrs": [
            "//tensorflow/core/platform:default/stacktrace.h",
        ],
        "deps": [
            "//tensorflow/core/platform:abi",
            "//tensorflow/core/platform:platform",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
    },
    "stacktrace_handler": {
        "name": "stacktrace_handler_impl",
        "hdrs": [
            "//tensorflow/core/platform:stacktrace_handler.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/stacktrace_handler.cc",
        ],
        "deps": [
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:stacktrace",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
    },
    "strong_hash": {
        "name": "strong_hash_impl",
        "textual_hdrs": [
            "//tensorflow/core/platform:default/strong_hash.h",
        ],
        "deps": [
            "@highwayhash//:sip_hash",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual"],
    },
    "subprocess": {
        "name": "subprocess_impl",
        "textual_hdrs": [
            "//tensorflow/core/platform:default/subprocess.h",
        ],
        "hdrs": [
            "//tensorflow/core/platform:subprocess.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/subprocess.cc",
        ],
        "deps": [
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform:types",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
        "alwayslink": 1,
    },
    "test": {
        "name": "test_impl",
        "testonly": True,
        "srcs": [
            "//tensorflow/core/platform:default/test.cc",
        ],
        "hdrs": [
            "//tensorflow/core/platform:test.h",
        ],
        "deps": [
            "@com_google_googletest//:gtest",
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:net",
            "//tensorflow/core/platform:strcat",
            "//tensorflow/core/platform:types",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
    },
    "tracing": {
        "name": "tracing_impl",
        "textual_hdrs": [
            "//tensorflow/core/platform:default/tracing_impl.h",
        ],
        "hdrs": [
            "//tensorflow/core/platform:tracing.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/tracing.cc",
            "//tensorflow/core/platform:tracing.cc",
        ],
        "deps": [
            "//tensorflow/core/lib/hash",
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:strcat",
            "//tensorflow/core/platform:str_util",
            "//tensorflow/core/platform:stringpiece",
            "//tensorflow/core/platform:types",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
        "alwayslink": 1,
    },
    "types": {
        "name": "types_impl",
        "textual_hdrs": [
            "//tensorflow/core/platform:default/integral_types.h",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
    },
    "unbounded_work_queue": {
        "name": "unbounded_work_queue_impl",
        "hdrs": [
            "//tensorflow/core/platform:default/unbounded_work_queue.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:default/unbounded_work_queue.cc",
        ],
        "deps": [
            "@com_google_absl//absl/memory",
            "//tensorflow/core/platform:env",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/lib/core:notification",
        ],
        "tags": ["no_oss", "manual"],
        "visibility": ["//visibility:private"],
    },
}

TF_WINDOWS_PLATFORM_LIBRARIES = {
    "env": {
        "name": "windows_env_impl",
        "hdrs": [
            "//tensorflow/core/platform:env.h",
            "//tensorflow/core/platform:file_system.h",
            "//tensorflow/core/platform:file_system_helper.h",
            "//tensorflow/core/platform:threadpool.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:env.cc",
            "//tensorflow/core/platform:file_system.cc",
            "//tensorflow/core/platform:file_system_helper.cc",
            "//tensorflow/core/platform:threadpool.cc",
            "//tensorflow/core/platform:windows/env.cc",
            "//tensorflow/core/platform:windows/windows_file_system.h",
            "//tensorflow/core/platform:windows/windows_file_system.cc",
        ],
        "deps": [
            "@com_google_absl//absl/time",
            "@com_google_absl//absl/types:optional",
            "//third_party/eigen3",
            "//tensorflow/core/lib/core:blocking_counter",
            "//tensorflow/core/lib/core:error_codes_proto_cc",
            "//tensorflow/core/lib/core:errors",
            "//tensorflow/core/lib/core:status",
            "//tensorflow/core/lib/core:stringpiece",
            "//tensorflow/core/lib/io:path",
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:context",
            "//tensorflow/core/platform:cord",
            "//tensorflow/core/platform:denormal",
            "//tensorflow/core/platform:error",
            "//tensorflow/core/platform:env_time",
            "//tensorflow/core/platform:file_statistics",
            "//tensorflow/core/platform:load_library",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform:platform_port",
            "//tensorflow/core/platform:protobuf",
            "//tensorflow/core/platform:setround",
            "//tensorflow/core/platform:stringpiece",
            "//tensorflow/core/platform:stringprintf",
            "//tensorflow/core/platform:strcat",
            "//tensorflow/core/platform:str_util",
            "//tensorflow/core/platform:threadpool_interface",
            "//tensorflow/core/platform:tracing",
            "//tensorflow/core/platform:types",
            "//tensorflow/core/platform:windows_wide_char_impl",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual", "nobuilder"],
    },
    "env_time": {
        "name": "windows_env_time_impl",
        "hdrs": [
            "//tensorflow/core/platform:env_time.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:windows/env_time.cc",
        ],
        "deps": [
            "//tensorflow/core/platform:types",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual", "nobuilder"],
    },
    "load_library": {
        "name": "windows_load_library_impl",
        "hdrs": [
            "//tensorflow/core/platform:load_library.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:windows/load_library.cc",
        ],
        "deps": [
            "//tensorflow/core/lib/core:errors",
            "//tensorflow/core/lib/core:status",
            "//tensorflow/core/platform:windows_wide_char_impl",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual", "nobuilder"],
    },
    "net": {
        "name": "windows_net_impl",
        "hdrs": [
            "//tensorflow/core/platform:net.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:windows/net.cc",
        ],
        "deps": [
            "//tensorflow/core/platform:error",
            "//tensorflow/core/platform:logging",
        ],
        "visibility": ["//visibility:private"],
        "tags": ["no_oss", "manual", "nobuilder"],
    },
    "stacktrace": {
        "name": "windows_stacktrace_impl",
        "hdrs": [
            "//tensorflow/core/platform:windows/stacktrace.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:windows/stacktrace.cc",
        ],
        "deps": [
            "//tensorflow/core/platform:mutex",
        ],
        "tags": ["no_oss", "manual", "nobuilder"],
        "visibility": ["//visibility:private"],
    },
    "stacktrace_handler": {
        "name": "windows_stacktrace_handler_impl",
        "hdrs": [
            "//tensorflow/core/platform:stacktrace_handler.h",
        ],
        "srcs": [
            "//tensorflow/core/platform:windows/stacktrace_handler.cc",
        ],
        "deps": [
            "//tensorflow/core/platform:mutex",
            "//tensorflow/core/platform:stacktrace",
            "//tensorflow/core/platform:types",
        ],
        "tags": ["no_oss", "manual", "nobuilder"],
        "visibility": ["//visibility:private"],
    },
    "subprocess": {
        "name": "windows_subprocess_impl",
        "textual_hdrs": [
            "//tensorflow/core/platform:windows/subprocess.h",
        ],
        "hdrs": [
            "//tensorflow/core/platform:subprocess.h",
        ],
        "deps": [
            "//tensorflow/core/platform",
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:macros",
            "//tensorflow/core/platform:types",
        ],
        "tags": ["no_oss", "manual", "nobuilder"],
        "visibility": ["//visibility:private"],
    },
    "wide_char": {
        "name": "windows_wide_char_impl",
        "hdrs": [
            "//tensorflow/core/platform:windows/wide_char.h",
        ],
        "tags": ["no_oss", "manual", "nobuilder"],
        "visibility": ["//visibility:private"],
    },
}

def tf_instantiate_platform_libraries(names = []):
    for name in names:
        # Unfortunately, this target cannot be represented as a dictionary
        # because it uses "select"
        if name == "platform_port":
            native.cc_library(
                name = "platform_port_impl",
                srcs = [
                    "//tensorflow/core/platform:cpu_info.cc",
                    "//tensorflow/core/platform:default/port.cc",
                ],
                hdrs = [
                    "//tensorflow/core/platform:cpu_info.h",
                    "//tensorflow/core/platform:demangle.h",
                    "//tensorflow/core/platform:host_info.h",
                    "//tensorflow/core/platform:init_main.h",
                    "//tensorflow/core/platform:mem.h",
                    "//tensorflow/core/platform:numa.h",
                    "//tensorflow/core/platform:snappy.h",
                ],
                defines = ["TF_USE_SNAPPY"] + select({
                    # TF Additional NUMA defines
                    "//tensorflow:with_numa_support": ["TENSORFLOW_USE_NUMA"],
                    "//conditions:default": [],
                }),
                copts = tf_copts(),
                deps = [
                    "@com_google_absl//absl/base",
                    "//tensorflow/core/platform:byte_order",
                    "//tensorflow/core/platform:dynamic_annotations",
                    "//tensorflow/core/platform:logging",
                    "//tensorflow/core/platform:types",
                    "//tensorflow/core/platform",
                    "@snappy",
                ] + select({
                    # TF Additional NUMA dependencies
                    "//tensorflow:android": [],
                    "//tensorflow:ios": [],
                    "//tensorflow:macos": [],
                    "//conditions:default": [
                        "@hwloc",
                    ],
                }),
                visibility = ["//visibility:private"],
                tags = ["no_oss", "manual"],
            )
            native.cc_library(
                name = "windows_platform_port_impl",
                srcs = [
                    "//tensorflow/core/platform:cpu_info.cc",
                    "//tensorflow/core/platform:windows/port.cc",
                ],
                hdrs = [
                    "//tensorflow/core/platform:cpu_info.h",
                    "//tensorflow/core/platform:demangle.h",
                    "//tensorflow/core/platform:host_info.h",
                    "//tensorflow/core/platform:init_main.h",
                    "//tensorflow/core/platform:mem.h",
                    "//tensorflow/core/platform:numa.h",
                    "//tensorflow/core/platform:snappy.h",
                ],
                defines = ["TF_USE_SNAPPY"],
                copts = tf_copts(),
                deps = [
                    "//tensorflow/core/platform",
                    "//tensorflow/core/platform:byte_order",
                    "//tensorflow/core/platform:dynamic_annotations",
                    "//tensorflow/core/platform:logging",
                    "//tensorflow/core/platform:types",
                    "@snappy",
                ],
                visibility = ["//visibility:private"],
                tags = ["no_oss", "manual"],
            )
        else:
            if name in TF_DEFAULT_PLATFORM_LIBRARIES:
                native.cc_library(**TF_DEFAULT_PLATFORM_LIBRARIES[name])
            if name in TF_WINDOWS_PLATFORM_LIBRARIES:
                native.cc_library(**TF_WINDOWS_PLATFORM_LIBRARIES[name])

def tf_mobile_aware_deps(name):
    return [":" + name]

def tf_platform_helper_deps(name):
    return select({
        "//tensorflow:windows": [":windows_" + name],
        "//conditions:default": [":" + name],
    })

def tf_logging_deps():
    return [":logging_impl"]
