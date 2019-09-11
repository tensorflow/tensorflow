"""
Build targets for default implementations of tf/core/platform libraries.
"""
# This is a temporary hack to mimic the presence of a BUILD file under
# tensorflow/core/platform/default. This is part of a large refactoring
# of BUILD rules under tensorflow/core/platform. We will remove this file
# and add real BUILD files under tensorflow/core/platform/default and
# tensorflow/core/platform/windows after the refactoring is complete.

load(
    "//tensorflow/core/platform:default/build_config.bzl",
    "tf_additional_numa_copts",
    "tf_additional_numa_deps",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

TF_PLATFORM_LIBRARIES = {
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
            "//tensorflow/core/platform:logging",
            "//tensorflow/core/platform:types",
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
    "fingerprint": {
        "name": "fingerprint_impl",
        "textual_hdrs": [
            "//tensorflow/core/platform:default/fingerprint.h",
        ],
        "deps": [
            "@farmhash_archive//:farmhash",
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
            "//tensorflow/core/lib/strings:string_utils",
            "//tensorflow/core/platform:protobuf",
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
}

TF_WINDOWS_PLATFORM_LIBRARIES = {
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
        "tags": ["no_oss", "manual"],
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
                copts = tf_copts() + tf_additional_numa_copts(),
                deps = [
                    "@com_google_absl//absl/base",
                    "//tensorflow/core/platform:byte_order",
                    "//tensorflow/core/platform:dynamic_annotations",
                    "//tensorflow/core/platform:logging",
                    "//tensorflow/core/platform:types",
                    "//tensorflow/core/platform",
                    "@snappy",
                ] + tf_additional_numa_deps(),
                visibility = ["//visibility:private"],
                tags = ["no_oss", "manual"],
            )
            native.cc_library(
                name = "windows_platform_port_impl",
                srcs = [
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
            native.cc_library(**TF_PLATFORM_LIBRARIES[name])
            if name in TF_WINDOWS_PLATFORM_LIBRARIES:
                native.cc_library(**TF_WINDOWS_PLATFORM_LIBRARIES[name])

def tf_platform_helper_deps(name):
    return select({
        "//tensorflow:windows": [":windows_" + name],
        "//conditions:default": [":" + name],
    })
