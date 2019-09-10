"""
Build targets for default implementations of tf/core/platform libraries.
"""
# This is a temporary hack to mimic the presence of a BUILD file under
# tensorflow/core/platform/default. This is part of a large refactoring
# of BUILD rules under tensorflow/core/platform. We will remove this file
# and add real BUILD files under tensorflow/core/platform/default and
# tensorflow/core/platform/windows after the refactoring is complete.

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
        native.cc_library(**TF_PLATFORM_LIBRARIES[name])
        if name in TF_WINDOWS_PLATFORM_LIBRARIES:
            native.cc_library(**TF_WINDOWS_PLATFORM_LIBRARIES[name])

def tf_platform_helper_deps(name):
    return select({
        "//tensorflow:windows": [":windows_" + name],
        "//conditions:default": [":" + name],
    })
