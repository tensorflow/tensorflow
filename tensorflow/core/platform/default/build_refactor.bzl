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
    },
}

def tf_instantiate_platform_libraries(names = []):
    for name in names:
        native.cc_library(**TF_PLATFORM_LIBRARIES[name])
