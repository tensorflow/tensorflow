# Experimental filesystem C APIs for TensorFlow.
# Will be moved in proper place once all filesystems are converted to the
# modular framework.

package(
    licenses = ["notice"],  # Apache 2.0
)

# This is only for plugins
cc_library(
    name = "filesystem_interface",
    hdrs = ["filesystem_interface.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:tf_file_statistics",
        "//tensorflow/c:tf_status",
    ],
)

# Core TensorFlow depends on this, this will be included in main library
cc_library(
    name = "filesystem_interface_impl",
    srcs = ["filesystem_interface.cc"],
    hdrs = ["filesystem_interface.h"],
    deps = [
        ":modular_filesystem",
        "//tensorflow/c:tf_file_statistics",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
    ],
    alwayslink = 1,
)

# Core TensorFlow depends on this, will be included in main library
cc_library(
    name = "modular_filesystem",
    srcs = ["modular_filesystem.cc"],
    hdrs = ["modular_filesystem.h"],
    deps = [
        ":filesystem_interface",
        "//tensorflow/core/platform:env",
    ],
)
