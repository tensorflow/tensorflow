load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental filesystem C APIs for TensorFlow.
# Will be moved in proper place once all filesystems are converted to the
# modular framework.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],
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

# Core TensorFlow depends on this, will be included in main library
cc_library(
    name = "modular_filesystem",
    srcs = [
        "modular_filesystem.cc",
        "modular_filesystem_registration.cc",
    ],
    hdrs = [
        "modular_filesystem.h",
        "modular_filesystem_registration.h",
    ],
    # TODO(mihaimaruseac): Visibility should be more restrictive once we
    # convert to modular filesystems everywhere
    visibility = ["//visibility:public"],
    deps = [
        ":filesystem_interface",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ],
)

# Compliance test for modules and for interface
tf_cc_test(
    name = "modular_filesystem_test",
    size = "small",
    srcs = ["modular_filesystem_test.cc"],
    linkopts = ["-ldl"],
    tags = [
        "manual",  # Requires DSOs as arguments, eventual setup
        "notap",  # b/139060984, requires implementing modular support for Google filesystem
    ],
    deps = [
        ":modular_filesystem",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/lib/io:path",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:error",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/platform:test",
    ],
)
