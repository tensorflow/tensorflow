# Experimental filesystem C APIs for TensorFlow.
# Will be moved in proper place once all filesystems are converted to the
# modular framework.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    # TODO(b/139060984): Visibility should be more restrictive once we
    # convert to modular filesystems everywhere
    visibility = ["//visibility:public"],
    deps = [
        ":filesystem_interface",
        "//tensorflow/c:tf_file_statistics",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:file_statistics",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/platform:test",
    ],
)
