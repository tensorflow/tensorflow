# Experimental GPUDirect Storage (GDS) filesystem plugin.
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

tf_cc_shared_object(
    name = "libgds_filesystem.so",
    framework_so = [],
    linkstatic = False,
    visibility = ["//visibility:public"],
    target_compatible_with = ["@platforms//os:linux"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":gds_filesystem_impl"],
)

cc_library(
    name = "gds_filesystem_impl",
    srcs = ["gds_filesystem.cc"],
    hdrs = ["gds_filesystem.h"],
    target_compatible_with = ["@platforms//os:linux"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/c:tf_file_statistics",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    linkopts = ["-lcufile"],
)