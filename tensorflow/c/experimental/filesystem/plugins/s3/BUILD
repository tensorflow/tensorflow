# Experimental s3 filesystem plugin.
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

# Filesystem implementation for AWS S3.
tf_cc_shared_object(
    name = "s3_filesystem",
    framework_so = [],
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [":s3_filesystem_impl"],
    per_os_targets = 1,
)

# The real implementation of the filesystem.
cc_library(
    name = "s3_filesystem_impl",
    srcs = ["s3_filesystem.cc"],
    copts = select({
        "//conditions:default": ["-std=c++17"],
        "//tensorflow:macos": ["-lc++fs", "-std=c++17"],
        "//tensorflow:windows": ["/std:c++17"],
    }),
    deps = [
        ":s3_shared",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)

cc_library(
    name = "s3_shared",
    srcs = [
        "s3_shared.cc",
    ],
    hdrs = [
        "s3_shared.h",
    ],
    deps = [
        "@aws",
    ]
)
