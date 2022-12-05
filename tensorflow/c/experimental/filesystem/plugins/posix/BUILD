load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental posix filesystem plugin.
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

# Filesystem implementation for POSIX environments: Linux, MacOS, Android, etc.
tf_cc_shared_object(
    name = "libposix_filesystem.so",
    framework_so = [],
    linkstatic = False,
    visibility = ["//visibility:public"],
    deps = [":posix_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "posix_filesystem_impl",
    srcs = ["posix_filesystem.cc"],
    hdrs = ["posix_filesystem.h"],
    deps = [
        ":posix_filesystem_helper",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
    ],
)

# Since building pip package and API tests require a filesystem, we provide a
# static registration target that they should link against.
cc_library(
    name = "posix_filesystem_static",
    srcs = ["posix_filesystem_static.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":posix_filesystem_impl",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "//tensorflow/c/experimental/filesystem:modular_filesystem",
    ],
    alwayslink = 1,
)

# Library implementing helper functionality, so that the above only contains
# the API implementation for modular filesystems.
cc_library(
    name = "posix_filesystem_helper",
    srcs = ["posix_filesystem_helper.cc"],
    hdrs = ["posix_filesystem_helper.h"],
    deps = [":copy_file"],
)

# On Linux, we can copy files faster using `sendfile`. But not elsewhere.
# Hence, this private library to select which implementation to use.
cc_library(
    name = "copy_file",
    srcs = select({
        "//tensorflow:linux_x86_64": ["copy_file_linux.cc"],
        "//conditions:default": ["copy_file_portable.cc"],
    }),
    hdrs = ["copy_file.h"],
)
