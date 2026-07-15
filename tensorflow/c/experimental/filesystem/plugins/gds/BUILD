# Experimental GPUDirect Storage (GDS) filesystem plugin.
#
# Requires the CUDA GDS package (cuFile) to be installed on the build
# machine: headers under $CUDA_HOME/include (cufile.h) and
# libcufile.so under $CUDA_HOME/lib64/. This target is Linux + NVIDIA-only;
# it is not built as part of the default `//tensorflow/...` build and must
# be requested explicitly, mirroring how other optional accelerator
# backends are gated in this repo.
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
    deps = [":gds_filesystem_impl"],
)

cc_library(
    name = "gds_filesystem_impl",
    srcs = ["gds_filesystem.cc"],
    hdrs = ["gds_filesystem.h"],
    deps = [
        "//tensorflow/c:tf_file_statistics",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "@local_config_cuda//cuda:cufile",  # NEW target; see cuda BUILD note below
    ],
)

# Static registration target, mirroring posix_filesystem_static, for anyone
# who wants to link GDS support directly into a binary rather than loading
# it as a plugin DSO.
cc_library(
    name = "gds_filesystem_static",
    srcs = ["gds_filesystem_static.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":gds_filesystem_impl",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "//tensorflow/c/experimental/filesystem:modular_filesystem",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/log",
    ],
    alwayslink = 1,
)