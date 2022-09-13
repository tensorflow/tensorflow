# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "check_deps",
    "tf_copts",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "determinism.cc",
        "determinism.h",
        "env_var.cc",
        "env_var.h",
    ],
)

filegroup(
    name = "determnism_hdr",
    srcs = [
        "determinism.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow/core/util:__pkg__",
    ],
)

filegroup(
    name = "framework_internal_private_hdrs",
    srcs = [
        "env_var.h",
    ],
)

filegroup(
    name = "lib_internal_public_hdrs",
    srcs = [
        "env_var.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/util:__pkg__",
    ],
)

filegroup(
    name = "determinism_hdr",
    srcs = [
        "determinism.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow/core/util:__pkg__",
    ],
)

cc_library(
    name = "determinism_hdr_lib",
    hdrs = [":determinism_hdr"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
)

# Note: This rule should not be used as a dependency for kernels. Use the
# "determinism_for_kernels" rule below instead.
cc_library(
    name = "determinism",
    srcs = ["determinism.cc"],
    hdrs = ["determinism.h"],
    copts = tf_copts(),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":env_var",
        "//tensorflow/tsl/platform:mutex",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

# This alias should be used as a dependency for kernels which use determinism,
# as well any other rules which are in the same shared library as the kernels.
# This rule does not include the determinism.cc file for nonstatic builds. The
# reason is that for nonstatic builds, the shared object which contains the
# kernels (e.g. _pywrap_tensorflow_internal.so) must not contain the global
# variable in determinism.cc, since the global variable is already in
# libtensorflow_framework.so.
#
# To test that determinism.cc is not improperly included in the shared object
# which contains the kernels, you can run the "determinism_check_deps" rule
# below.
alias(
    name = "determinism_for_kernels",
    actual = if_static(":determinism", ":determinism_hdr_lib"),
    visibility = ["//tensorflow:__subpackages__"],
)

check_deps(
    name = "determinism_check_deps",
    disallowed_deps = if_static(
        [],
        otherwise = [":determinism"],
    ),
    deps = [
        "//tensorflow:tensorflow_cc",
    ],
)

cc_library(
    name = "determinism_test_util",
    hdrs = [":determinism_test_util.h"],
    data = [
        # Adding this data dependency ensures determinism_check_deps is run
        # whenever determinism tests are run.
        ":determinism_check_deps",
    ],
    deps = [":determinism"],
)

cc_library(
    name = "env_var",
    srcs = ["env_var.cc"],
    hdrs = ["env_var.h"],
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
    ],
)
