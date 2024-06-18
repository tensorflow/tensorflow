# TensorFlow Lite for iOS

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library")

# LINT.IfChange

sh_binary(
    name = "hide_symbols_with_allowlist",
    srcs = [
        "hide_symbols_with_allowlist.sh",
    ],
    visibility = [
        "//tensorflow/lite:__subpackages__",
        "//tensorflow_lite_support:__subpackages__",
    ],
)

bzl_library(
    name = "ios_bzl",
    srcs = ["ios.bzl"],
    visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow:tensorflow_bzl",
        "//tools/build_defs/apple",
        "//tools/build_defs/apple:apple_genrule",
        "//tools/build_defs/apple:ios",
    ],
)

pytype_strict_library(
    name = "extract_object_files",
    srcs = [
        "extract_object_files.py",
    ],
)

pytype_strict_binary(
    name = "extract_object_files_main",
    srcs = [
        "extract_object_files_main.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = [
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow_lite_support:__subpackages__",
    ],
    deps = [
        ":extract_object_files",
    ],
)

# LINT.ThenChange(//tensorflow/lite/ios/BUILD)
