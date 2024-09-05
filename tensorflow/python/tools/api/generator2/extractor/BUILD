load("//tensorflow:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/tools/api/generator2:__subpackages__"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "extractor",
    srcs = ["extractor.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/tools/api/generator2/shared:exported_api",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_binary(
    name = "main",
    srcs = ["main.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":extractor",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "extractor_test",
    srcs = ["extractor_test.py"],
    tags = ["no_pip"],
    deps = [
        ":extractor",
        "@absl_py//absl/testing:absltest",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/tools/api/generator2/shared:exported_api",
    ],
)
