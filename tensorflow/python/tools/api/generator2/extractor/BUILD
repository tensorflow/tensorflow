load("//tensorflow:pytype.default.bzl", "pytype_strict_binary", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/tools/api/generator2:__subpackages__"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "parser",
    srcs = ["parser.py"],
    deps = [
        "//tensorflow/python/tools/api/generator2/shared:exported_api",
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    tags = ["no_pip"],
    deps = [
        ":parser",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tools/api/generator2/shared:exported_api",
    ],
)

pytype_strict_binary(
    name = "extractor",
    srcs = ["extractor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":parser",
        "//tensorflow/python/tools/api/generator2/shared:exported_api",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
