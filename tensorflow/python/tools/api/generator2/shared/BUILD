load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/python/tools/api/generator2:__subpackages__"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "exported_api",
    srcs = ["exported_api.py"],
)

py_strict_test(
    name = "exported_api_test",
    srcs = ["exported_api_test.py"],
    tags = ["no_pip"],
    deps = [
        ":exported_api",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/platform:client_testlib",
    ],
)
