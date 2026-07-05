load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    ["LICENSE"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "gen_gpu_kernel_list",
    srcs = ["gen_gpu_kernel_list.py"],
    visibility = [
        "@onednn_async//:__subpackages__",
    ],
)
