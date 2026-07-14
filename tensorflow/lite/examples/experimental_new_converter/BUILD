load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "stack_trace_example",
    srcs = ["stack_trace_example.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "@absl_py//absl:app",
    ],
)
