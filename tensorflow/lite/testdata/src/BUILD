package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "intermediate_tensor_output",
    srcs = ["intermediate_tensor_output.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)
