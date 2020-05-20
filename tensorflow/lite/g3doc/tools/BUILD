package(
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "build_py_api_docs",
    srcs = ["build_py_api_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
