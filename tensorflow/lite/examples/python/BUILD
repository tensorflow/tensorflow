load("//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "label_image",
    srcs = ["label_image.py"],
    main = "label_image.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
    ],
)
