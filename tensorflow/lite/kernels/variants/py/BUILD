load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.default.bzl", "pybind_extension")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:LICENSE"])

pybind_extension(
    name = "register_list_ops_py",
    srcs = ["register_list_ops_py.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "register_list_ops_py.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/kernels/variants:list_ops_lib",
        "@pybind11",
    ],
)

py_test(
    name = "end_to_end_test",
    srcs = ["end_to_end_test.py"],
    strict_deps = True,
    tags = ["nochromiumos_arm"],
    deps = [
        ":register_list_ops_py",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/platform:test",
    ],
)
