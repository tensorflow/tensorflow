load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_python_pybind_extension(
    name = "mlir_extension",
    srcs = [
        "mlir.cc",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
        "-Wno-c++98-c++11-compat",
    ],
    features = ["-use_header_modules"],
    module_name = "mlir_extension",
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@llvm//:support",
        "@pybind11",
    ],
)

py_test(
    name = "mlir_test",
    srcs = ["mlir_test.py"],
    main = "mlir_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":mlir_extension",
        "//tensorflow/python:client_testlib",
    ],
)
