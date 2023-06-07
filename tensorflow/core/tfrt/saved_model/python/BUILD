# This package loads pybind11 dependencies to ensure compatibility with Python API.
# Files affected: saved_model_aot_compile

load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/core/tfrt/saved_model/...",
    ],
)

py_binary(
    name = "saved_model_aot_compile_py",
    srcs = ["saved_model_aot_compile.py"],
    main = "saved_model_aot_compile.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":_pywrap_saved_model_aot_compile",
        "@absl_py//absl:app",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_saved_model_aot_compile",
    srcs = ["saved_model_aot_compile_wrapper.cc"],
    module_name = "_pywrap_saved_model_aot_compile",
    deps = [
        "//tensorflow/core/tfrt/saved_model:saved_model_aot_compile",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
