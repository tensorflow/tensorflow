# This package loads pybind11 dependencies to ensure compatibility with Python API.
# Files affected: graph_execution_options
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
        "//tensorflow/core/tfrt/graph_executor/...",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    enable_stub_generation = True,
    module_name = "_pywrap_graph_execution_options",
    pytype_srcs = [
        "_pywrap_graph_execution_options.pyi",
    ],
    deps = [
        "//tensorflow/core/tfrt/graph_executor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model",
        "//tensorflow/core/tfrt/saved_model:saved_model_aot_compile",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
