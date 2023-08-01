load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")
load("//tensorflow:pytype.default.bzl", "pytype_strict_binary")

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
        "//learning/brain/tfrt/cpp_tests/gpu_inference/...",
    ],
)

pytype_strict_binary(
    name = "saved_model_load_and_run_py",
    srcs = [
        "saved_model_load_and_run.py",
    ],
    data = ["//tensorflow/core/tfrt/graph_executor:graph_execution_options.so"],  # copybara:comment
    main = "saved_model_load_and_run.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":_pywrap_saved_model",
        "@absl_py//absl:app",
        # copybara:uncomment "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_saved_model_aot_compile",
    srcs = ["saved_model_aot_compile_wrapper.cc"],
    module_name = "_pywrap_saved_model_aot_compile",
    deps = [
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model:saved_model_aot_compile",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

cc_library(
    name = "saved_model_load_and_run",
    srcs = ["saved_model_load_and_run.cc"],
    hdrs = ["saved_model_load_and_run.h"],
    deps = [
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_saved_model",
    srcs = ["saved_model_load_and_run_wrapper.cc"],
    module_name = "_pywrap_saved_model",
    deps = [
        ":saved_model_load_and_run",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/log",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
