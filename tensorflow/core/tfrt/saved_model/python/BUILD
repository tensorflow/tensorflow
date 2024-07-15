load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")

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

# copybara:uncomment_begin(Test working locally but failing Koroko CPU submits)
# tf_py_strict_test(
#     name = "saved_model_load_and_run_test_py",
#     srcs = ["saved_model_load_and_run_test.py"],
#     main = "saved_model_load_and_run_test.py",
#     python_version = "PY3",
#     srcs_version = "PY3",
#     tags = ["requires-gpu-nvidia"],
#     deps = [
#         ":_pywrap_saved_model",
#         "//tensorflow/python/eager:context",
#         "//tensorflow/python/framework:constant_op",
#         "//tensorflow/python/framework:ops",
#         "//tensorflow/python/platform:client_testlib",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "saved_model_load_and_run",
    srcs = ["saved_model_load_and_run.cc"],
    hdrs = ["saved_model_load_and_run.h"],
    deps = [
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime:work_queue_interface",
        "//tensorflow/core/tfrt/saved_model",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:refcount",
        "@tf_runtime//:hostcontext",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_saved_model",
    srcs = ["saved_model_load_and_run_wrapper.cc"],
    enable_stub_generation = True,
    module_name = "_pywrap_saved_model",
    pytype_srcs = [
        "_pywrap_saved_model.pyi",
    ],
    deps = [
        ":saved_model_load_and_run",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/saved_model:saved_model_cpu",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
