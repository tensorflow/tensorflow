load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = ["//tensorflow/python/profiler:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "flops_registry",
    srcs = ["flops_registry.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:graph_util",
    ],
)

py_library(
    name = "model_analyzer_testlib",
    srcs = ["model_analyzer_testlib.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:rnn",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "print_model_analysis_test",
    srcs = ["print_model_analysis_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:variable_scope",
    ],
)

cuda_py_test(
    name = "run_metadata_test",
    srcs = ["run_metadata_test.py"],
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/138442728
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer_testlib",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/profiler:model_analyzer",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_traceme",
    srcs = ["traceme_wrapper.cc"],
    module_name = "_pywrap_traceme",
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//tensorflow/python/profiler:__subpackages__",
    ],
    deps = [
        ":traceme_wrapper",
        "@pybind11",
    ],
)

cc_library(
    name = "traceme_wrapper",
    hdrs = ["traceme_wrapper.h"],
    features = ["-layering_check"],
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_profiler",
    srcs = ["profiler_wrapper.cc"],
    features = ["-layering_check"],
    module_name = "_pywrap_profiler",
    visibility = [
        "//tensorflow/python/eager:__pkg__",
        "//tensorflow/python/profiler:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/convert:xplane_to_profile_response",
        "//tensorflow/core/profiler/convert:xplane_to_trace_events",
        "//tensorflow/core/profiler/lib:profiler_session_headers",
        "//tensorflow/core/profiler/rpc:profiler_server",
        "//tensorflow/core/profiler/rpc/client:capture_profile",
        "//tensorflow/core/profiler/rpc/client:save_profile",
        "//tensorflow/python:pybind11_status",
        "@com_google_absl//absl/memory",
        "@pybind11",
    ],
)

cc_library(
    name = "python_hooks",
    srcs = ["python_hooks.cc"],
    hdrs = ["python_hooks.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
    alwayslink = True,
)
