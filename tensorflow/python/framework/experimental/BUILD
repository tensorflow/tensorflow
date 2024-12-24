# Experimental Unified APIs for Eager and Graph modes.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")
load(
    "//tensorflow/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_unified_api",
    srcs = ["unified_api.cc"],
    enable_stub_generation = True,
    features = ["-layering_check"],
    pytype_srcs = [
        "_unified_api.pyi",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/python:unified_api_pywrap_required_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_tape",
    srcs = ["tape.cc"],
    features = ["-layering_check"],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/python:unified_api_pywrap_required_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//tensorflow/c/experimental/gradients/tape:tape_context",
            "//tensorflow/c/experimental/gradients:math_grad",
            "//tensorflow/c/experimental/gradients:nn_grad",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_math_ops",
    srcs = ["math_ops.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_math_ops.pyi",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/python:unified_api_pywrap_required_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//tensorflow/c/experimental/ops:math_ops",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_nn_ops",
    srcs = ["nn_ops.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_nn_ops.pyi",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/python:unified_api_pywrap_required_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//tensorflow/c/experimental/ops:nn_ops",
        ],
    ),
)

py_strict_library(
    name = "gradient_registry",
    srcs = ["gradient_registry.py"],
    srcs_version = "PY3",
    deps = [":_tape"],
)

py_strict_library(
    name = "math_ops",
    srcs = ["math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":_math_ops",
        ":context_stack",
    ],
)

py_strict_library(
    name = "nn_ops",
    srcs = ["nn_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":_nn_ops",
        ":context_stack",
    ],
)

py_strict_library(
    name = "tape",
    srcs = ["tape.py"],
    srcs_version = "PY3",
    deps = [
        ":_tape",
        ":context_stack",
        ":gradient_registry",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "def_function",
    srcs = ["def_function.py"],
    srcs_version = "PY3",
    deps = [
        ":_unified_api",
        ":context_stack",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "thread_local_stack",
    srcs = ["thread_local_stack.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "context_stack",
    srcs = ["context_stack.py"],
    srcs_version = "PY3",
    deps = [":thread_local_stack"],
)

cuda_py_strict_test(
    name = "unified_api_test",
    size = "small",
    srcs = ["unified_api_test.py"],
    tags = [
        # Note(srbs): These python bindings are not
        # exported as part of the pip package yet so
        # this test is disabled.
        "no_pip",
        "no_windows",  # b/168218876
    ],
    deps = [
        ":_unified_api",
        ":context_stack",
        ":def_function",
        ":math_ops",
        ":nn_ops",
        ":tape",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_benchmark_test(
    name = "graph_building_test",
    size = "small",
    srcs = ["graph_building_test.py"],
    deps = [
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/platform:client_testlib",
    ],
)
