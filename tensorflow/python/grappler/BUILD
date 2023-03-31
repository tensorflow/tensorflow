load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "get_compatible_with_cloud", "tf_py_test", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config.bzl", "tf_protos_grappler")
load("//tensorflow:tensorflow.bzl", "if_not_windows")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

# TODO(gunan): Investigate making this action hermetic so we do not need
# to run it locally.
cc_library(
    name = "cost_analyzer_lib",
    srcs = ["cost_analyzer.cc"],
    hdrs = ["cost_analyzer.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:analytical_cost_estimator",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/grappler/costs:measuring_cost_estimator",
        "//tensorflow/core/grappler/costs:utils",
    ] + tf_protos_grappler(),
    alwayslink = 1,
)

# Necessary for the pywrap inclusion below. Combining targets does not work
# properly.
tf_pybind_cc_library_wrapper(
    name = "cost_analyzer_headers",
    deps = [
        ":cost_analyzer_lib",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_cost_analyzer",
    srcs = ["cost_analyzer_wrapper.cc"],
    hdrs = [
        "cost_analyzer.h",
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/public:session.h",
        "//tensorflow/core/public:session_options.h",
    ],
    deps = [
        ":cost_analyzer_headers",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

cc_library(
    name = "model_analyzer_lib",
    srcs = ["model_analyzer.cc"],
    hdrs = ["model_analyzer.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/costs:graph_properties",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_model_analyzer",
    srcs = ["model_analyzer_wrapper.cc"],
    hdrs = [
        "model_analyzer.h",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_library(
    name = "tf_item",
    srcs = [
        "item.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_item",
        "//tensorflow/core/grappler/costs:op_performance_data_py",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_item",
    srcs = ["item_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/grappler/utils:pywrap_required_hdrs",
    ],
    deps = [
        "@pybind11",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/python/lib/core:pybind11_status",
    ] + if_not_windows(["//tensorflow/core/grappler/costs:graph_properties"]),  # b/148556093,
)

tf_py_test(
    name = "item_test",
    size = "small",
    srcs = [
        "item_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)

tf_py_test(
    name = "datasets_test",
    size = "medium",
    srcs = [
        "datasets_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)

py_library(
    name = "tf_cluster",
    srcs = [
        "cluster.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_cluster",
        "//tensorflow/core/grappler/costs:op_performance_data_py",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_cluster",
    srcs = ["cluster_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/grappler/utils:pywrap_required_hdrs",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cuda_py_test(
    name = "cluster_test",
    size = "small",
    srcs = [
        "cluster_test.py",
    ],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
        "no_windows",  # b/173520599
        "notap",  # TODO(b/135924227): Re-enable after fixing flakiness.
    ],
    # This test will not run on XLA because it primarily tests the TF Classic flow.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_cluster",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)

py_library(
    name = "tf_optimizer",
    srcs = [
        "tf_optimizer.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_optimizer",
        ":tf_cluster",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_optimizer",
    srcs = ["tf_optimizer_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/grappler/optimizers:pywrap_required_hdrs",
        "//tensorflow/core/grappler/verifiers:pywrap_required_hdrs",
    ],
    # This fails Windows builds. Please check b/266870200 for details.
    #    dynamic_deps = ["//tensorflow/python:_pywrap_tensorflow_internal.so"] + select({
    #        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
    #        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
    #        "//tensorflow:windows": [],
    #    }),
    #    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

tf_py_test(
    name = "tf_optimizer_test",
    size = "small",
    srcs = [
        "tf_optimizer_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "memory_optimizer_test",
    size = "medium",
    srcs = [
        "memory_optimizer_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
    ],
    deps = [
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:random_seed",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "constant_folding_test",
    size = "medium",
    srcs = [
        "constant_folding_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "arithmetic_optimizer_test",
    size = "small",
    srcs = [
        "arithmetic_optimizer_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

# TODO(b/131764887) Remove once LayoutOptimizer is swapped out with GenericLayoutOptimizer.
#
# cuda_py_test(
#     name = "layout_optimizer_test",
#     size = "medium",
#     srcs = [
#         "layout_optimizer_test.py",
#     ],
#     deps = [
#         "//tensorflow/python:client_testlib",
#         "//tensorflow/python/framework:for_generated_wrappers",
#         "//tensorflow/python:array_ops",
#         "//tensorflow/python:functional_ops",
#         "//tensorflow/python:math_ops",
#         "//tensorflow/python:nn",
#         "//tensorflow/python:ops",
#         "//tensorflow/python:random_ops",
#         "//tensorflow/python:state_ops",
#         ":tf_cluster",
#         ":tf_optimizer",
#         "//tensorflow/python:training",
#         "//third_party/py/numpy",
#         "//tensorflow/core:protos_all_py",
#         "//tensorflow/python/framework:constant_op",
#         "//tensorflow/python/framework:dtypes",
#     ],
#     shard_count = 10,
#     tags = [
#         "grappler",
#     ],
#     # This test will not run on XLA because it primarily tests the TF Classic flow.
#     xla_enable_strict_auto_jit = False,
# )

py_library(
    name = "cost_analyzer",
    srcs = [
        "cost_analyzer.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_cost_analyzer",
        ":tf_cluster",
        ":tf_item",
    ],
)

py_binary(
    name = "cost_analyzer_tool",
    srcs = [
        "cost_analyzer_tool.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cost_analyzer",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)

tf_py_test(
    name = "cost_analyzer_test",
    size = "small",
    srcs = ["cost_analyzer_test.py"],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_cuda_on_cpu_tap",
        "no_mac",
        "no_pip",
        "no_windows",  # TODO(b/151942037)
    ],
    deps = [
        ":cost_analyzer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "model_analyzer",
    srcs = [
        "model_analyzer.py",
    ],
    srcs_version = "PY3",
    deps = [":_pywrap_model_analyzer"],
)

tf_py_test(
    name = "model_analyzer_test",
    size = "small",
    srcs = ["model_analyzer_test.py"],
    tags = [
        "grappler",
        "no_pip",
    ],
    deps = [
        ":model_analyzer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "auto_mixed_precision_test",
    size = "medium",
    srcs = [
        "auto_mixed_precision_test.py",
    ],
    python_version = "PY3",
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "remapper_test",
    size = "medium",
    srcs = [
        "remapper_test.py",
    ],
    python_version = "PY3",
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_graph_analyzer",
    srcs = ["graph_analyzer_tool_wrapper.cc"],
    deps = [
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
        "@pybind11",
    ],
)

py_binary(
    name = "graph_analyzer",
    srcs = [
        "graph_analyzer.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":_pywrap_graph_analyzer",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)
