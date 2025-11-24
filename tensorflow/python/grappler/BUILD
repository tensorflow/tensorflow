load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "get_compatible_with_portable", "tf_py_strict_test", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config.bzl", "tf_protos_grappler")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")

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
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:analytical_cost_estimator",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/grappler/costs:measuring_cost_estimator",
        "//tensorflow/core/grappler/costs:utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_cost_analyzer.pyi",
    ],
    starlark_only = True,
    deps = [
        ":cost_analyzer_headers",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:threadpool_options",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
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
        "@com_google_absl//absl/status",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_model_analyzer",
    srcs = ["model_analyzer_wrapper.cc"],
    hdrs = [
        "model_analyzer.h",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_model_analyzer.pyi",
    ],
    starlark_only = True,
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:platform_port",
        "@pybind11",
    ] + if_pywrap(["//tensorflow/python/grappler:model_analyzer_lib"]),
)

py_strict_library(
    name = "tf_item",
    srcs = ["item.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_item",
        "//tensorflow/core:protos_all_py",
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_item.pyi",
    ],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@pybind11",
    ] + if_not_windows(["//tensorflow/core/grappler/costs:graph_properties"]),  # b/148556093,
)

tf_py_strict_test(
    name = "item_test",
    size = "small",
    srcs = ["item_test.py"],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "datasets_test",
    size = "medium",
    srcs = ["datasets_test.py"],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tf_cluster",
    srcs = ["cluster.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_cluster",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/grappler/costs:op_performance_data_py",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_cluster",
    srcs = ["cluster_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
    ] + if_pywrap(
        if_false = [
            "//tensorflow/core/grappler:pywrap_required_hdrs",
            "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
            "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
            "//tensorflow/core/grappler/utils:pywrap_required_hdrs",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_cluster.pyi",
    ],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//tensorflow/core/grappler/costs:measuring_cost_estimator",
            "//tensorflow/core/grappler/clusters:single_machine",
        ],
    ),
)

cuda_py_strict_test(
    name = "cluster_test",
    size = "small",
    srcs = ["cluster_test.py"],
    shard_count = 5,
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
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "tf_optimizer",
    srcs = ["tf_optimizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_optimizer",
        ":tf_cluster",
        "//tensorflow/core:protos_all_py",
        "@absl_py//absl/logging",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_optimizer",
    srcs = ["tf_optimizer_wrapper.cc"],
    hdrs = if_pywrap(
        if_false = [
            "//tensorflow/cc:pywrap_required_hdrs",
            "//tensorflow/core/grappler:pywrap_required_hdrs",
            "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
            "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
            "//tensorflow/core/grappler/optimizers:pywrap_required_hdrs",
            "//tensorflow/core/grappler/verifiers:pywrap_required_hdrs",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_optimizer.pyi",
    ],
    # This fails Windows builds. Please check b/266870200 for details.
    #    dynamic_deps = ["//tensorflow/python:_pywrap_tensorflow_internal.so"] + select({
    #        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
    #        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
    #        "//tensorflow:windows": [],
    #    }),
    #    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "@pybind11",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        if_true = [
            "//tensorflow/core/grappler/clusters:cluster",
            "//tensorflow/core/grappler/clusters:utils",
            "//tensorflow/core/grappler:grappler_item_builder",
            "//tensorflow/core/grappler/optimizers:meta_optimizer",
            "//tensorflow/core/grappler/optimizers:graph_optimizer",
            "//tensorflow/core/grappler/verifiers:graph_verifier",
        ],
    ),
)

tf_py_strict_test(
    name = "tf_optimizer_test",
    size = "small",
    srcs = ["tf_optimizer_test.py"],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "memory_optimizer_test",
    size = "medium",
    srcs = ["memory_optimizer_test.py"],
    tags = [
        "grappler",
    ],
    deps = [
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:training_lib",
    ],
)

cuda_py_strict_test(
    name = "constant_folding_test",
    size = "medium",
    srcs = ["constant_folding_test.py"],
    tags = [
        "grappler",
    ],
    deps = [
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "arithmetic_optimizer_test",
    size = "small",
    srcs = ["arithmetic_optimizer_test.py"],
    tags = [
        "grappler",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
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
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/framework:for_generated_wrappers",
#         "//tensorflow/python/ops:array_ops",
#         "//tensorflow/python/ops:functional_ops",
#         "//tensorflow/python/ops:math_ops",
#         "//tensorflow/python:nn",
#         "//tensorflow/python/user_ops:ops",
#         "//tensorflow/python/ops:random_ops",
#         "//tensorflow/python/ops:state_ops",
#         ":tf_cluster",
#         ":tf_optimizer",
#         "//tensorflow/python/training:training",
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

py_strict_library(
    name = "cost_analyzer",
    srcs = ["cost_analyzer.py"],
    deps = [
        ":_pywrap_cost_analyzer",
        ":tf_cluster",
        ":tf_item",
    ],
)

py_strict_binary(
    name = "cost_analyzer_tool",
    srcs = ["cost_analyzer_tool.py"],
    deps = [
        ":cost_analyzer",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/training:saver",
        "@absl_py//absl:app",
    ],
)

tf_py_strict_test(
    name = "cost_analyzer_test",
    size = "small",
    srcs = ["cost_analyzer_test.py"],
    tags = [
        "grappler",
        "no_cuda_on_cpu_tap",
        "no_mac",
        "no_pip",
        "no_windows",  # TODO(b/151942037)
    ],
    deps = [
        ":cost_analyzer",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:adam",
    ],
)

py_strict_library(
    name = "model_analyzer",
    srcs = [
        "model_analyzer.py",
    ],
    deps = [":_pywrap_model_analyzer"],
)

tf_py_strict_test(
    name = "model_analyzer_test",
    size = "small",
    srcs = ["model_analyzer_test.py"],
    tags = [
        "grappler",
        "no_pip",
    ],
    deps = [
        ":model_analyzer",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "auto_mixed_precision_test",
    size = "medium",
    srcs = [
        "auto_mixed_precision_test.py",
    ],
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:nn_impl",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/training:adam",
        "//tensorflow/python/training:gradient_descent",
        "//tensorflow/python/util:_pywrap_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "remapper_test",
    size = "medium",
    srcs = ["remapper_test.py"],
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/util:_pywrap_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_graph_analyzer",
    srcs = ["graph_analyzer_tool_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_graph_analyzer.pyi",
    ],
    deps = [
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
        "@pybind11",
    ],
)

py_strict_binary(
    name = "graph_analyzer",
    srcs = ["graph_analyzer.py"],
    deps = [
        ":_pywrap_graph_analyzer",
        "@absl_py//absl:app",
    ],
)
