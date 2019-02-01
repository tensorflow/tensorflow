# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
    "tf_custom_op_library",
    "tf_custom_op_library_additional_deps",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)

tf_cuda_cc_test(
    name = "tensorrt_test_cc",
    size = "small",
    srcs = ["tensorrt_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        "//tensorflow/core:gpu_init",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:tensorrt",
    ]),
)

tf_custom_op_library(
    name = "python/ops/_trt_ops.so",
    srcs = [
        "ops/get_serialized_resource_op.cc",
        "ops/trt_engine_op.cc",
    ],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
)

cc_library(
    name = "trt_op_kernels",
    srcs = [
        "kernels/get_serialized_resource_op.cc",
        "kernels/trt_engine_op.cc",
    ],
    hdrs = [
        "kernels/trt_engine_op.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":test_utils",
        ":trt_allocator",
        ":trt_conversion",
        ":trt_logging",
        ":trt_plugins",
        ":trt_resources",
        ":utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/core/grappler/costs:graph_properties",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]) + tf_custom_op_library_additional_deps(),
    # TODO(laigd): fix this by merging header file in cc file.
    alwayslink = 1,  # buildozer: disable=alwayslink-with-hdrs
)

tf_cuda_cc_test(
    name = "get_serialized_resource_op_test",
    size = "small",
    srcs = ["kernels/get_serialized_resource_op_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":get_serialized_resource_op_op_lib",
        ":trt_op_kernels",
        ":trt_resources",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "trt_engine_op",
        "get_serialized_resource_op",
    ],
)

tf_cuda_library(
    name = "trt_logging",
    srcs = ["utils/trt_logger.cc"],
    hdrs = ["utils/trt_logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
)

tf_gen_op_wrapper_py(
    name = "trt_ops",
    deps = [
        ":get_serialized_resource_op_op_lib",
        ":trt_engine_op_op_lib",
        ":trt_logging",
    ],
)

tf_custom_op_py_library(
    name = "trt_ops_loader",
    srcs = ["python/ops/trt_ops.py"],
    dso = [
        "python/ops/_trt_ops.so",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
    kernels = [
        ":trt_op_kernels",
        ":trt_engine_op_op_lib",
        ":get_serialized_resource_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
    ],
)

tf_cuda_library(
    name = "trt_resources",
    srcs = [
        "utils/trt_int8_calibrator.cc",
        "utils/trt_resource_manager.cc",
        "utils/trt_resources.cc",
    ],
    hdrs = [
        "utils/trt_int8_calibrator.h",
        "utils/trt_lru_cache.h",
        "utils/trt_resource_manager.h",
        "utils/trt_resources.h",
    ],
    deps = [
        ":trt_allocator",
        ":trt_logging",
        ":utils",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
)

tf_cuda_library(
    name = "trt_allocator",
    srcs = ["utils/trt_allocator.cc"],
    hdrs = ["utils/trt_allocator.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
)

tf_cc_test(
    name = "trt_allocator_test",
    size = "small",
    srcs = ["utils/trt_allocator_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_allocator",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "trt_lru_cache_test",
    size = "small",
    srcs = ["utils/trt_lru_cache_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_resources",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Library for the node-level conversion portion of TensorRT operation creation
tf_cuda_library(
    name = "trt_conversion",
    srcs = [
        "convert/convert_graph.cc",
        "convert/convert_nodes.cc",
        "convert/trt_optimization_pass.cc",
    ],
    hdrs = [
        "convert/convert_graph.h",
        "convert/convert_nodes.h",
        "convert/trt_optimization_pass.h",
    ],
    deps = [
        ":segment",
        ":test_utils",
        ":trt_allocator",
        ":trt_plugins",
        ":trt_logging",
        ":trt_resources",
        ":utils",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]) + tf_custom_op_library_additional_deps(),
)

tf_cuda_cc_test(
    name = "convert_graph_test",
    size = "medium",
    srcs = ["convert/convert_graph_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_conversion",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
)

tf_cuda_cc_test(
    name = "convert_nodes_test",
    size = "medium",
    srcs = ["convert/convert_nodes_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_logging",
        ":trt_conversion",
        ":trt_plugins",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:tensorrt",
    ]),
)

# Library for the segmenting portion of TensorRT operation creation
cc_library(
    name = "segment",
    srcs = ["segment/segment.cc"],
    hdrs = [
        "segment/segment.h",
        "segment/union_find.h",
    ],
    deps = [
        "//tensorflow/core:graph",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@protobuf_archive//:protobuf_headers",
    ],
)

tf_cc_test(
    name = "segment_test",
    size = "small",
    srcs = ["segment/segment_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":segment",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Library for the plugin factory
tf_cuda_library(
    name = "trt_plugins",
    srcs = [
        "plugin/trt_plugin.cc",
        "plugin/trt_plugin_factory.cc",
        "plugin/trt_plugin_utils.cc",
    ],
    hdrs = [
        "plugin/trt_plugin.h",
        "plugin/trt_plugin_factory.h",
        "plugin/trt_plugin_utils.h",
    ],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:tensorrt",
    ]),
)

tf_cuda_cc_test(
    name = "trt_plugin_factory_test",
    size = "small",
    srcs = ["plugin/trt_plugin_factory_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_plugins",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:tensorrt",
    ]),
)

cc_library(
    name = "utils",
    srcs = ["convert/utils.cc"],
    hdrs = ["convert/utils.h"],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "test_utils",
    srcs = ["utils/test_utils.cc"],
    hdrs = ["utils/test_utils.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_googlesource_code_re2//:re2",
    ],
)
