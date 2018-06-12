# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
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
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)

tf_cuda_cc_test(
    name = "tensorrt_test_cc",
    size = "small",
    srcs = ["tensorrt_test.cc"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:nv_infer",
    ]),
)

tf_custom_op_library(
    name = "python/ops/_trt_engine_op.so",
    srcs = [
        "ops/trt_engine_op.cc",
    ],
    deps = [
        ":trt_shape_function",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]),
)

tf_cuda_library(
    name = "trt_shape_function",
    srcs = ["shape_fn/trt_shfn.cc"],
    hdrs = ["shape_fn/trt_shfn.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":trt_logging",
        ":trt_plugins",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]) + tf_custom_op_library_additional_deps(),
)

cc_library(
    name = "trt_engine_op_kernel",
    srcs = [
        "kernels/trt_engine_op.cc",
    ],
    hdrs = [
        "kernels/trt_engine_op.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":trt_logging",
        ":trt_plugins",
        ":trt_resources",
        ":trt_conversion",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/core/grappler/costs:graph_properties",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]) + tf_custom_op_library_additional_deps(),
    # TODO(laigd)
    alwayslink = 1,  # buildozer: disable=alwayslink-with-hdrs
)

tf_gen_op_libs(
    op_lib_names = [
        "trt_engine_op",
    ],
)

tf_cuda_library(
    name = "trt_logging",
    srcs = ["log/trt_logger.cc"],
    hdrs = ["log/trt_logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]),
)

tf_gen_op_wrapper_py(
    name = "trt_engine_op",
    gen_locally = True,
    deps = [
        ":trt_engine_op_op_lib",
        ":trt_logging",
        ":trt_shape_function",
    ],
)

tf_custom_op_py_library(
    name = "trt_engine_op_loader",
    srcs = ["python/ops/trt_engine_op.py"],
    dso = [
        ":python/ops/_trt_engine_op.so",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]),
    kernels = [
        ":trt_engine_op_kernel",
        ":trt_engine_op_op_lib",
        ":trt_shape_function",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:resources",
    ],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":trt_convert_py",
        ":trt_ops_py",
        "//tensorflow/python:errors",
    ],
)

py_library(
    name = "trt_ops_py",
    srcs_version = "PY2AND3",
    deps = [
        ":trt_engine_op",
        ":trt_engine_op_loader",
    ],
)

py_library(
    name = "trt_convert_py",
    srcs = ["python/trt_convert.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":wrap_conversion",
        "//tensorflow/python:tf_optimizer",
    ],
)

tf_py_wrap_cc(
    name = "wrap_conversion",
    srcs = ["trt_conversion.i"],
    copts = tf_copts(),
    deps = [
        ":trt_conversion",
        ":trt_engine_op_kernel",
        "//third_party/python_runtime:headers",
    ],
)

tf_cuda_library(
    name = "trt_resources",
    srcs = [
        "resources/trt_allocator.cc",
        "resources/trt_int8_calibrator.cc",
        "resources/trt_resource_manager.cc",
    ],
    hdrs = [
        "resources/trt_allocator.h",
        "resources/trt_int8_calibrator.h",
        "resources/trt_resource_manager.h",
        "resources/trt_resources.h",
    ],
    deps = [
        ":trt_logging",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]),
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
        ":trt_plugins",
        ":trt_logging",
        ":trt_resources",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
    ] + if_tensorrt([
        "@local_config_tensorrt//:nv_infer",
    ]) + tf_custom_op_library_additional_deps(),
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
        "@protobuf_archive//:protobuf_headers",
    ],
)

tf_cc_test(
    name = "segment_test",
    size = "small",
    srcs = ["segment/segment_test.cc"],
    deps = [
        ":segment",
        "//tensorflow/c:c_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
        "@local_config_tensorrt//:nv_infer",
    ]),
)

tf_cuda_cc_test(
    name = "trt_plugin_factory_test",
    size = "small",
    srcs = ["plugin/trt_plugin_factory_test.cc"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":trt_plugins",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_tensorrt//:nv_infer",
    ]),
)

py_test(
    name = "tf_trt_integration_test",
    srcs = ["test/tf_trt_integration_test.py"],
    main = "test/tf_trt_integration_test.py",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":init_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)
