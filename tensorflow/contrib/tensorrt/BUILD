# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_py_wrap_cc",
    "tf_cc_test",
    "tf_kernel_library",
    "tf_cuda_cc_test",
    "tf_custom_op_py_library",
    "tf_copts",
)
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
        "kernels/trt_engine_op.cc",
        "kernels/trt_engine_op.h",
        "ops/trt_engine_op.cc",
    ],
    gpu_srcs = [],
    deps = [
        ":trt_shape_function",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//tensorflow/core/kernels:ops_util_hdrs",
        "@local_config_tensorrt//:nv_infer",
    ],
)

cc_library(
    name = "trt_shape_function",
    srcs = [
        "shape_fn/trt_shfn.cc",
    ],
    hdrs = ["shape_fn/trt_shfn.h"],
    copts = tf_copts(),
    deps = [
        ":trt_logging",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@local_config_tensorrt//:nv_infer",
        "@nsync//:nsync_headers",
        "@protobuf_archive//:protobuf",
    ],
    visibility = ["//visibility:public"], #for c/c++ linking    
)

tf_kernel_library(
    name = "trt_engine_op_kernel",
    srcs = [
        "kernels/trt_engine_op.cc",
    ],
    hdrs = [
        "kernels/trt_engine_op.h",
    ],
    gpu_srcs = [
    ],
    deps = [
        ":trt_logging",
        ":trt_shape_function",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//third_party/eigen3",
        "@local_config_tensorrt//:nv_infer",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = [
        "trt_engine_op",
    ],
    deps = [
        "@local_config_tensorrt//:nv_infer",
    ],
)

cc_library(
    name = "trt_logging",
    srcs = [
        "log/trt_logger.cc",
    ],
    hdrs = [
        "log/trt_logger.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib_proto_parsing",
        "@local_config_tensorrt//:nv_infer",
    ],
)

tf_gen_op_wrapper_py(
    name = "trt_engine_op",
    deps = [
        ":trt_engine_op_op_lib",
        ":trt_shape_function",
    ],
)

tf_custom_op_py_library(
    name = "trt_engine_op_loader",
    srcs = ["python/ops/trt_engine_op.py"],
    dso = [
        ":python/ops/_trt_engine_op.so",
        "@local_config_tensorrt//:nv_infer",
    ],
    srcs_version = "PY2AND3",
    deps = [
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
    ],
)

tf_py_wrap_cc(
    name = "wrap_conversion",
    srcs = ["trt_conversion.i"],
    deps = [
        ":trt_conversion",
        "//tensorflow/core:framework_lite",
        "//util/python:python_headers",
    ],
)

cc_library(
    name = "trt_conversion",
    srcs = [
        "convert/convert_graph.cc",
        "convert/convert_nodes.cc",
        "segment/segment.cc",
    ],
    hdrs = [
        "convert/convert_graph.h",
        "convert/convert_nodes.h",
        "segment/segment.h",
        "segment/union_find.h",
    ],
    deps = [
        "@local_config_tensorrt//:nv_infer",
        "@protobuf_archive//:protobuf_headers",
        "@nsync//:nsync_headers",
        ":trt_logging",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/grappler/optimizers:constant_folding",
        "//tensorflow/core/grappler/optimizers:layout_optimizer",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/costs:graph_properties",
        #"//tensorflow/core/grappler/clusters:single_machine",
    ],
)

tf_custom_op_library(
    name = "tensorrt_ops.so",
    srcs = [
        "ops/tensorrt_ops.cc",
    ],
    deps = [
        "@local_config_tensorrt//:nv_infer",
    ],
)

# Library for the segmenting portion of TensorRT operation creation
cc_library(
    name = "segment",
    srcs = [
        "segment/segment.cc",
    ],
    hdrs = [
        "segment/segment.h",
        "segment/union_find.h",
    ],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib_proto_parsing",
        "//third_party/eigen3",
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
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Library for the node-level conversion portion of TensorRT operation creation

filegroup(
    name = "cppfiles",
    srcs = glob(["**/*.cc"]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "headers",
    srcs = glob(["**/*.h"]),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
