# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
    "tf_custom_op_library_additional_deps",
    "tf_gen_op_wrapper_py",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tensorrt_stub",
    srcs = if_tensorrt([
        "stub/nvinfer_stub.cc",
        "stub/nvinfer_plugin_stub.cc",
    ]),
    textual_hdrs = glob(["stub/*.inc"]),
    deps = if_tensorrt([
        "@local_config_tensorrt//:tensorrt_headers",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/platform:dso_loader",
    ]),
)

alias(
    name = "tensorrt_lib",
    actual = select({
        "@local_config_tensorrt//:use_static_tensorrt": "@local_config_tensorrt//:tensorrt",
        "//conditions:default": ":tensorrt_stub",
    }),
    visibility = ["//visibility:private"],
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
        ":trt_logging",
        ":utils",
        "//tensorflow/core/common_runtime/gpu:gpu_init",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        ":tensorrt_lib",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cc_library(
    name = "common_utils",
    srcs = ["common/utils.cc"],
    hdrs = [
        "common/datavec.h",
        "common/utils.h",
    ],
    copts = tf_copts(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
    ] + if_tensorrt([":tensorrt_lib"]),
)

cc_library(
    name = "trt_op_kernels",
    srcs = [
        "kernels/get_calibration_data_op.cc",
        "kernels/trt_engine_op.cc",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":trt_allocator",
        ":trt_conversion",
        ":trt_engine_utils",
        ":trt_logging",
        ":trt_plugins",
        ":trt_resources",
        ":utils",
        ":common_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:stream_executor",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/core/common_runtime:core_cpu_lib_no_ops",
        "//tensorflow/core/grappler/costs:graph_properties",
    ] + if_tensorrt([
        ":tensorrt_lib",
        "@local_config_cuda//cuda:cuda_headers",
    ]) + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

cc_library(
    name = "trt_engine_resource_op_kernels",
    srcs = ["kernels/trt_engine_resource_ops.cc"],
    copts = tf_copts(),
    visibility = ["//tensorflow/core:__subpackages__"],
    deps = [
        ":trt_allocator",
        ":trt_engine_instance_proto_cc",
        ":trt_logging",
        ":trt_plugins",
        ":trt_resources",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:gpu_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]) + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "trt_engine_resource_ops_test",
    size = "small",
    srcs = ["kernels/trt_engine_resource_ops_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":common_utils",
        ":trt_engine_instance_proto_cc",
        ":trt_engine_resource_op_kernels",
        ":trt_engine_resource_ops_op_lib",
        ":trt_logging",
        ":trt_resources",
        ":utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:resource_variable_ops",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
    ],
)

tf_cuda_cc_test(
    name = "trt_engine_op_test",
    size = "small",
    srcs = ["kernels/trt_engine_op_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_op_kernels",
        ":trt_op_libs",
        ":trt_resources",
        ":trt_conversion",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//third_party/eigen3",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/framework:fake_input",
    ] + if_tensorrt([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

tf_gen_op_libs(
    op_lib_names = [
        "trt_engine_op",
        "get_calibration_data_op",
        "trt_engine_resource_ops",
    ],
)

cc_library(
    name = "trt_op_libs",
    deps = [
        ":get_calibration_data_op_op_lib",
        ":trt_engine_op_op_lib",
        ":trt_engine_utils",
    ],
)

tf_cuda_library(
    name = "trt_engine_utils",
    srcs = [
        "utils/trt_engine_utils.cc",
        "utils/trt_shape_optimization_profiles.cc",
    ],
    hdrs = [
        "utils/trt_engine_utils.h",
        "utils/trt_execution_context.h",
        "utils/trt_shape_optimization_profiles.h",
    ],
    deps = [
        ":common_utils",
        ":trt_logging",
        ":utils",
        ":trt_allocator",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "//tensorflow/core:stream_executor_headers_lib",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_library(
    name = "trt_logging",
    srcs = ["utils/trt_logger.cc"],
    hdrs = ["utils/trt_logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_utils",
        ":logger_registry",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_gen_op_wrapper_py(
    name = "trt_ops",
    deps = [
        ":trt_engine_resource_ops_op_lib",
        ":trt_op_libs",
    ],
)

tf_custom_op_py_library(
    name = "trt_ops_loader",
    srcs_version = "PY3",
    deps = [
        ":_pywrap_py_utils",
        ":trt_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:resources",
    ],
)

tf_cuda_library(
    name = "trt_resources",
    srcs = [
        "utils/trt_int8_calibrator.cc",
        "utils/trt_lru_cache.cc",
    ],
    hdrs = [
        "utils/trt_int8_calibrator.h",
        "utils/trt_lru_cache.h",
        "utils/trt_shape_optimization_profiles.h",
        "utils/trt_tensor_proxy.h",
    ],
    deps = [
        ":common_utils",
        ":trt_allocator",
        ":trt_engine_utils",
        ":trt_logging",
        ":utils",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core:graph",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_library(
    name = "trt_allocator",
    srcs = ["utils/trt_allocator.cc"],
    hdrs = ["utils/trt_allocator.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
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

tf_cuda_cc_test(
    name = "trt_shape_optimization_profiles_test",
    size = "small",
    srcs = ["utils/trt_shape_optimization_profiles_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_resources",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_library(
    name = "logger_registry",
    srcs = ["convert/logger_registry.cc"],
    hdrs = [
        "convert/logger_registry.h",
    ],
    copts = tf_copts(),
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
    ] + if_tensorrt([":tensorrt_lib"]),
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
        ":common_utils",
        ":logger_registry",
        ":segment",
        ":trt_allocator",
        ":trt_plugins",
        ":trt_logging",
        ":trt_resources",
        ":utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:graph",
        "//tensorflow/core/kernels/linalg:einsum_op",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/stream_executor/lib",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ] + if_tensorrt([":tensorrt_lib"]) + tf_custom_op_library_additional_deps(),
    alwayslink = 1,
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
        ":trt_op_kernels",
        ":trt_op_libs",
        ":trt_conversion",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
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
    ] + if_tensorrt([":tensorrt_lib"]),
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
        ":trt_engine_utils",
        ":utils",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([
        ":tensorrt_lib",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

# Library for the segmenting portion of TensorRT operation creation
cc_library(
    name = "union_find",
    srcs = ["segment/union_find.cc"],
    hdrs = [
        "segment/union_find.h",
    ],
    copts = tf_copts(),
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "segment",
    srcs = ["segment/segment.cc"],
    hdrs = [
        "segment/segment.h",
    ],
    copts = tf_copts(),
    deps = [
        ":common_utils",
        ":union_find",
        ":utils",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/grappler/costs:graph_properties",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

tf_cuda_cc_test(
    name = "segment_test",
    size = "small",
    srcs = ["segment/segment_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
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
    ],
)

tf_cuda_library(
    name = "trt_plugins",
    srcs = ["plugin/trt_plugin.cc"],
    hdrs = ["plugin/trt_plugin.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_tensorrt([":tensorrt_lib"]),
)

cc_library(
    name = "utils",
    srcs = ["convert/utils.cc"],
    hdrs = [
        "common/utils.h",
        "convert/utils.h",
        "utils/trt_tensor_proxy.h",
    ],
    copts = tf_copts(),
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:lib",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_proto_library(
    name = "trt_engine_instance_proto",
    srcs = ["utils/trt_engine_instance.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

cc_library(
    name = "py_utils",
    srcs = ["utils/py_utils.cc"],
    hdrs = ["utils/py_utils.h"],
    copts = tf_copts(),
    local_defines = select({
        "@local_config_tensorrt//:use_static_tensorrt": ["TF_USE_TENSORRT_STATIC=1"],
        "//conditions:default": [],
    }),
    deps = if_tensorrt([
        ":common_utils",
        ":tensorrt_lib",
        "//tensorflow/stream_executor/platform:dso_loader",
    ]),
)

pybind_extension(
    name = "_pywrap_py_utils",
    srcs = ["utils/py_utils_wrapper.cc"],
    link_in_framework = True,
    module_name = "_pywrap_py_utils",
    deps = [
        ":common_utils",
        ":py_utils",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:status",
        "@pybind11",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "trt_engine_instance_proto_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     deps = [":trt_engine_instance_proto"],
# )
# copybara:uncomment_end
