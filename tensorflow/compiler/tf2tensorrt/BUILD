# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with tensorflow
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_google",
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

# Platform specific build config
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
    licenses = ["notice"],
)

# Whether or not to use the EfficientNMSPlugin from NVIDIA OSS archive.
config_setting(
    name = "use_efficient_nms_plugin",
    define_values = {"use_efficient_nms_plugin": "1"},
)

# Config setting to conditionally link GPU targets.
alias(
    name = "efficient_nms_plugin_wrapper",
    actual = if_google(
        "//third_party/tensorrt/plugin:nvinfer_plugin_nms",
        "@tensorrt_oss_archive//:nvinfer_plugin_nms",
    ),
)

cc_library(
    name = "efficient_nms_plugin",
    deps = if_tensorrt([
        ":efficient_nms_plugin_wrapper",
    ]),
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
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
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
    srcs = [
        "tensorrt_test.cc",
    ],
    extra_copts = select({
        ":use_efficient_nms_plugin": ["-DTF_TRT_USE_EFFICIENT_NMS_PLUGIN=1"],
        "//conditions:default": [],
    }),
    tags = [
        "no_cuda_on_cpu_tap",
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
    ]) + select({
        ":use_efficient_nms_plugin": [":efficient_nms_plugin"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "trt_convert_api",
    srcs = ["trt_convert_api.cc"],
    hdrs = [
        "trt_convert_api.h",
    ],
    copts = tf_copts(),
    deps = [
        ":trt_parameters",
        ":trt_resources",
        "//tensorflow/cc/tools:freeze_saved_model",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/strings",
    ] + if_tensorrt([":tensorrt_lib"]),
)

filegroup(
    name = "headers",
    srcs = [
        "trt_convert_api.h",
    ],
)

tf_cuda_cc_test(
    name = "trt_convert_api_test",
    size = "small",
    srcs = ["trt_convert_api_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":common_utils",
        ":testutils",
        ":trt_conversion",
        ":trt_convert_api",
        ":trt_logging",
        ":trt_op_kernels",
        ":trt_resources",
        ":utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:function_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:assign_op",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:partitioned_function_ops",
        "//tensorflow/core/kernels:resource_variable_ops",
    ],
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
        "//tensorflow/core/profiler/lib:annotated_traceme",
    ] + if_tensorrt([":tensorrt_lib"]),
)

cc_library(
    name = "testutils",
    testonly = 1,
    srcs = ["utils/trt_testutils.cc"],
    hdrs = [
        "utils/trt_testutils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":trt_conversion",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core/framework:tensor_testutil",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_cc_test(
    name = "testutils_test",
    size = "small",
    srcs = ["utils/trt_testutils_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":testutils",
        "//tensorflow/core:test_main",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:protobuf",
    ] + if_tensorrt([
        ":tensorrt_lib",
    ]),
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
        ":testutils",
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
        "@com_google_absl//absl/strings",
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
        ":testutils",
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
        ":trt_parameters",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:annotated_traceme",
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
        ":utils",
        "@com_google_absl//absl/strings",
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
    name = "trt_parameters",
    srcs = ["convert/trt_parameters.cc"],
    hdrs = [
        "convert/trt_parameters.h",
    ],
    copts = tf_copts(),
    deps = [
        ":utils",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
    ] + if_tensorrt([":tensorrt_lib"]),
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

tf_cuda_cc_test(
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

tf_cuda_cc_test(
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

tf_cuda_library(
    name = "trt_weights",
    srcs = ["convert/weights.cc"],
    hdrs = [
        "convert/weights.h",
    ],
    copts = tf_copts(),
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_library(
    name = "op_converter",
    srcs = [],
    hdrs = [
        "convert/op_converter.h",
    ],
    deps = [
        ":trt_parameters",
        ":trt_weights",
    ] + if_tensorrt([":tensorrt_lib"]),
)

# This rule contains static variables for the converter registry. Do not depend
# on it directly; use :op_converter_registry, and link against
# libtensorflow_framework.so for the registry symbols. The library
# libtensorflow_framework.so depends on this target so that users can
# register custom op converters without the need to incorporate Tensorflow into
# their build system.
tf_cuda_library(
    name = "op_converter_registry_impl",
    srcs = ["convert/op_converter_registry.cc"],
    hdrs = [
        "convert/op_converter_registry.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":utils",
        ":op_converter",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
    ] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_library(
    name = "op_converter_registry",
    hdrs = [
        "convert/op_converter_registry.h",
    ],
    copts = tf_copts(),
    deps = [
        ":utils",
        ":op_converter",
        "//tensorflow/core:lib",
    ] + if_static([":op_converter_registry_impl"]),
)

tf_cuda_cc_test(
    name = "op_converter_registry_test",
    size = "small",
    srcs = ["convert/op_converter_registry_test.cc"],
    tags = [
        "no_windows",
        "nomac",
    ],
    deps = [
        ":op_converter_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_library(
    name = "algorithm_selector",
    srcs = [
        "convert/algorithm_selector.cc",
    ],
    hdrs = [
        "convert/algorithm_selector.h",
    ],
    deps = [":common_utils"] + if_tensorrt([":tensorrt_lib"]),
)

tf_cuda_cc_test(
    name = "algorithm_selector_test",
    srcs = [
        "convert/algorithm_selector_test.cc",
    ],
    deps = [
        ":algorithm_selector",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_tensorrt([":tensorrt_lib"]),
)

# Library for the node-level conversion portion of TensorRT operation creation
tf_cuda_library(
    name = "trt_conversion",
    srcs = [
        "convert/convert_graph.cc",
        "convert/convert_nodes.cc",
        "convert/ops/binary_ops.cc",
        "convert/ops/data_format_vec_permute.cc",
        "convert/ops/einsum.cc",
        "convert/ops/fill_ops.cc",
        "convert/ops/like_ops.cc",
        "convert/ops/log_softmax.cc",
        "convert/ops/quantization_ops.cc",
        "convert/ops/slice_ops.cc",
        "convert/ops/softmax.cc",
        "convert/ops/tile.cc",
        "convert/ops/unary_ops.cc",
        "convert/ops/variable_ops.cc",
        "convert/timing_cache.cc",
        "convert/trt_optimization_pass.cc",
    ],
    hdrs = [
        "convert/convert_graph.h",
        "convert/convert_nodes.h",
        "convert/ops/layer_utils.h",
        "convert/ops/quantization_ops.h",
        "convert/ops/slice_ops.h",
        "convert/timing_cache.h",
        "convert/trt_optimization_pass.h",
    ],
    copts = tf_copts() + select({
        ":use_efficient_nms_plugin": ["-DTF_TRT_USE_EFFICIENT_NMS_PLUGIN=1"],
        "//conditions:default": [],
    }),
    deps = [
        ":algorithm_selector",
        ":common_utils",
        ":logger_registry",
        ":segment",
        ":trt_allocator",
        ":trt_parameters",
        ":trt_plugins",
        ":trt_logging",
        ":trt_resources",
        ":utils",
        ":trt_weights",
        ":op_converter",
        ":op_converter_registry",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:array_ops",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/profiler/lib:annotated_traceme",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/tools/graph_transforms:transform_utils",
    ] + if_tensorrt([":tensorrt_lib"]) + tf_custom_op_library_additional_deps() + select({
        ":use_efficient_nms_plugin": [":efficient_nms_plugin"],
        "//conditions:default": [],
    }),
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
        ":testutils",
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
    srcs = [
        "convert/convert_nodes_test.cc",
        "convert/op_converter_test.cc",
    ],
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
        ":testutils",
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
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:resource_variable_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
    ] + if_tensorrt([
        ":tensorrt_lib",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

tf_cuda_cc_test(
    name = "convert_qdq_test",
    size = "medium",
    srcs = [
        "convert/ops/quantization_ops_test.cc",
    ],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":trt_logging",
        ":trt_conversion",
        ":trt_convert_api",
        ":trt_plugins",
        ":trt_engine_utils",
        ":trt_op_kernels",
        ":trt_resources",
        ":utils",
        ":testutils",
        "//tensorflow/compiler/jit:shape_inference",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:nn",
        "//tensorflow/core/kernels:pooling_ops",
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
        "@com_google_absl//absl/container:flat_hash_set",
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
    srcs = [
        "convert/utils.cc",
        "utils/trt_experimental_features.cc",
    ],
    hdrs = [
        "common/utils.h",
        "convert/utils.h",
        "utils/trt_experimental_features.h",
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

tf_cuda_library(
    name = "py_utils",
    srcs = ["utils/py_utils.cc"],
    hdrs = ["utils/py_utils.h"],
    local_defines = select({
        "@local_config_tensorrt//:use_static_tensorrt": ["TF_USE_TENSORRT_STATIC=1"],
        "//conditions:default": [],
    }),
    deps = if_tensorrt([
        ":common_utils",
        ":tensorrt_lib",
        ":op_converter_registry",
        "//tensorflow/compiler/xla/stream_executor/platform:dso_loader",
    ]),
)

pybind_extension(
    name = "_pywrap_py_utils",
    srcs = ["utils/py_utils_wrapper.cc"],
    link_in_framework = True,
    static_deps = [
        # TODO(b/229550590): Uncomment to use cc_shared_library instead of cc_binary.
        # "@bazel_tools//:__subpackages__",
        # "@boringssl//:__subpackages__",
        # "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
        # "@com_google_absl//:__subpackages__",
        # "@com_google_googleapis//:__subpackages__",
        # "@com_google_protobuf//:__subpackages__",
        # "@com_googlesource_code_re2//:__subpackages__",
        # "@curl//:__subpackages__",
        # "@double_conversion//:__subpackages__",
        # "@eigen_archive//:__subpackages__",
        # "@farmhash_archive//:__subpackages__",
        # "@fft2d//:__subpackages__",
        # "@gif//:__subpackages__",
        # "@highwayhash//:__subpackages__",
        # "@hwloc//:__subpackages__",
        # "@jsoncpp_git//:__subpackages__",
        # "@libjpeg_turbo//:__subpackages__",
        # "@libxsmm_archive//:__subpackages__",
        # "@llvm_openmp//:__subpackages__",
        # "@llvm-project//:__subpackages__",
        # "@llvm_terminfo//:__subpackages__",
        # "@llvm_zlib//:__subpackages__",
        # "@local_config_cuda//:__subpackages__",
        # "@local_config_git//:__subpackages__",
        # "@local_config_python//:__subpackages__",
        # "@local_config_rocm//:__subpackages__",
        # "@local_config_tensorrt//:__subpackages__",
        # "@local_execution_config_platform//:__subpackages__",
        # "@nsync//:__subpackages__",
        # "@platforms//:__subpackages__",
        # "@pybind11//:__subpackages__",
        # "@snappy//:__subpackages__",
        # "//:__subpackages__",
        # "@zlib//:__subpackages__",
    ],
    deps = [
        ":common_utils",
        ":py_utils",
        "//tensorflow/compiler/xla/stream_executor",
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
