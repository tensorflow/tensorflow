#
# Public targets:
#  ":platform" - Low-level and platform-specific Python code.

load("//tensorflow:tensorflow.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "cc_header_only_library", "if_mlir", "if_not_windows", "if_xla_available", "py_test", "py_tests", "tf_cc_shared_object", "tf_cc_test", "tf_cuda_library", "tf_enable_mlir_bridge", "tf_gen_op_wrapper_py")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_monitoring_python_deps")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pywrap_tensorflow_macro")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_external_workspace_visible")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_static", "tf_additional_plugin_deps", "tf_additional_profiler_deps", "tf_additional_xla_deps_py")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")

# TODO(mdan): Break into per-directory files.

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//tensorflow:internal",
    "//tensorflow/lite/toco/python:__pkg__",
    "//tensorflow_models:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/courier:__subpackages__",
    "//third_party/py/courier:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/tensorflow_examples:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/tensorflow_docs:__subpackages__",
    "//third_party/py/keras:__subpackages__",
    "//third_party/py/tensorflow_gnn:__subpackages__",
]

package(
    default_visibility = visibility,
    licenses = ["notice"],  # Apache 2.0
)

# Description:

py_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/aot/tests:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/lite/toco/python:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/debug:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/tools:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/tools/quantization:__pkg__",  # TODO(b/34059704): remove when fixed
    ],
    deps = [
        ":no_contrib",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/tpu:tpu_estimator",
    ],
)

py_library(
    name = "keras_lib",
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:internal",
        "//tensorflow/python/estimator:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//tensorflow/tools/api/tests:__pkg__",
        "//tensorflow/tools/compatibility/update:__pkg__",
        "//tensorflow_estimator:__subpackages__",
        "//third_party/py/tensorflow_privacy:__subpackages__",  # TODO(b/163395075): remove when fixed
    ],
    deps = [
        ":layers",
        ":rnn",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/keras",
    ],
)

py_library(
    name = "no_contrib",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python/estimator:__subpackages__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//tensorflow/tools/api/tests:__pkg__",
        "//tensorflow/tools/compatibility/update:__pkg__",
        "//third_party/py/keras:__subpackages__",
        "//third_party/py/tensorflow_core:__subpackages__",
    ],
    deps = [
        ":_pywrap_events_writer",
        ":_pywrap_py_exception_registry",
        ":_pywrap_python_op_gen",
        ":_pywrap_quantize_training",
        ":_pywrap_utils",
        ":array_ops",
        ":audio_ops_gen",
        ":bincount_ops",
        ":bitwise_ops",
        ":boosted_trees_ops",
        ":check_ops",
        ":client",
        ":client_testlib",
        ":clustering_ops",
        ":collective_ops",
        ":cond_v2",
        ":config",
        ":confusion_matrix",
        ":control_flow_ops",
        ":cudnn_rnn_ops_gen",
        ":distributed_framework_test_lib",
        ":errors",
        ":framework",
        ":framework_combinations",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":graph_util",
        ":histogram_ops",
        ":image_ops",
        ":initializers_ns",
        ":io_ops",
        ":keras_lib",
        ":kernels",
        ":lib",
        ":list_ops",
        ":manip_ops",
        ":map_fn",
        ":map_ops",
        ":math_ops",
        ":metrics",
        ":nccl_ops",
        ":nn",
        ":ops",
        ":platform",
        ":proto_ops",
        ":pywrap_tensorflow",
        ":pywrap_tf_session",
        ":pywrap_tfe",
        ":rnn_ops_gen",
        ":script_ops",
        ":sendrecv_ops_gen",
        ":session_ops",
        ":sets",
        ":sparse_ops",
        ":standard_ops",
        ":state_ops",
        ":string_ops",
        ":subscribe",
        ":summary",
        ":tensor_array_ops",
        ":test_ops",  # TODO: Break testing code out into separate rule.
        ":tf_cluster",
        ":tf_item",
        ":tf_optimizer",
        ":training",
        ":weights_broadcast_ops",
        ":while_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/compat",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/compiler",
        "//tensorflow/python/data",
        "//tensorflow/python/debug:debug_py",
        "//tensorflow/python/distribute",
        "//tensorflow/python/distribute:combinations",  # For tf.__internal__ API.
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:estimator_training",
        "//tensorflow/python/distribute:strategy_combinations",  # For tf.__internal__,
        "//tensorflow/python/dlpack",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:profiler",
        "//tensorflow/python/eager:profiler_client",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg/sparse",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/ops/parallel_for",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/profiler",
        "//tensorflow/python/profiler:profiler_client",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/tools:module_util",
        "//tensorflow/python/tools/api/generator:create_python_api",
        "//tensorflow/python/tpu:tpu_noestimator",
        "//tensorflow/python/training:saver_test_utils",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "core",
    visibility = [
        "//tensorflow:__pkg__",
    ],
    deps = [
        "//tensorflow/python/types",
        "//tensorflow/python/util:core",
    ],
)

# This target should only be used for API generation.
py_library(
    name = "modules_with_exports",
    srcs = ["modules_with_exports.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
    ],
    deps = [
        ":no_contrib",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
    ],
)

# TODO(gunan): Investigate making this action hermetic so we do not need
# to run it locally.
cc_library(
    name = "cost_analyzer_lib",
    srcs = ["grappler/cost_analyzer.cc"],
    hdrs = ["grappler/cost_analyzer.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler/costs:analytical_cost_estimator",
        "//tensorflow/core/grappler/costs:measuring_cost_estimator",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:cost_estimator",
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
    srcs = ["grappler/cost_analyzer_wrapper.cc"],
    hdrs = [
        "grappler/cost_analyzer.h",
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/public:session.h",
        "//tensorflow/core/public:session_options.h",
    ],
    module_name = "_pywrap_cost_analyzer",
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
    srcs = ["grappler/model_analyzer.cc"],
    hdrs = ["grappler/model_analyzer.h"],
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
    srcs = ["grappler/model_analyzer_wrapper.cc"],
    hdrs = [
        "grappler/model_analyzer.h",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
    ],
    module_name = "_pywrap_model_analyzer",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

alias(
    name = "util",
    actual = "//tensorflow/python/util:util",
    visibility = visibility + [
        "//tensorflow:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tfx:__subpackages__",
    ],
)

alias(
    name = "tf_decorator",
    actual = "//tensorflow/python/util:tf_decorator",
)

alias(
    name = "bfloat16_lib",
    actual = "//tensorflow/python/lib/core:bfloat16_lib",
)

# Necessary for the pywrap inclusion below.
tf_pybind_cc_library_wrapper(
    name = "tfcompile_headers_lib",
    compatible_with = [],
    deps = [
        "//tensorflow/compiler/aot:tfcompile_lib",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfcompile",
    srcs = ["tfcompile_wrapper.cc"],
    features = ["-layering_check"],
    module_name = "_pywrap_tfcompile",
    deps = [
        ":tfcompile_headers_lib",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        # The headers here cannot be brought in via cc_header_only_library
        "//tensorflow/compiler/aot:llvm_targets",
    ],
)

py_library(
    name = "pywrap_tf_session",
    srcs = ["client/pywrap_tf_session.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_session",
        ":pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_session",
    srcs = ["client/tf_session_wrapper.cc"],
    hdrs = [
        "client/tf_session_helper.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:numpy_hdr",
        "//tensorflow/python/lib/core:safe_ptr_hdr",
    ],
    module_name = "_pywrap_tf_session",
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//third_party/py/numpy:headers",
        "//tensorflow/c:pywrap_required_hdrs",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/core/lib/llvm_rtti",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
            "//tensorflow/core:version_lib",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

alias(
    name = "_pywrap_utils",
    actual = "//tensorflow/python/util:_pywrap_utils",
)

tf_python_pybind_extension(
    name = "_pywrap_quantize_training",
    srcs = [
        "//tensorflow/python/training:quantize_training_wrapper.cc",
    ],
    hdrs = ["//tensorflow/core/common_runtime:quantize_training_hdrs"],
    module_name = "_pywrap_quantize_training",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_debug_events_writer",
    srcs = ["client/debug_events_writer_wrapper.cc"],
    module_name = "_pywrap_debug_events_writer",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_events_writer",
    srcs = ["client/events_writer_wrapper.cc"],
    module_name = "_pywrap_events_writer",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

# TODO(yanhuasun): Move this back and the source file back to lib/core directory.
tf_python_pybind_extension(
    name = "_pywrap_py_exception_registry",
    srcs = ["py_exception_registry_wrapper.cc"],
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/python/lib/core:py_exception_registry_hdr",
    ],
    module_name = "_pywrap_py_exception_registry",
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:fixed_array",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_toco_api",
    srcs = [
        "lite/toco_python_api_wrapper.cc",
    ],
    hdrs = ["//tensorflow/lite/toco/python:toco_python_api_hdrs"],
    module_name = "_pywrap_toco_api",
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

# TODO(edloper): Remove unused dependency on safe_ptr.  (Blocker: there are
# targets that depend are relying on cpp_python_util to pull in safe_ptr's
# third_party/tensorflow/c:c_api_no_xla dependency, which registers
# ops/gradients, rather than depending on it themselves.)
cc_header_only_library(
    name = "py_func_headers_lib",
    features = ["-parse_headers"],
    tags = ["no-ide"],
    deps = [
        "//tensorflow/python/lib/core:py_func_lib",
    ],
)

tf_cc_shared_object(
    name = "framework/test_file_system.so",
    srcs = ["framework/test_file_system.cc"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    linkopts = select({
        "//conditions:default": [
            "-lm",
        ],
        "//tensorflow:macos": [],
        "//tensorflow:windows": [],
    }),
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
    ],
)

tf_py_test(
    name = "file_system_test",
    size = "small",
    srcs = ["framework/file_system_test.py"],
    data = [":framework/test_file_system.so"],
    main = "framework/file_system_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",  # Path issues due to test environment
        "no_windows",
        "notap",
    ],
    deps = [
        ":client_testlib",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":io_ops",
        ":platform",
        "//tensorflow/python/util",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = [
        "framework/python_op_gen.cc",
        "framework/python_op_gen_internal.cc",
    ],
    hdrs = [
        "framework/python_op_gen.h",
        "framework/python_op_gen_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_header_only_library(
    name = "python_op_gen_headers_lib",
    extra_deps = [
        "//tensorflow/core:protos_all_cc",
    ],
    features = ["-parse_headers"],
    tags = ["no-ide"],
    deps = [
        ":python_op_gen",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_python_op_gen",
    srcs = ["framework/python_op_gen_wrapper.cc"],
    module_name = "_pywrap_python_op_gen",
    deps = [
        ":python_op_gen_headers_lib",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "python_op_gen_main",
    srcs = ["framework/python_op_gen_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":python_op_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "python_op_gen_test",
    srcs = ["framework/python_op_gen_test.cc"],
    deps = [
        ":python_op_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

py_library(
    name = "framework_for_generated_wrappers",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":constant_op",
        ":device",
        ":device_spec",
        ":dtypes",
        ":framework_ops",
        ":function",
        ":op_def_library",
        ":op_def_registry",
        ":registry",
        ":tensor_shape",
        ":versions",
    ],
)

# What is needed for tf_gen_op_wrapper_py. This is the same as
# "framework_for_generated_wrappers" minus the "function" dep. This is to avoid
# circular dependencies, as "function" uses generated op wrappers.
py_library(
    name = "framework_for_generated_wrappers_v2",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":constant_op",
        ":device",
        ":device_spec",
        ":dtypes",
        ":framework_ops",
        ":op_def_library",
        ":op_def_registry",
        ":registry",
        ":tensor_shape",
        ":versions",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "subscribe",
    srcs = ["framework/subscribe.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_ops",
        ":platform",
        ":variables",
    ],
)

py_library(
    name = "framework",
    srcs = [
        "framework/framework_lib.py",
        "framework/graph_io.py",
        "framework/importer.py",
        "framework/load_library.py",
        "framework/meta_graph.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_debug_events_writer",
        ":_pywrap_events_writer",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        ":_pywrap_py_exception_registry",
        "//tensorflow/python/lib/core:_pywrap_py_func",  # TODO(b/142001480): remove once the bug is fixed.
        ":_pywrap_python_api_dispatcher",
        ":_pywrap_python_api_info",
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_op_gen",
        ":_pywrap_quantize_training",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        ":_pywrap_utils",
        ":_errors_test_helper",
        ":composite_tensor",
        ":config",
        ":convert_to_constants",
        ":cpp_shape_inference_proto_py",
        ":errors",
        ":framework_fast_tensor_util",
        ":framework_for_generated_wrappers",
        ":function",
        ":graph_util",
        ":lib",
        ":platform",
        ":pywrap_tensorflow",
        ":pywrap_tf_session",
        ":pywrap_tfe",
        ":pywrap_mlir",
        ":random_seed",
        ":sparse_tensor",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        "//tensorflow/python/util:util",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python/eager:context",
    ] + if_xla_available([
        ":_pywrap_tfcompile",
    ]),
)

py_library(
    name = "c_api_util",
    srcs = ["framework/c_api_util.py"],
    srcs_version = "PY3",
    deps = [
        ":pywrap_tf_session",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "common_shapes",
    srcs = ["framework/common_shapes.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_shape",
    ],
)

py_library(
    name = "constant_op",
    srcs = ["framework/constant_op.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/profiler:traceme",
    ],
)

py_library(
    name = "device_spec",
    srcs = ["framework/device_spec.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "device",
    srcs = ["framework/device.py"],
    srcs_version = "PY3",
)

tf_python_pybind_extension(
    name = "_dtypes",
    srcs = ["framework/dtypes.cc"],
    module_name = "_dtypes",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
        "@pybind11",
    ],
)

py_library(
    name = "dtypes",
    srcs = ["framework/dtypes.py"],
    srcs_version = "PY3",
    deps = [
        ":_dtypes",
        ":pywrap_tensorflow",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/lib/core:_pywrap_bfloat16",
    ],
)

py_library(
    name = "errors",
    srcs = [
        "framework/errors.py",
        "framework/errors_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_py_exception_registry",
        ":c_api_util",
        ":error_interpolation",
        ":pywrap_tf_session",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "error_interpolation",
    srcs = [
        "framework/error_interpolation.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "function",
    srcs = ["framework/function.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":graph_to_function_def",
        ":op_def_registry",
        ":pywrap_tf_session",
        ":variable_scope",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "graph_to_function_def",
    srcs = ["framework/graph_to_function_def.py"],
    srcs_version = "PY3",
    deps = [
        ":op_def_registry",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "function_def_to_graph",
    srcs = ["framework/function_def_to_graph.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":framework",
        ":framework_ops",
        ":function",
        ":tensor_shape",
        ":versions",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "function_def_to_graph_test",
    size = "small",
    srcs = ["framework/function_def_to_graph_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":constant_op",
        ":dtypes",
        ":framework_ops",
        ":function",
        ":function_def_to_graph",
        ":graph_to_function_def",
        ":math_ops",
        ":op_def_library",
        ":test_ops",
    ],
)

py_library(
    name = "graph_util",
    srcs = [
        "framework/graph_util.py",
        "framework/graph_util_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_proto_comparators",
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "convert_to_constants",
    srcs = [
        "framework/convert_to_constants.py",
    ],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":dtypes",
        ":framework_ops",
        ":platform",
        ":tensor_util",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "kernels",
    srcs = [
        "framework/kernels.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":pywrap_tf_session",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "op_def_library",
    srcs = ["framework/op_def_library.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":op_callbacks",
        ":op_def_registry",
        ":platform",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_registry",
    srcs = ["framework/op_def_registry.cc"],
    module_name = "_op_def_registry",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_library(
    name = "op_def_registry",
    srcs = ["framework/op_def_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":_op_def_registry",
        "//tensorflow/core:protos_all_py",
    ],
)

cc_library(
    name = "py_context_manager",
    srcs = ["framework/py_context_manager.cc"],
    hdrs = ["framework/py_context_manager.h"],
    deps = [
        "//tensorflow/core:lib",  # for core/platform/logging.h
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
    ],
)

# Pybind extension used by py_context_manager_test.
tf_python_pybind_extension(
    name = "_py_context_manager",
    srcs = ["framework/py_context_manager_pybind.cc"],
    module_name = "_py_context_manager",
    deps = [
        ":py_context_manager",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_test(
    name = "py_context_manager_test",
    srcs = ["framework/py_context_manager_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_py_context_manager",
    ],
)

cc_library(
    name = "op_def_util_cc",
    srcs = ["framework/op_def_util.cc"],
    hdrs = ["framework/op_def_util.h"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

# Note: this target is only used for op_def_util_test.  It includes op_def_util.cc
# directly in its srcs (rather than depending on the `op_def_util_cc` target) because
# depending on that target adds dependencies that register objects; and since the
# extension is built as a shared object in some kokoro tests, this causes those objects
# to get registered multiple times (which fails).
# TODO(edloper): Simplify this, once cpp_python_util is changed to not depend on
# safe_ptr (which transitively depends on third_party/tensorflow/c:c_api_no_xla).
tf_python_pybind_extension(
    name = "_op_def_util",
    srcs = [
        "framework/op_def_util.cc",
        "framework/op_def_util_pybind.cc",
    ],
    hdrs = [
        "framework/op_def_util.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/python/lib/core:safe_ptr_hdr",
        "//tensorflow/python/util:util_hdr",
    ],
    module_name = "_op_def_util",
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

tf_py_test(
    name = "op_def_util_test",
    srcs = ["framework/op_def_util_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
)

cc_library(
    name = "python_api_parameter_converter",
    srcs = ["framework/python_api_parameter_converter.cc"],
    hdrs = ["framework/python_api_parameter_converter.h"],
    deps = [
        ":op_def_util_cc",
        ":python_api_info",
        ":python_tensor_converter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

# Note: this target is only used by python_api_parameter_converter_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_parameter_converter",
    srcs = ["framework/python_api_parameter_converter_wrapper.cc"],
    hdrs = [
        "framework/op_def_util.h",
        "framework/python_api_info.h",
        "framework/python_api_parameter_converter.h",
        "framework/python_tensor_converter.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:numpy_hdr",
    ],
    module_name = "_pywrap_python_api_parameter_converter",
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//third_party/py/numpy:headers",
        "//tensorflow/c:pywrap_required_hdrs",
        "@com_google_absl//absl/types:span",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_test(
    name = "python_api_parameter_converter_test",
    srcs = ["framework/python_api_parameter_converter_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_tensor_converter",
        ":client_testlib",
    ],
)

cc_library(
    name = "python_api_info",
    srcs = ["framework/python_api_info.cc"],
    hdrs = ["framework/python_api_info.h"],
    deps = [
        ":op_def_util_cc",
        ":python_tensor_converter",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

# Note: this target is only used by python_api_info_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_info",
    srcs = ["framework/python_api_info_wrapper.cc"],
    hdrs = [
        "framework/op_def_util.h",
        "framework/python_api_info.h",
        "framework/python_tensor_converter.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:numpy_hdr",
    ],
    module_name = "_pywrap_python_api_info",
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//third_party/py/numpy:headers",
        "//tensorflow/c:pywrap_required_hdrs",
        "@com_google_absl//absl/types:span",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_test(
    name = "python_api_info_test",
    srcs = ["framework/python_api_info_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_info",
        ":_pywrap_python_tensor_converter",
        ":client_testlib",
    ],
)

cc_library(
    name = "python_api_dispatcher",
    srcs = ["framework/python_api_dispatcher.cc"],
    hdrs = ["framework/python_api_dispatcher.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

# Note: this target is only used by python_api_dispatcher_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_dispatcher",
    # testonly = True,
    srcs = ["framework/python_api_dispatcher_wrapper.cc"],
    hdrs = ["framework/python_api_dispatcher.h"],
    module_name = "_pywrap_python_api_dispatcher",
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

tf_py_test(
    name = "python_api_dispatcher_test",
    srcs = ["framework/python_api_dispatcher_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_dispatcher",
        ":client_testlib",
    ],
)

cc_library(
    name = "python_tensor_converter",
    srcs = ["framework/python_tensor_converter.cc"],
    hdrs = ["framework/python_tensor_converter.h"],
    deps = [
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

# Note: this target is only used by python_tensor_converter_test.
tf_python_pybind_extension(
    name = "_pywrap_python_tensor_converter",
    srcs = ["framework/python_tensor_converter_wrapper.cc"],
    hdrs = [
        "framework/python_tensor_converter.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:numpy_hdr",
    ],
    module_name = "_pywrap_python_tensor_converter",
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//third_party/py/numpy:headers",
        "//tensorflow/c:pywrap_required_hdrs",
        "@com_google_absl//absl/types:span",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_test(
    name = "python_tensor_converter_test",
    srcs = ["framework/python_tensor_converter_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_tensor_converter",
        ":client_testlib",
    ],
)

py_library(
    name = "framework_ops",  # "ops" is already the name of a deprecated target
    srcs = ["framework/ops.py"],
    srcs_version = "PY3",
    deps = [
        ":_op_def_util",
        ":c_api_util",
        ":control_flow_util",
        ":device",
        ":dtypes",
        ":error_interpolation",
        ":indexed_slices",
        ":op_def_registry",
        ":platform",
        ":registry",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tf2",
        ":traceable_stack",
        ":type_spec",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/profiler:traceme",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "op_callbacks",
    srcs = ["framework/op_callbacks.py"],
    srcs_version = "PY3",
)

cuda_py_test(
    name = "op_callbacks_test",
    srcs = ["framework/op_callbacks_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":control_flow_ops",
        ":framework_test_lib",
        ":script_ops",
        ":sparse_ops",
        ":sparse_tensor",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "indexed_slices",
    srcs = ["framework/indexed_slices.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":type_spec",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "tensor_conversion_registry",
    srcs = ["framework/tensor_conversion_registry.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "map_fn",
    srcs = ["ops/map_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "func_graph",
    srcs = ["framework/func_graph.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps",
        ":framework_ops",
        ":sparse_tensor",
        ":tensor_array_ops",
        "//tensorflow/python/autograph",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "auto_control_deps",
    srcs = ["framework/auto_control_deps.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps_utils",
        ":control_flow_ops",
        ":framework_ops",
        ":sparse_tensor",
        ":tensor_array_ops",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "auto_control_deps_utils",
    srcs = ["framework/auto_control_deps_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
    ],
)

tf_py_test(
    name = "auto_control_deps_test",
    size = "small",
    srcs = ["framework/auto_control_deps_test.py"],
    python_version = "PY3",
    deps = [
        ":auto_control_deps",
        ":client_testlib",
        ":sendrecv_ops_gen",
    ],
)

py_library(
    name = "config",
    srcs = ["framework/config.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_ops",
        "//tensorflow/python/util",
    ],
)

cuda_py_test(
    name = "config_test",
    size = "small",
    srcs = ["framework/config_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # test_ops are not available in pip.
    deps = [
        ":client_testlib",
        ":config",
        ":constant_op",
        ":platform",
        ":test_ops",
        "//tensorflow/python/util:util",
    ] + tf_additional_xla_deps_py(),
)

py_library(
    name = "random_seed",
    srcs = ["framework/random_seed.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_ops",
    ],
)

py_library(
    name = "registry",
    srcs = ["framework/registry.py"],
    srcs_version = "PY3",
    deps = [
        ":platform",
        "//tensorflow/python/util",
        # TODO(mdan): Remove this once the transitive dependency is fixed.
        "//tensorflow/python/util:tf_stack",
    ],
)

py_library(
    name = "smart_cond",
    srcs = ["framework/smart_cond.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":pywrap_tf_session",
        ":tensor_util",
    ],
)

tf_py_test(
    name = "smart_cond_test",
    size = "small",
    srcs = ["framework/smart_cond_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":constant_op",
        ":framework_ops",
        ":math_ops",
        ":session",
        ":smart_cond",
    ],
)

py_library(
    name = "sparse_tensor",
    srcs = ["framework/sparse_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":framework_ops",
        ":tensor_util",
        ":type_spec",
        "//tensorflow/python/types",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["framework/composite_tensor.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":dtypes",
        ":tensor_util",
    ],
)

py_test(
    name = "framework_composite_tensor_test",
    srcs = ["framework/composite_tensor_test.py"],
    main = "framework/composite_tensor_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "tensor_shape",
    srcs = ["framework/tensor_shape.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":tf2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "type_spec",
    srcs = ["framework/type_spec.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":dtypes",
        ":tensor_shape",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tensor_spec",
    srcs = ["framework/tensor_spec.py"],
    srcs_version = "PY3",
    deps = [
        ":common_shapes",
        ":dtypes",
        ":tensor_shape",
        ":type_spec",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tensor_util",
    srcs = ["framework/tensor_util.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/types",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "traceable_stack",
    srcs = ["framework/traceable_stack.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "versions",
    srcs = ["framework/versions.py"],
    srcs_version = "PY3",
    deps = [
        ":pywrap_tf_session",
    ],
)

py_library(
    name = "extra_py_tests_deps",
    srcs_version = "PY3",
    deps = [
        ":keras_lib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "gpu_util",
    srcs = ["framework/gpu_util.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "framework_test_lib",
    srcs = ["framework/test_util.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow_estimator/python/estimator:__subpackages__",
    ],
    deps = [
        ":array_ops",
        ":client",
        ":errors",
        ":framework_for_generated_wrappers",
        ":gpu_util",
        ":platform",
        ":platform_test",
        ":pywrap_tf_session",
        ":random_seed",
        ":resource_variable_ops",
        ":session",
        ":tensor_array_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

# Including this as a dependency will result in tests using
# :framework_test_lib to use XLA.
py_library(
    name = "is_xla_test_true",
    srcs = ["framework/is_xla_test_true.py"],
    srcs_version = "PY3",
)

# Including this as a dependency will result in tests using
# :framework_test_lib to use MLIR.
py_library(
    name = "is_mlir_bridge_test_true",
    srcs = ["framework/is_mlir_bridge_test_true.py"],
    srcs_version = "PY3",
    visibility = visibility,
)

# Including this as a dependency will result in tests using
# :framework_test_lib to NOT use MLIR.
py_library(
    name = "is_mlir_bridge_test_false",
    srcs = ["framework/is_mlir_bridge_test_false.py"],
    srcs_version = "PY3",
    visibility = visibility,
)

# Including this as a dependency will result in tests to use TFRT.
# TODO(b/170139514): Add a curated list of TFRT native ops.
# TODO(kkb): Deprecated by `tfrt_utils`, remove.
py_library(
    name = "is_tfrt_test_true",
    srcs = ["framework/is_tfrt_test_true.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tfrt_utils",
    srcs = ["framework/tfrt_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "distributed_framework_test_lib",
    srcs_version = "PY3",
    deps = [":framework_test_lib"],
)

py_library(
    name = "framework_combinations",
    srcs = ["framework/combinations.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_ops",
        ":framework_test_combinations_lib",
        ":tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "framework_test_combinations_lib",
    srcs = ["framework/test_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "test_combinations_test",
    srcs = ["framework/test_combinations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":framework_test_combinations_lib",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "memory_checker",
    srcs = [
        "framework/memory_checker.py",
        "framework/python_memory_checker.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_python_memory_checker_helper",
        "//tensorflow/python/profiler:traceme",
    ],
)

tf_python_pybind_extension(
    name = "_python_memory_checker_helper",
    srcs = ["framework/python_memory_checker_helper.cc"],
    module_name = "_python_memory_checker_helper",
    deps = [
        "@pybind11",
    ],
)

tf_py_test(
    name = "framework_constant_op_test",
    size = "small",
    srcs = ["framework/constant_op_test.py"],
    main = "framework/constant_op_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
    ],
)

tf_py_test(
    name = "framework_registry_test",
    size = "small",
    srcs = ["framework/registry_test.py"],
    main = "framework/registry_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "framework_errors_test",
    size = "small",
    srcs = ["framework/errors_test.py"],
    main = "framework/errors_test.py",
    python_version = "PY3",
    deps = [
        ":_errors_test_helper",
        ":client_testlib",
        ":errors",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "framework_error_interpolation_test",
    size = "small",
    srcs = ["framework/error_interpolation_test.py"],
    main = "framework/error_interpolation_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":constant_op",
        ":error_interpolation",
        ":traceable_stack",
    ],
)

tf_py_test(
    name = "framework_subscribe_test",
    size = "small",
    srcs = ["framework/subscribe_test.py"],
    main = "framework/subscribe_test.py",
    python_version = "PY3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":script_ops",
        ":subscribe",
    ],
)

tf_py_test(
    name = "proto_test",
    size = "small",
    srcs = ["framework/proto_test.py"],
    main = "framework/proto_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "functional_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

py_library(
    name = "functional_ops",
    srcs = ["ops/functional_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":auto_control_deps_utils",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        ":functional_ops_gen",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":variable_scope",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
    ],
)

py_test(
    name = "functional_ops_test",
    srcs = ["ops/functional_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":dtypes",
        ":function",
        ":functional_ops",
        ":tensor_spec",
        "//tensorflow/python/eager:def_function",
    ],
)

cuda_py_test(
    name = "function_test",
    size = "medium",
    srcs = ["framework/function_test.py"],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
        "noasan",
        "optonly",
    ],
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":clip_ops",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradients",
        ":init_ops",
        ":logging_ops",
        ":logging_ops_gen",
        ":math_ops",
        ":nn_ops",
        ":platform",
        ":random_ops",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "framework_versions_test",
    size = "small",
    srcs = ["framework/versions_test.py"],
    main = "framework/versions_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "framework_importer_test",
    size = "small",
    srcs = ["framework/importer_test.py"],
    main = "framework/importer_test.py",
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":random_ops",
        ":test_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "meta_graph_testdata",
    srcs = [
        "framework/testdata/metrics_export_meta_graph.pb",
    ],
    visibility = ["//visibility:public"],
)

tf_py_test(
    name = "framework_meta_graph_test",
    size = "small",
    srcs = ["framework/meta_graph_test.py"],
    data = [":meta_graph_testdata"],
    main = "framework/meta_graph_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":metrics",
        ":nn_ops",
        ":platform",
        ":random_ops",
        ":training",
        ":variables",
    ],
)

tf_py_test(
    name = "framework_traceable_stack_test",
    size = "small",
    srcs = ["framework/traceable_stack_test.py"],
    main = "framework/traceable_stack_test.py",
    python_version = "PY3",
    deps = [
        ":framework_test_lib",
        ":platform_test",
        ":test_ops",
        ":traceable_stack",
        "//tensorflow/python/util",
    ],
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "framework/test_ops.py",
    deps = [":test_ops_kernels"],
)

tf_gen_op_wrapper_private_py(
    name = "debug_ops_gen",
    out = "ops/gen_debug_ops.py",
    visibility = ["//tensorflow/python/debug:__pkg__"],
    deps = ["//tensorflow/core:debug_ops_op_lib"],
)

cc_library(
    name = "test_ops_kernels",
    srcs = ["framework/test_ops.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "test_ops_2",
    out = "framework/test_ops_2.py",
    deps = [":test_ops_2_kernels"],
)

cc_library(
    name = "test_ops_2_kernels",
    srcs = ["framework/test_ops_2.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_py_test(
    name = "framework_common_shapes_test",
    size = "small",
    srcs = ["framework/common_shapes_test.py"],
    main = "framework/common_shapes_test.py",
    python_version = "PY3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "framework_ops_test",
    size = "small",
    srcs = ["framework/ops_test.py"],
    main = "framework/ops_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # test_ops_2 is not available in pip.
    deps = [
        ":cond_v2",
        ":control_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":resources",
        ":test_ops",
        ":test_ops_2",
        ":variable_scope",
        ":variables",
        ":while_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util",
    ],
)

tf_py_test(
    name = "framework_ops_enable_eager_test",
    size = "small",
    srcs = ["framework/ops_enable_eager_test.py"],
    main = "framework/ops_enable_eager_test.py",
    python_version = "PY3",
    deps = [
        ":framework",
        ":platform_test",
        "//tensorflow/python/eager:context",
    ],
)

tf_py_test(
    name = "framework_tensor_shape_test",
    size = "small",
    srcs = ["framework/tensor_shape_test.py"],
    main = "framework/tensor_shape_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "framework_type_spec_test",
    size = "small",
    srcs = ["framework/type_spec_test.py"],
    main = "framework/type_spec_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":type_spec",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "framework_tensor_spec_test",
    size = "small",
    srcs = ["framework/tensor_spec_test.py"],
    main = "framework/tensor_spec_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        ":tensor_spec",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "framework_sparse_tensor_test",
    size = "small",
    srcs = ["framework/sparse_tensor_test.py"],
    main = "framework/sparse_tensor_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "framework_device_spec_test",
    size = "small",
    srcs = ["framework/device_spec_test.py"],
    main = "framework/device_spec_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "framework_device_test",
    size = "small",
    srcs = ["framework/device_test.py"],
    main = "framework/device_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "framework_random_seed_test",
    size = "small",
    srcs = ["framework/random_seed_test.py"],
    main = "framework/random_seed_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework",
    ],
)

tf_py_test(
    name = "framework_tensor_shape_div_test",
    size = "small",
    srcs = ["framework/tensor_shape_div_test.py"],
    main = "framework/tensor_shape_div_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "framework_tensor_util_test",
    size = "small",
    srcs = ["framework/tensor_util_test.py"],
    main = "framework/tensor_util_test.py",
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":state_ops_gen",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "framework_test_util_test",
    size = "small",
    srcs = ["framework/test_util_test.py"],
    main = "framework/test_util_test.py",
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":control_flow_ops",
        ":errors",
        ":framework_combinations",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":lookup_ops",
        ":platform_test",
        ":random_ops",
        ":resource_variable_ops",
        ":session",
        ":test_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "framework_memory_checker_test",
    size = "medium",
    srcs = ["framework/memory_checker_test.py"],
    main = "framework/memory_checker_test.py",
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
        "noasan",  # TODO(b/149948895): Re-enable.
        "nomsan",  # TODO(b/149948895): Re-enable.
        "notsan",  # TODO(b/149948895): Re-enable.
    ],
    deps = [
        ":framework_test_lib",
        # TODO(kkb): Find more appropriate place to add `memory_checker` as deps
        # Adding directly to `framework_test_lib` caused a Colab binary size
        # regression b/149433910 .
        ":memory_checker",
        ":_memory_checker_test_helper",
    ],
)

tf_python_pybind_extension(
    name = "_memory_checker_test_helper",
    srcs = ["framework/memory_checker_test_helper.cc"],
    module_name = "_memory_checker_test_helper",
    deps = [
        "@pybind11",
    ],
)

tf_py_test(
    name = "framework_dtypes_test",
    size = "small",
    srcs = ["framework/dtypes_test.py"],
    main = "framework/dtypes_test.py",
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["framework/op_def_library_test.py"],
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
    ],
)

tf_py_test(
    name = "framework_kernels_test",
    size = "small",
    srcs = ["framework/kernels_test.py"],
    main = "framework/kernels_test.py",
    python_version = "PY3",
    deps = [
        ":framework_test_lib",
        ":kernels",
        ":platform_test",
        ":test_ops",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "array_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/v1_compat_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
    deps = [
        "//tensorflow/c/kernels:bitcast_op_lib",
        "//tensorflow/core:array_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "bitwise_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "boosted_trees_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:boosted_trees_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "summary_ops_gen",
    visibility = ["//tensorflow:__subpackages__"],
    deps = ["//tensorflow/core:summary_ops_op_lib"],
)

tf_gen_op_wrapper_private_py(
    name = "audio_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "cudnn_rnn_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "candidate_sampling_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "checkpoint_ops_gen",
    visibility = [
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "clustering_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:clustering_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "collective_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:collective_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "control_flow_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
    deps = [
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "ctc_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "data_flow_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "dataset_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "experimental_dataset_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "image_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "io_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
        "//tensorflow/python/training/tracking:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "linalg_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "logging_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
    deps = [
        "//tensorflow/c/kernels:histogram_summary_op_lib",
        "//tensorflow/c/kernels:merge_summary_op_lib",
        "//tensorflow/c/kernels:summary_op_lib",
        "//tensorflow/core:logging_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "lookup_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "batch_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "batch_ops",
    srcs = [
        "ops/batch_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":batch_ops_gen",
    ],
)

cuda_py_test(
    name = "batch_ops_test",
    size = "small",
    srcs = ["ops/batch_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "no_cuda_asan",  # b/177916286
        "no_pip",
        "nomac",
    ],
    deps = [
        ":array_ops",
        ":batch_ops",
        ":client_testlib",
        ":dtypes",
        ":framework",
        ":gradients",
        ":script_ops",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "manip_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "math_ops_gen",
    visibility = [
        "//learning/brain/google/python/ops:__pkg__",
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "nn_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/tools:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "count_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "parsing_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "random_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "special_math_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "stateful_random_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "resource_variable_ops_gen",
    visibility = [
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/python/distribute:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "stateless_random_ops_gen",
    visibility = [
        "//tensorflow/python/data/experimental/ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "stateless_random_ops_v2_gen",
)

tf_gen_op_wrapper_private_py(
    name = "list_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "map_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "script_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "sdca_ops_gen",
    visibility = ["//tensorflow_estimator/python/estimator/canned/linear_optimizer:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "set_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "state_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "sparse_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "spectral_ops_gen",
    visibility = ["//tensorflow/python/ops/signal:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "string_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
)

tf_gen_op_wrapper_private_py(
    name = "training_ops_gen",
    visibility = [
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "ragged_array_ops_gen",
    visibility = ["//tensorflow/python/ops/ragged:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "ragged_math_ops_gen",
    visibility = ["//tensorflow/python/ops/ragged:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "ragged_conversion_ops_gen",
    visibility = ["//tensorflow/python/ops/ragged:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "risc_ops_gen",
    visibility = [
        "//tensorflow/python/ops/risc:__pkg__",
    ],
    deps = [
        "//tensorflow/core:risc_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(name = "rnn_ops_gen")

tf_gen_op_wrapper_private_py(
    name = "sendrecv_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:sendrecv_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "tpu_ops_gen",
    visibility = [
        "//smartass/brain/configure/python:__pkg__",
        "//tensorflow/python/tpu:__pkg__",
    ],
    deps = [
        "//tensorflow/core:tpu_configuration_ops_op_lib",
        "//tensorflow/core:tpu_cross_replica_ops_op_lib",
        "//tensorflow/core:tpu_embedding_load_retrieve_ops_op_lib",
        "//tensorflow/core:tpu_embedding_ops_op_lib",
        "//tensorflow/core:tpu_functional_ops_op_lib",
        "//tensorflow/core:tpu_heartbeat_ops_op_lib",
        "//tensorflow/core:tpu_host_compute_ops_op_lib",
        "//tensorflow/core:tpu_infeed_ops_op_lib",
        "//tensorflow/core:tpu_ordinal_selector_ops_op_lib",
        "//tensorflow/core:tpu_outfeed_ops_op_lib",
        "//tensorflow/core:tpu_replication_ops_op_lib",
    ],
)

py_library(
    name = "array_grad",
    srcs = ["ops/array_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
    ],
)

py_library(
    name = "array_ops",
    srcs = [
        "ops/array_ops.py",
        "ops/inplace_ops.py",
    ],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":array_ops_gen",
        ":common_shapes",
        ":constant_op",
        ":dtypes",
        ":framework_ops",
        ":math_ops_gen",
        ":sparse_tensor",
        ":tensor_shape",
        ":tensor_util",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "bitwise_ops",
    srcs = ["ops/bitwise_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":bitwise_ops_gen",
        ":framework",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "boosted_trees_ops",
    srcs = ["ops/boosted_trees_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":boosted_trees_ops_gen",
        ":framework",
        ":ops",
        ":training",
        "//tensorflow/core/kernels/boosted_trees:boosted_trees_proto_py",
    ],
)

py_library(
    name = "optional_grad",
    srcs = ["ops/optional_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_ops",
    ],
)

py_library(
    name = "sets",
    srcs = [
        "ops/sets.py",
        "ops/sets_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":set_ops_gen",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "candidate_sampling_ops",
    srcs = ["ops/candidate_sampling_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops_gen",
        ":framework",
        ":math_ops",
    ],
)

py_library(
    name = "check_ops",
    srcs = ["ops/check_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_tensor",
        ":tensor_util",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clip_ops",
    srcs = ["ops/clip_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_ops_gen",
        ":numerics",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["ops/clip_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clustering_ops",
    srcs = ["ops/clustering_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":clustering_ops_gen",
        ":framework",
        ":ops",
        ":training",
    ],
)

tf_py_test(
    name = "clustering_ops_test",
    size = "medium",
    srcs = ["ops/clustering_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":clustering_ops",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "collective_ops",
    srcs = ["ops/collective_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":collective_ops_gen",
        ":framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "collective_ops_test",
    size = "small",
    srcs = ["ops/collective_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
        "no_tfrt",  # TODO(b/179692150)
    ],
    deps = [
        ":client_testlib",
        ":collective_ops",
        ":framework_for_generated_wrappers",
        ":kernels",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "collective_ops_xla_test",
    size = "small",
    srcs = ["ops/collective_ops_xla_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_rocm",
        "no_windows",
        "nomac",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":client_testlib",
        ":collective_ops",
        ":framework_for_generated_wrappers",
        ":kernels",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "collective_ops_gpu_test",
    size = "small",
    srcs = ["ops/collective_ops_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_gpu",
        "no_windows",
    ],
    deps = [
        ":client_testlib",
        ":collective_ops",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "control_flow_grad",
    srcs =
        ["ops/control_flow_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":control_flow_ops_gen",
        ":control_flow_util",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

# Note: targets depending on this should also depend on ":cond_v2" and ":while_v2".
# See b/118513001.
py_library(
    name = "control_flow_ops",
    srcs = ["ops/control_flow_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":constant_op",
        ":control_flow_ops_gen",
        ":control_flow_util",
        ":dtypes",
        ":framework_ops",
        ":logging_ops_gen",
        ":math_ops",
        ":platform",
        ":sparse_tensor",
        ":tensor_array_ops",
        ":tensor_shape",
        ":tf2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_should_use",
        "@six_archive//:six",
    ],
)

py_library(
    name = "control_flow_util",
    srcs = ["ops/control_flow_util.py"],
    srcs_version = "PY3",
    deps = [
        ":platform",
    ],
)

py_library(
    name = "control_flow_util_v2",
    srcs = ["ops/control_flow_util_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        ":control_flow_v2_func_graphs",
        ":framework_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "control_flow_v2_func_graphs",
    srcs = ["ops/control_flow_v2_func_graphs.py"],
    srcs_version = "PY3",
    deps = [
        ":func_graph",
    ],
)

py_library(
    name = "control_flow_v2_toggles",
    srcs = ["ops/control_flow_v2_toggles.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        ":control_flow_util_v2",
        ":framework_ops",
        "//tensorflow/python/util",
    ],
)

tf_py_test(
    name = "control_flow_v2_toggles_test",
    size = "small",
    srcs = ["ops/control_flow_v2_toggles_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":control_flow_util_v2",
        ":control_flow_v2_toggles",
        ":platform_test",
    ],
)

tf_py_test(
    name = "control_flow_v2_enable_test",
    size = "small",
    srcs = ["ops/control_flow_v2_enable_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":control_flow_util",
        ":platform_test",
        ":tf2",
    ],
)

tf_py_test(
    name = "control_flow_v2_disable_test",
    size = "small",
    srcs = ["ops/control_flow_v2_disable_test.py"],
    python_version = "PY3",
    # This tests that it is possible to disable cfv2 using env vars.
    # This does not apply to TF 2.0 nightly builds which enable
    # v2 behavior using `tf.compat.v1.enable_v2_behavior()` in which case
    # `tf.compat.v1.disable_control_flow_v2()` needs to be used.
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":client_testlib",
        ":control_flow_util",
        ":platform_test",
        ":tf2",
    ],
)

py_library(
    name = "cond_v2",
    srcs = [
        "ops/cond_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":auto_control_deps_utils",
        ":c_api_util",
        ":control_flow_util_v2",
        ":framework_ops",
        ":function",
        ":function_def_to_graph",
        ":functional_ops_gen",
        ":gradients",
        ":gradients_util",
        ":graph_to_function_def",
        ":handle_data_util",
        ":pywrap_tensorflow",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "while_v2",
    srcs = [
        "ops/while_v2.py",
        "ops/while_v2_indexed_slices_rewriter.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":auto_control_deps_utils",
        ":constant_op",
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":dtypes",
        ":framework_ops",
        ":function_def_to_graph",
        ":functional_ops_gen",
        ":gradients_util",
        ":list_ops",
        ":map_ops",
        ":pywrap_tf_session",
        ":tensor_array_ops",
        ":tensor_shape",
        ":tensor_util",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "bincount_ops",
    srcs = ["ops/bincount_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":count_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        "//tensorflow/python/compat",
    ],
)

tf_py_test(
    name = "bincount_ops_test",
    size = "small",
    srcs = ["ops/bincount_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":bincount_ops",
        ":platform_test",
    ],
)

py_library(
    name = "ctc_ops",
    srcs = ["ops/ctc_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":ctc_ops_gen",
        ":framework",
        ":framework_for_generated_wrappers",
        ":nn_grad",
    ],
)

py_library(
    name = "cudnn_rnn_grad",
    srcs = ["ops/cudnn_rnn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":cudnn_rnn_ops_gen",
        ":framework_for_generated_wrappers",
    ],
)

py_library(
    name = "data_flow_grad",
    srcs = ["ops/data_flow_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "data_flow_ops",
    srcs = ["ops/data_flow_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":data_flow_ops_gen",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":random_seed",
        ":tensor_util",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_library(
    name = "embedding_ops",
    srcs = ["ops/embedding_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        ":resource_variable_ops",
        ":sparse_ops",
        ":tensor_shape",
        ":variables",
    ],
)

py_library(
    name = "handle_data_util",
    srcs = [
        "ops/handle_data_util.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":protos_all_py",
        ":pywrap_tf_session",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "gradients",
    srcs = [
        "ops/custom_gradient.py",
        "ops/gradients.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":gradients_impl",
        ":gradients_util",
        ":handle_data_util",
        ":pywrap_tf_session",
        ":unconnected_gradients",
        "//tensorflow/python/eager:forwardprop",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "gradients_impl",
    srcs = [
        "ops/gradients_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":bitwise_ops",
        ":check_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":control_flow_util",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":image_grad",
        ":linalg_grad",
        ":linalg_ops",
        ":logging_ops",
        ":manip_grad",
        ":manip_ops",
        ":math_grad",
        ":math_ops",
        ":optional_grad",
        ":platform",
        ":random_grad",
        ":tensor_array_ops",
        ":unconnected_gradients",
        "//tensorflow/python/ops/linalg/sparse",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "gradients_util",
    srcs = [
        "ops/gradients_util.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":control_flow_state",
        ":control_flow_util",
        ":default_gradient",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_ops",
        ":functional_ops",
        ":math_ops",
        ":platform",
        ":resource_variable_ops",
        ":tensor_util",
        ":unconnected_gradients",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "default_gradient",
    srcs = [
        "ops/default_gradient.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":resource_variable_ops",
    ],
)

py_library(
    name = "control_flow_state",
    srcs = [
        "ops/control_flow_state.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":control_flow_util",
        ":data_flow_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":tensor_util",
    ],
)

py_library(
    name = "unconnected_gradients",
    srcs = ["ops/unconnected_gradients.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "histogram_ops",
    srcs = ["ops/histogram_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
    ],
)

py_library(
    name = "image_grad",
    srcs = ["ops/image_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":image_ops_gen",
    ],
)

py_library(
    name = "image_ops",
    srcs = [
        "ops/image_ops.py",
        "ops/image_ops_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":clip_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":image_ops_gen",
        ":math_ops",
        ":nn",
        ":nn_ops_gen",
        ":random_ops",
        ":string_ops",
        ":variables",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "init_ops",
    srcs = ["ops/init_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":math_ops",
        ":random_ops",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "init_ops_v2",
    srcs = ["ops/init_ops_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":dtypes",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":math_ops",
        ":random_ops",
        ":stateless_random_ops",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "initializers_ns",
    srcs = ["ops/initializers_ns.py"],
    srcs_version = "PY3",
    deps = [
        ":init_ops",
        ":variables",
    ],
)

py_library(
    name = "io_ops",
    srcs = ["ops/io_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":io_ops_gen",
        ":lib",
    ],
)

py_library(
    name = "linalg_grad",
    srcs = ["ops/linalg_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":linalg_ops",
        ":math_ops",
        "//tensorflow/python/ops/linalg:linalg_impl",
    ],
)

py_library(
    name = "linalg_ops",
    srcs = ["ops/linalg_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":map_fn",
        ":math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "linalg_ops_impl",
    srcs = ["ops/linalg_ops_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "manip_grad",
    srcs = ["ops/manip_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":manip_ops",
    ],
)

py_library(
    name = "manip_ops",
    srcs = ["ops/manip_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":manip_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "logging_ops",
    srcs = ["ops/logging_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":logging_ops_gen",
        ":platform",
        ":string_ops",
        "//tensorflow/python/compat",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "lookup_ops",
    srcs = ["ops/lookup_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":lookup_ops_gen",
        ":math_ops",
        ":sparse_tensor",
        ":string_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "math_grad",
    srcs = ["ops/math_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":math_ops_gen",
        ":pywrap_tf_session",
        ":tensor_util",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "op_selector",
    srcs = ["ops/op_selector.py"],
    srcs_version = "PY3",
    deps = [":framework_ops"],
)

py_library(
    name = "math_ops",
    srcs = ["ops/math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":bitwise_ops_gen",
        ":common_shapes",
        ":constant_op",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":dtypes",
        ":framework_ops",
        ":graph_util",
        ":math_ops_gen",
        ":nn_ops_gen",
        ":sparse_ops_gen",
        ":sparse_tensor",
        ":state_ops",
        ":state_ops_gen",
        ":tensor_shape",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "resources",
    srcs = ["ops/resources.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/python/util:tf_should_use",
    ],
)

py_library(
    name = "resource_variable_ops",
    srcs = ["ops/resource_variable_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":auto_control_deps_utils",
        ":dtypes",
        ":framework_ops",
        ":handle_data_util",
        ":pywrap_tf_session",
        ":resource_variable_ops_gen",
        ":tensor_shape",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "critical_section_ops",
    srcs = ["ops/critical_section_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":resource_variable_ops_gen",
        ":tensor_array_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "list_ops",
    srcs = ["ops/list_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":handle_data_util",
        ":list_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "map_ops",
    srcs = ["ops/map_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":map_ops_gen",
    ],
)

py_library(
    name = "nn",
    srcs = [
        "ops/nn.py",
        "ops/nn_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops",
        ":ctc_ops",
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops",
        ":variables",
        "//tensorflow/python/platform:device_context",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "nn_grad",
    srcs = ["ops/nn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops",
        ":tensor_util",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "nn_ops",
    srcs = ["ops/nn_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":graph_util",
        ":math_ops",
        ":nn_ops_gen",
        ":platform",
        ":random_ops",
        ":tensor_shape",
        ":tensor_util",
        ":variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "numerics",
    srcs = ["ops/numerics.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "parsing_config",
    srcs = ["ops/parsing_config.py"],
    srcs_version = "PY3",
    deps = [
        ":framework",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["ops/parsing_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":parsing_config",
        ":parsing_ops_gen",
        ":sparse_ops",
    ],
)

py_library(
    name = "partitioned_variables",
    srcs = ["ops/partitioned_variables.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":platform",
        ":variable_scope",
    ],
)

py_library(
    name = "random_grad",
    srcs = ["ops/random_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":random_ops_gen",
    ],
)

py_library(
    name = "random_ops",
    srcs = ["ops/random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":random_ops_gen",
        ":random_seed",
        ":stateless_random_ops",
    ],
)

py_library(
    name = "stateful_random_ops",
    srcs = ["ops/stateful_random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":stateful_random_ops_gen",
        ":stateless_random_ops_v2_gen",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["ops/stateful_random_ops_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":client_testlib",
        ":config",
        ":logging_ops",
        ":random_ops_gen",
        ":stateful_random_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/kernel_tests/random:util",
    ],
)

py_library(
    name = "stateless_random_ops",
    srcs = ["ops/stateless_random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":stateless_random_ops_gen",
        ":stateless_random_ops_v2_gen",
    ],
)

py_library(
    name = "rnn",
    srcs = ["ops/rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":rnn_cell",
        ":tensor_array_ops",
        ":variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "rnn_cell",
    srcs = [
        "ops/rnn_cell.py",
        "ops/rnn_cell_impl.py",
        "ops/rnn_cell_wrapper_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":framework_for_generated_wrappers",
        ":init_ops",
        ":math_ops",
        ":nn_ops",
        ":partitioned_variables",
        ":random_ops",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_impl",
        "//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_wrapper_impl",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "script_ops",
    srcs = ["ops/script_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":script_ops_gen",
        "//tensorflow/python/lib/core:_pywrap_py_func",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sdca_ops",
    srcs = ["ops/sdca_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":sdca_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "session_ops",
    srcs = ["ops/session_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        ":framework_for_generated_wrappers",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "sparse_grad",
    srcs = ["ops/sparse_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
        ":sparse_ops_gen",
    ],
)

py_library(
    name = "sparse_ops",
    srcs = ["ops/sparse_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops_gen",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_ops_test",
    srcs = ["ops/sparse_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":array_grad",
        ":constant_op",
        ":dtypes",
        ":framework_test_lib",
        ":gradient_checker_v2",
        ":sparse_grad",
        ":sparse_ops",
        ":sparse_tensor",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "sort_ops",
    srcs = ["ops/sort_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework",
        ":math_ops",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sort_ops_test",
    srcs = ["ops/sort_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":random_ops",
        ":sort_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "confusion_matrix",
    srcs = ["ops/confusion_matrix.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":sparse_ops",
    ],
)

py_library(
    name = "weights_broadcast_ops",
    srcs = [
        "ops/weights_broadcast_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":framework",
        ":math_ops",
        ":sets",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "ops/metrics.py",
        "ops/metrics_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":confusion_matrix",
        ":control_flow_ops",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":sets",
        ":sparse_ops",
        ":state_ops",
        ":variable_scope",
        ":variables",
        ":weights_broadcast_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "special_math_ops",
    srcs = ["ops/special_math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":platform",
        ":special_math_ops_gen",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "@functools32_archive//:functools32",
        "@opt_einsum_archive//:opt_einsum",
    ],
)

py_library(
    name = "rnn_grad",
    srcs = ["ops/rnn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":rnn_ops_gen",
    ],
)

cuda_py_test(
    name = "rnn_grad_test",
    srcs = ["ops/rnn_grad_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":dtypes",
        ":framework_ops",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":rnn_grad",
        ":rnn_ops_gen",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "script_ops_test",
    srcs = ["ops/script_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":dtypes",
        ":framework_ops",
        ":framework_test_lib",
        ":script_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "standard_ops",
    srcs = ["ops/standard_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":batch_ops",
        ":check_ops",
        ":clip_ops",
        ":confusion_matrix",
        ":control_flow_ops",
        ":critical_section_ops",
        ":cudnn_rnn_grad",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":gradients",
        ":histogram_ops",
        ":init_ops",
        ":io_ops",
        ":linalg_ops",
        ":logging_ops",
        ":lookup_ops",
        ":manip_grad",
        ":manip_ops",
        ":math_grad",
        ":math_ops",
        ":numerics",
        ":parsing_ops",
        ":partitioned_variables",
        ":proto_ops",
        ":random_ops",
        ":rnn_grad",
        ":script_ops",
        ":session_ops",
        ":sort_ops",
        ":sparse_grad",
        ":sparse_ops",
        ":special_math_ops",
        ":state_grad",
        ":state_ops",
        ":stateful_random_ops",
        ":stateless_random_ops",
        ":string_ops",
        ":template",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/compiler",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg/sparse",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/structured",
        "//tensorflow/python/training/experimental:loss_scaling_gradient_tape",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "state_grad",
    srcs = ["ops/state_grad.py"],
    srcs_version = "PY3",
    deps = [":framework_for_generated_wrappers"],
)

py_library(
    name = "state_ops",
    srcs = ["ops/state_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_ops",
        ":math_ops_gen",
        ":resource_variable_ops_gen",
        ":state_ops_gen",
        ":tensor_shape",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["ops/string_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":string_ops_gen",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "summary_ops_v2",
    srcs = ["ops/summary_ops_v2.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":resource_variable_ops",
        ":smart_cond",
        ":summary_op_util",
        ":summary_ops_gen",
        ":tensor_util",
        ":training_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:profiler",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "template",
    srcs = ["ops/template.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":platform",
        ":variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "tensor_array_grad",
    srcs = ["ops/tensor_array_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":tensor_array_ops",
    ],
)

py_library(
    name = "tensor_array_ops",
    srcs = ["ops/tensor_array_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":constant_op",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":dtypes",
        ":errors",
        ":framework_ops",
        ":list_ops",
        ":math_ops",
        ":tensor_shape",
        ":tensor_util",
        ":tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_should_use",
    ],
)

py_library(
    name = "variable_scope",
    srcs = ["ops/variable_scope.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":dtypes",
        ":framework_ops",
        ":init_ops",
        ":platform",
        ":resource_variable_ops",
        ":tensor_shape",
        ":tf2",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "variables",
    srcs = ["ops/variables.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":dtypes",
        ":framework_ops",
        ":math_ops",
        ":state_ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_should_use",
    ],
)

py_library(
    name = "gradient_checker",
    srcs = ["ops/gradient_checker.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "gradient_checker_v2",
    srcs = ["ops/gradient_checker_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        "//third_party/py/numpy",
    ],
)

# This target is deprecated.
py_library(
    name = "ops",
    srcs = ["user_ops/user_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":user_ops_gen",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "bitwise_ops_test",
    size = "small",
    srcs = ["ops/bitwise_ops_test.py"],
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":bitwise_ops",
        ":constant_op",
        ":dtypes",
        ":framework_test_lib",
    ],
)

cuda_py_test(
    name = "control_flow_ops_test",
    size = "small",
    srcs = ["ops/control_flow_ops_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "no_cuda_asan",  # b/173241932
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":cond_v2",
        ":control_flow_ops",
        ":control_flow_v2_toggles",
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":init_ops",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":training",
        ":variable_scope",
        ":variables",
        ":while_v2",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/util",
    ],
)

py_test(
    name = "op_selector_test",
    srcs = ["ops/op_selector_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":constant_op",
        ":framework_ops",
        ":math_ops",
        ":op_selector",
    ],
)

cuda_py_test(
    name = "embedding_ops_test",
    srcs = ["ops/embedding_ops_test.py"],
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    deps = [
        ":embedding_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":resource_variable_ops",
        "//tensorflow/python/eager:def_function",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gradient_checker_v2_test",
    size = "medium",
    srcs = ["ops/gradient_checker_v2_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["ops/gradients_test.py"],
    python_version = "PY3",
    tags = ["no_oss"],  # b/118709825
    deps = [
        ":array_grad",
        ":array_ops",
        ":control_flow_grad",
        ":control_flow_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":functional_ops",
        ":gradients",
        ":init_ops",
        ":list_ops",
        ":map_ops",
        ":math_grad",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform_test",
        ":state_grad",
        ":state_ops",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":test_ops",
        ":unconnected_gradients",
        ":variable_scope",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "histogram_ops_test",
    size = "small",
    srcs = ["ops/histogram_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":histogram_ops",
        ":init_ops",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "image_grad_deterministic_test",
    size = "large",
    srcs = ["ops/image_grad_deterministic_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":image_grad_test_base",
    ],
)

cuda_py_test(
    name = "image_grad_test",
    size = "large",
    srcs = ["ops/image_grad_test.py"],
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":image_grad_test_base",
    ],
)

py_library(
    name = "image_grad_test_base",
    srcs = ["ops/image_grad_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":image_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["ops/image_ops_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "no_cuda_asan",  # TODO(b/171511582): re-enable.
    ],
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":image_ops",
        ":io_ops",
        ":math_ops",
        ":platform_test",
        ":random_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "init_ops_test",
    size = "small",
    srcs = ["ops/init_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_ops",
        ":init_ops",
        ":resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "init_ops_v2_test",
    size = "medium",
    srcs = ["ops/init_ops_v2_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_ops",
        ":init_ops_v2",
        ":random_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_grad_test",
    size = "small",
    srcs = ["ops/math_grad_test.py"],
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/python/debug:check_numerics_callback",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_ops_test",
    size = "medium",
    srcs = ["ops/math_ops_test.py"],
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    deps = [
        ":array_ops",
        ":errors",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":platform_test",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_ops_linspace_test",
    size = "medium",
    srcs = ["ops/math_ops_linspace_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_batchnorm_test",
    size = "medium",
    srcs = ["ops/nn_batchnorm_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_windows"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_fused_batchnorm_test",
    size = "large",
    srcs = ["ops/nn_fused_batchnorm_test.py"],
    python_version = "PY3",
    shard_count = 24,
    tags = ["no_rocm"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_test",
    size = "medium",
    srcs = ["ops/nn_test.py"],
    python_version = "PY3",
    tags = ["no_windows"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn",
        ":nn_grad",
        ":nn_ops",
        ":partitioned_variables",
        ":variable_scope",
        ":variables",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "nn_loss_scaling_utilities_test",
    size = "small",
    srcs = ["ops/nn_loss_scaling_utilities_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "nn_xent_test",
    size = "medium",
    srcs = ["ops/nn_xent_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["ops/tensor_array_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "sobol_ops_test",
    size = "small",
    srcs = ["ops/sobol_ops_test.py"],
    kernels = [
        "//tensorflow/core/kernels:libtfkernel_sobol_op.so",
    ],
    tags = [
        "no_oss",  # TODO(b/149565560)
        "no_windows_gpu",
    ],
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "special_math_ops_test",
    size = "medium",
    srcs = ["ops/special_math_ops_test.py"],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
        "no_windows_gpu",
        "noasan",  # b/159332048
        "nomsan",  # b/148630708
    ],
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":special_math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "variable_spec_test",
    size = "small",
    srcs = ["ops/variable_spec_test.py"],
    python_version = "PY3",
    deps = [
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform_test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "client",
    srcs = [
        "client/client_lib.py",
        "client/device_lib.py",
        "client/timeline.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_device_lib",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
        ":session",
        ":session_ops",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

# Note: this is a heavyweight library specialized for TensorFlow graphs. Do not use for
# other purposes.

py_library(
    name = "global_test_configuration",
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    deps = if_mlir(["//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_pass_registration"]) +
           tf_enable_mlir_bridge(),
)

# `tree.compat` requires visibility exception to test against `nest_test`
# to facilitate convergence between `tree.compat` and `nest`.

tf_proto_library(
    name = "protos_all",
    srcs = glob(
        ["**/*.proto"],
        exclude = [
            "//tensorflow/python/util:compare_test_proto_src",
            "framework/cpp_shape_inference.proto",
        ],
    ),
    protodeps = ["//tensorflow/python/training:checkpoint_state"],
    visibility = visibility,
)

tf_proto_library(
    name = "cpp_shape_inference_proto",
    srcs = ["framework/cpp_shape_inference.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    visibility = ["//tensorflow:internal"],
)

tf_py_test(
    name = "events_writer_test",
    size = "small",
    srcs = ["client/events_writer_test.py"],
    python_version = "PY3",
    deps = [
        ":errors",
        ":framework_test_lib",
        ":lib",
        ":platform_test",
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "device_lib",
    srcs = ["client/device_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_device_lib",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_device_lib",
    srcs = ["client/device_lib_wrapper.cc"],
    module_name = "_pywrap_device_lib",
    deps = [
        "//tensorflow/core:framework_internal_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_errors_test_helper",
    srcs = ["framework/errors_test_helper.cc"],
    module_name = "_errors_test_helper",
    deps = [
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

cuda_py_tests(
    name = "device_lib_test",
    size = "small",
    srcs = [
        "client/device_lib_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":client",
        ":client_testlib",
        ":framework_test_lib",
        ":platform_test",
        "//tensorflow/core:protos_all_py",
    ],
)

cc_library(
    name = "session_ref",
    srcs = ["client/session_ref.cc"],
    hdrs = ["client/session_ref.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:master_proto_cc",
        "//tensorflow/core/protobuf:replay_log_proto_cc",
    ],
)

tf_cuda_library(
    name = "tf_session_helper",
    srcs = ["client/tf_session_helper.cc"],
    hdrs = ["client/tf_session_helper.h"],
    compatible_with = [],
    deps = [
        ":construction_fails_op",
        ":ndarray_tensor",
        ":ndarray_tensor_bridge",
        ":numpy_lib",
        ":safe_ptr",
        ":session_ref",
        ":test_ops_kernels",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
    alwayslink = 1,
)

py_library(
    name = "pywrap_tensorflow",
    srcs = [
        "pywrap_tensorflow.py",
    ] + if_static(
        ["pywrap_dlopen_global_flags.py"],
        # Import will fail, indicating no global dlopen flags
        otherwise = [],
    ),  # b/153585257
    srcs_version = "PY3",
    deps = [":pywrap_tensorflow_internal"],
)

pywrap_tensorflow_macro(
    name = "pywrap_tensorflow_internal",
    srcs = ["pywrap_tensorflow_internal.cc"],
    # copybara:comment_begin(OSS Windows only: DEF file for exported symbols)
    win_def_file = select({
        "//tensorflow:windows": ":pywrap_tensorflow_filtered_def_file",
        "//conditions:default": None,
    }),
    # copybara:comment_end
    deps = [
        ":bfloat16_lib",
        ":cost_analyzer_lib",
        ":model_analyzer_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/python/util:cpp_nest",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:function_parameter_canonicalizer",
        "//tensorflow/python/util:kernel_registry",
        ":numpy_lib",
        ":safe_ptr",
        ":py_exception_registry",
        "//tensorflow/python/lib/core:py_func_lib",
        "//tensorflow/python/lib/io:py_record_reader_lib",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:pybind11_proto",
        ":python_api_dispatcher",
        ":python_api_info",
        ":python_api_parameter_converter",
        ":python_op_gen",
        ":python_tensor_converter",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        ":tf_session_helper",
        "//third_party/python_runtime:headers",
        "//tensorflow/c:c_api",
        "//tensorflow/c:ops",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:python_api",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/experimental/ops",
        "//tensorflow/c/experimental/gradients",
        "//tensorflow/c/experimental/gradients/tape",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core/data/service:server_lib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_memory",
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:reader_base",
        "//tensorflow/core/kernels:data_service_ops",
        "//tensorflow/core/debug",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/profiler/rpc:profiler_server_impl",
        "//tensorflow/core/profiler/rpc/client:profiler_client_impl",
        "//tensorflow/core/profiler/internal:print_model_analysis",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/core/profiler/internal/cpu:python_tracer",
        "//tensorflow/tools/graph_transforms:transform_graph_lib",
        "//tensorflow/lite/toco/python:toco_python_api",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/compiler/mlir/python:mlir",
    ] + (tf_additional_lib_deps() +
         tf_monitoring_python_deps() +
         tf_additional_plugin_deps() +
         tf_additional_profiler_deps()) + if_xla_available([
        "//tensorflow/compiler/aot:tfcompile_lib",
    ]) + if_static(extra_deps = [
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core/platform:enable_tf2_utils",
    ]),
)

# ** Targets for Windows build (start) **
# We need the following targets to expose symbols from _pywrap_tensorflow.dll

filegroup(
    name = "win_lib_files_for_exported_symbols",
    srcs = [
        ":bfloat16_lib",  # bfloat16
        ":cost_analyzer_lib",  # cost_analyzer
        "//tensorflow/python/util:cpp_nest",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:kernel_registry",
        ":model_analyzer_lib",  # model_analyzer
        ":ndarray_tensor",  # checkpoint_reader
        ":numpy_lib",  # checkpoint_reader
        ":py_exception_registry",  # py_exception_registry
        "//tensorflow/python/lib/core:py_func_lib",
        ":python_api_dispatcher",  # python_api_dispatcher
        ":python_api_info",  # python_api_info
        ":python_api_parameter_converter",  # python_api_parameter_converter
        ":python_tensor_converter",  # python_tensor_converter
        ":python_op_gen",  # python_op_gen
        ":safe_ptr",  # checkpoint_reader
        "//tensorflow/c:checkpoint_reader",  # checkpoint_reader
        "//tensorflow/c:python_api",  # tf_session
        "//tensorflow/c:tf_status_helper",  # tfe
        "//tensorflow/compiler/jit:get_compiler_ir",  #tfe
        "//tensorflow/compiler/jit:flags",  #tfe
        "//tensorflow/compiler/mlir/python:mlir",  # mlir
        "//tensorflow/core/common_runtime:graph_constructor",  # tf_session
        "//tensorflow/core/common_runtime:quantize_training",  # quantize_training
        "//tensorflow/core/common_runtime:session_options",  # device_lib, tfe, tf_session
        "//tensorflow/core/common_runtime:session_state",  # tf_session
        "//tensorflow/core/data/service:server_lib",  # server_lib
        "//tensorflow/core:framework_internal_impl",  # op_def_registry
        "//tensorflow/core:lib_internal_impl",  # device_lib
        "//tensorflow/core:op_gen_lib",  # tf_session
        "//tensorflow/core/common_runtime/eager:context",  # tfe
        "//tensorflow/core/common_runtime/eager:tensor_handle",  # tfe
        "//tensorflow/core/common_runtime/eager:eager_executor",  # tfe
        "//tensorflow/core/grappler:devices",  # tf_cluster
        "//tensorflow/core/grappler:grappler_item",  # tf_item
        "//tensorflow/core/grappler:grappler_item_builder",  # tf_item
        "//tensorflow/core/grappler/clusters:cluster",  # tf_cluster
        "//tensorflow/core/grappler/clusters:single_machine",  # tf_cluster
        "//tensorflow/core/grappler/clusters:utils",  # tf_optimizer
        "//tensorflow/core/grappler/clusters:virtual_cluster",  # tf_cluster
        "//tensorflow/core/grappler/costs:analytical_cost_estimator",  # cost analyzer
        "//tensorflow/core/grappler/costs:graph_memory",  # tf_cluster
        "//tensorflow/core/grappler/costs:graph_properties",  # tf_item
        "//tensorflow/core/grappler/costs:measuring_cost_estimator",  # tf_cluster
        "//tensorflow/core/grappler/costs:op_level_cost_estimator",  # tf_cluster
        "//tensorflow/core/grappler/costs:utils",  # tf_cluster
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",  # graph_analyzer
        "//tensorflow/core/grappler/optimizers:meta_optimizer",  # tf_optimizer
        "//tensorflow/core/grappler/utils:topological_sort",  # tf_item
        "//tensorflow/core/platform:tensor_float_32_utils",  # tensor_float_32
        "//tensorflow/core/profiler/internal:print_model_analysis",  # tfprof
        "//tensorflow/core/profiler/internal/cpu:traceme_recorder_impl",  # profiler
        "//tensorflow/core/profiler/lib:profiler_session_impl",  # profiler
        "//tensorflow/core/profiler/rpc:profiler_server_impl",  # profiler
        "//tensorflow/core/profiler/rpc/client:profiler_client_impl",  # profiler
        "//tensorflow/core/util:port",  # util_port
        "//tensorflow/core/util/tensor_bundle",  # checkpoint_reader
        "//tensorflow/lite/toco/python:toco_python_api",  # toco
        ":tf_session_helper",  # tf_session
        "//tensorflow/python/eager:pywrap_tfe_lib",  # pywrap_tfe_lib
        "//tensorflow/stream_executor:stream_executor_pimpl",  # stat_summarizer
        "//tensorflow/tools/graph_transforms:transform_graph_lib",  # transform_graph
    ] + if_xla_available([
        "//tensorflow/compiler/aot:tfcompile_lib",  # tfcompile
        "//tensorflow/compiler/xla:status_macros",  # tfcompile
        "//tensorflow/compiler/xla/service:hlo",  # tfcompile
    ]),
    visibility = ["//visibility:private"],
)

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
genrule(
    name = "pywrap_tensorflow_filtered_def_file",
    srcs = select({
        "//tensorflow:windows": [
            ":pybind_symbol_target_libs_file",
            ":win_lib_files_for_exported_symbols",
            "//tensorflow:tensorflow_def_file",
            "//tensorflow/tools/def_file_filter:symbols_pybind",
        ],
        "//conditions:default": [],
    }),
    outs = ["pywrap_tensorflow_filtered_def_file.def"],
    cmd = select({
        "//tensorflow:windows": """
              $(location @local_config_def_file_filter//:def_file_filter) \\
              --input $(location //tensorflow:tensorflow_def_file) \\
              --output $@ \\
              --target _pywrap_tensorflow_internal.pyd \\
              --symbols $(location //tensorflow/tools/def_file_filter:symbols_pybind) \\
              --lib_paths_file $(location :pybind_symbol_target_libs_file)
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    tools = ["@local_config_def_file_filter//:def_file_filter"],
    visibility = ["//visibility:public"],
)

# Write to a file a list of all cc_library targets that we need for exporting symbols on Windows.
genrule(
    name = "pybind_symbol_target_libs_file",
    srcs = [":win_lib_files_for_exported_symbols"],
    outs = ["pybind_symbol_target_libs_file.txt"],
    cmd = select({
        "//tensorflow:windows": """
            for SRC in $(SRCS); do
              echo $$SRC | sed 's/third_party\\///g' >> $@
            done
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# Get the import library of _pywrap_tensorflow_internal.pyd
filegroup(
    name = "get_pywrap_tensorflow_import_lib_file",
    srcs = [":_pywrap_tensorflow_internal.so"],
    output_group = "interface_library",
)

cc_import(
    name = "_pywrap_tensorflow_internal_linux",
    shared_library = "//tensorflow/python:lib_pywrap_tensorflow_internal.so",
    visibility = tf_external_workspace_visible(visibility),
)

cc_import(
    name = "_pywrap_tensorflow_internal_macos",
    shared_library = "//tensorflow/python:lib_pywrap_tensorflow_internal.dylib",
    visibility = tf_external_workspace_visible(visibility),
)

cc_import(
    name = "_pywrap_tensorflow_internal_windows",
    interface_library = "//tensorflow/python:pywrap_tensorflow_import_lib_file",
    shared_library = "//tensorflow/python:_pywrap_tensorflow_internal.dll",
    visibility = tf_external_workspace_visible(visibility),
)

# Rename the import library for _pywrap_tensorflow_internal.pyd to _pywrap_tensorflow_internal.lib
# (It was _pywrap_tensorflow_internal.so.if.lib).
genrule(
    name = "pywrap_tensorflow_import_lib_file",
    srcs = [":get_pywrap_tensorflow_import_lib_file"],
    outs = ["_pywrap_tensorflow_internal.lib"],
    cmd = select({
        "//tensorflow:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# Create a cc_import rule for the import library of _pywrap_tensorflow_internal.dll
# so that custom ops' dynamic libraries can link against it.
cc_import(
    name = "pywrap_tensorflow_import_lib",
    interface_library = select({
        "//tensorflow:windows": ":pywrap_tensorflow_import_lib_file",
        "//conditions:default": "not_existing_on_unix.lib",  # Just a placeholder for Unix platforms
    }),
    system_provided = 1,
)

# ** Targets for Windows build (end) **

py_library(
    name = "session",
    srcs = ["client/session.py"],
    srcs_version = "PY3",
    deps = [
        ":c_api_util",
        ":error_interpolation",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":mixed_precision_global_state",
        ":platform",
        ":pywrap_tensorflow",
        ":session_ops",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@wrapt",
    ],
)

py_library(
    name = "timeline",
    srcs = ["client/timeline.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
    ],
)

# Just used by tests.
tf_cuda_library(
    name = "construction_fails_op",
    srcs = ["client/test_construction_fails_op.cc"],
    deps = [
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_py_test(
    name = "session_test",
    size = "medium",
    srcs = ["client/session_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/127001953
        "no_pip_gpu",  # testInteractivePlacePrunedGraph fails on invalid assumption about GPU ops.
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client",
        ":config",
        ":control_flow_ops",
        ":data_flow_ops",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":training",
        ":variables",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "session_clusterspec_prop_test",
    size = "small",
    srcs = ["client/session_clusterspec_prop_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_gpu",
        "no_oss",
        "no_pip",
        "no_pip_gpu",
        "notap",
    ],
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":state_ops",
        ":training",
        ":variables",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "session_list_devices_test",
    size = "small",
    srcs = ["client/session_list_devices_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_gpu",
        "no_pip_gpu",
        "notsan",  # data race due to b/62910646
    ],
    deps = [
        ":client",
        ":framework",
        ":framework_test_lib",
        ":platform_test",
        ":training",
    ],
)

tf_py_test(
    name = "session_partial_run_test",
    size = "small",
    srcs = ["client/session_partial_run_test.py"],
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_gpu",
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client",
        ":errors",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform_test",
        ":training",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "timeline_test",
    size = "small",
    srcs = ["client/timeline_test.py"],
    python_version = "PY3",
    tags = [
        "gpu_cupti",
        "no_gpu",  # b/154742661
    ],
    xla_enable_strict_auto_jit = False,  # Graph structure is different with autojit
    deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
    ],
)

cuda_py_test(
    name = "virtual_gpu_test",
    size = "small",
    srcs = ["client/virtual_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "no_gpu",  # b/127386241
        "no_windows_gpu",
    ],
    deps = [
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "c_api_util_test",
    size = "small",
    srcs = ["framework/c_api_util_test.py"],
    python_version = "PY3",
    deps = [
        ":c_api_util",
        ":framework_test_lib",
        ":platform_test",
    ],
)

tf_py_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["framework/graph_util_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":client_testlib",
        ":control_flow_v2_toggles",
        ":framework",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":state_ops_gen",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "convert_to_constants_test",
    size = "small",
    srcs = ["framework/convert_to_constants_test.py"],
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":client_testlib",
        ":control_flow_v2_toggles",
        ":convert_to_constants",
        ":framework_test_lib",
        ":math_ops",
    ],
)

py_library(
    name = "summary_op_util",
    srcs = ["ops/summary_op_util.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":framework",
        ":framework_for_generated_wrappers",
        ":platform",
    ],
)

py_library(
    name = "summary",
    srcs = glob(
        ["summary/**/*.py"],
        exclude = [
            "**/fake*",
            "**/*test*",
        ],
    ),
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":client",
        ":constant_op",
        ":framework",
        ":framework_for_generated_wrappers",
        ":lib",
        ":logging_ops_gen",
        ":platform",
        ":protos_all_py",
        ":pywrap_tensorflow",
        ":summary_op_util",
        ":summary_ops_gen",
        ":summary_ops_v2",
        "//tensorflow/python/distribute:summary_op_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "fake_summary_writer",
    testonly = 1,
    srcs = ["summary/writer/fake_summary_writer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":framework_test_lib",
        ":protos_all_py",
        ":summary",
    ],
)

py_tests(
    name = "summary_tests",
    size = "small",
    srcs = [
        "summary/plugin_asset_test.py",
        "summary/summary_iterator_test.py",
        "summary/summary_test.py",
        "summary/writer/writer_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":platform",
        ":platform_test",
        ":summary",
        ":summary_ops_v2",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

# TODO(scottzhu): Move all the tf.layer related targets.
py_library(
    name = "layers_base",
    srcs = [
        "layers/__init__.py",
        "layers/base.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/keras/legacy_tf_layers:layers_base",
    ],
)

py_library(
    name = "layers_util",
    srcs = [
        "layers/utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":smart_cond",
        ":variables",
    ],
)

py_library(
    name = "layers",
    srcs = [
        "layers/convolutional.py",
        "layers/core.py",
        "layers/layers.py",
        "layers/normalization.py",
        "layers/pooling.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//tensorflow/python/keras/legacy_tf_layers:convolutional",
        "//tensorflow/python/keras/legacy_tf_layers:core",
        "//tensorflow/python/keras/legacy_tf_layers:normalization",
        "//tensorflow/python/keras/legacy_tf_layers:pooling",
    ],
)

# -----------------------------------------------------------------------------
# Quantization

tf_py_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["ops/dequantize_op_test.py"],
    python_version = "PY3",
    tags = [
        "no_tfrt",  # TODO(b/169901260)
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "quantized_ops_test",
    size = "small",
    srcs = ["ops/quantized_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_tfrt",  # TODO(b/169901260)
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["ops/quantized_conv_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_tfrt",  # TODO(b/169901260)
        "no_windows",
    ],
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "ops/array_ops_test",
    srcs = ["ops/array_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":constant_op",
        ":dtypes",
        ":framework_ops",
        ":framework_test_lib",
        ":gradients",
        ":math_ops",
        ":random_ops",
    ],
)

cuda_py_test(
    name = "accumulate_n_benchmark",
    size = "medium",
    srcs = ["ops/accumulate_n_benchmark.py"],
    main = "ops/accumulate_n_benchmark.py",
    python_version = "PY3",
    shard_count = 6,
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops_gen",
        ":data_flow_ops",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":random_ops",
        ":state_ops",
        ":state_ops_gen",
    ],
)

cuda_py_test(
    name = "batch_norm_benchmark",
    srcs = ["ops/batch_norm_benchmark.py"],
    main = "ops/batch_norm_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        ":platform",
        ":random_ops",
        ":variables",
    ],
)

cuda_py_test(
    name = "collective_ops_benchmark",
    srcs = ["ops/collective_ops_benchmark.py"],
    main = "ops/collective_ops_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":collective_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

cuda_py_test(
    name = "concat_benchmark",
    srcs = ["ops/concat_benchmark.py"],
    main = "ops/concat_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":gradients",
        ":platform",
        ":variables",
        "//tensorflow/core:protos_all_py",
    ],
)

cuda_py_test(
    name = "control_flow_ops_benchmark",
    srcs = ["ops/control_flow_ops_benchmark.py"],
    main = "ops/control_flow_ops_benchmark.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":constant_op",
        ":control_flow_ops",
        ":framework_ops",
        "//tensorflow/python/eager:function",
    ],
)

cuda_py_test(
    name = "conv2d_benchmark",
    size = "large",
    srcs = ["ops/conv2d_benchmark.py"],
    main = "ops/conv2d_benchmark.py",
    python_version = "PY3",
    deps = [
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":nn_ops",
        ":platform",
        ":platform_benchmark",
        ":random_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "split_benchmark",
    srcs = ["ops/split_benchmark.py"],
    main = "ops/split_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "transpose_benchmark",
    size = "medium",
    srcs = ["ops/transpose_benchmark.py"],
    main = "ops/transpose_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":platform",
        ":platform_benchmark",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "matmul_benchmark",
    size = "medium",
    srcs = ["ops/matmul_benchmark.py"],
    main = "ops/matmul_benchmark.py",
    python_version = "PY3",
    deps = [":matmul_benchmark_main_lib"],
)

py_library(
    name = "matmul_benchmark_main_lib",
    testonly = True,
    srcs = ["ops/matmul_benchmark.py"],
    srcs_version = "PY3",
    deps = [
        ":client",
        ":client_testlib",
        ":control_flow_ops",
        ":framework_for_generated_wrappers",
        ":framework_test_lib",
        ":math_ops",
        ":platform",
        ":platform_benchmark",
        ":random_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "session_benchmark",
    srcs = ["client/session_benchmark.py"],
    grpc_enabled = True,
    main = "client/session_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":random_ops",
        ":training",
        ":variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "graph_building_benchmark",
    size = "medium",
    srcs = ["framework/graph_building_benchmark.py"],
    main = "framework/graph_building_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":dtypes",
        ":math_ops",
        ":platform_benchmark",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "nn_grad_test",
    size = "medium",
    srcs = ["ops/nn_grad_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":nn_grad",
        ":nn_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tf_item",
    srcs = [
        "grappler/item.py",
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
    srcs = ["grappler/item_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/grappler/utils:pywrap_required_hdrs",
    ],
    module_name = "_pywrap_tf_item",
    deps = [
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core:protos_all_cc",
    ] + if_not_windows(["//tensorflow/core/grappler/costs:graph_properties"]),  # b/148556093,
)

tf_py_test(
    name = "item_test",
    size = "small",
    srcs = [
        "grappler/item_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "datasets_test",
    size = "small",
    srcs = [
        "grappler/datasets_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_combinations",
        ":framework_for_generated_wrappers",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data",
    ],
)

py_library(
    name = "tf_cluster",
    srcs = [
        "grappler/cluster.py",
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
    srcs = ["grappler/cluster_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/grappler/utils:pywrap_required_hdrs",
    ],
    module_name = "_pywrap_tf_cluster",
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
        "grappler/cluster_test.py",
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
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":tf_cluster",
        ":tf_item",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "tf_optimizer",
    srcs = [
        "grappler/tf_optimizer.py",
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
    srcs = ["grappler/tf_optimizer_wrapper.cc"],
    hdrs = [
        "//tensorflow/cc:pywrap_required_hdrs",
        "//tensorflow/core/grappler:pywrap_required_hdrs",
        "//tensorflow/core/grappler/clusters:pywrap_required_hdrs",
        "//tensorflow/core/grappler/costs:pywrap_required_hdrs",
        "//tensorflow/core/grappler/optimizers:pywrap_required_hdrs",
        "//tensorflow/core/grappler/verifiers:pywrap_required_hdrs",
    ],
    module_name = "_pywrap_tf_optimizer",
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
        "grappler/tf_optimizer_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":tf_item",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "memory_optimizer_test",
    size = "medium",
    srcs = [
        "grappler/memory_optimizer_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
    ],
    deps = [
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":random_seed",
        ":session",
        ":tf_optimizer",
        ":training",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "constant_folding_test",
    size = "medium",
    srcs = [
        "grappler/constant_folding_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_tfrt",  #TODO(b/174712583)
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":dtypes",
        ":framework_for_generated_wrappers",
        ":functional_ops",
        ":math_ops",
        ":ops",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "arithmetic_optimizer_test",
    size = "small",
    srcs = [
        "grappler/arithmetic_optimizer_test.py",
    ],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_tfrt",  #TODO(b/174712583)
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

# TODO(b/131764887) Remove once LayoutOptimizer is swapped out with GenericLayoutOptimizer.
#
# cuda_py_test(
#     name = "layout_optimizer_test",
#     size = "medium",
#     srcs = [
#         "grappler/layout_optimizer_test.py",
#     ],
#     deps = [
#         ":client_testlib",
#         ":framework_for_generated_wrappers",
#         ":array_ops",
#         ":constant_op",
#         ":dtypes",
#         ":functional_ops",
#         ":math_ops",
#         ":nn",
#         ":ops",
#         ":random_ops",
#         ":state_ops",
#         ":tf_cluster",
#         ":tf_optimizer",
#         ":training",
#         "//third_party/py/numpy",
#         "//tensorflow/core:protos_all_py",
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
        "grappler/cost_analyzer.py",
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
        "grappler/cost_analyzer_tool.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cost_analyzer",
        ":framework_for_generated_wrappers",
        ":tf_optimizer",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "cost_analyzer_test",
    size = "small",
    srcs = ["grappler/cost_analyzer_test.py"],
    python_version = "PY3",
    tags = [
        "grappler",
        "no_cuda_on_cpu_tap",
        "no_mac",
        "no_pip",
        "no_windows",  # TODO(b/151942037)
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":cost_analyzer",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":random_ops",
        ":state_ops",
        ":training",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "model_analyzer",
    srcs = [
        "grappler/model_analyzer.py",
    ],
    srcs_version = "PY3",
    deps = [":_pywrap_model_analyzer"],
)

tf_py_test(
    name = "model_analyzer_test",
    size = "small",
    srcs = ["grappler/model_analyzer_test.py"],
    tags = [
        "grappler",
        "no_pip",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":model_analyzer",
        ":state_ops",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "auto_mixed_precision_test",
    size = "medium",
    srcs = [
        "grappler/auto_mixed_precision_test.py",
    ],
    python_version = "PY3",
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":array_ops",
        ":client_testlib",
        ":constant_op",
        ":control_flow_ops",
        ":dtypes",
        ":framework_for_generated_wrappers",
        ":math_ops",
        ":nn",
        ":ops",
        ":random_ops",
        ":training",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "nccl_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:nccl_ops_op_lib",
    ],
)

py_library(
    name = "nccl_ops",
    srcs = ["ops/nccl_ops.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//learning/deepmind/tensorflow:__subpackages__",
        "//third_party/car/deep_nets/tensorflow:__subpackages__",
    ],
    deps = [
        ":framework_for_generated_wrappers",
        ":nccl_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
    ],
)

cuda_py_test(
    name = "nccl_ops_test",
    size = "small",
    srcs = ["ops/nccl_ops_test.py"],
    python_version = "PY3",
    # Disabled on jenkins until errors finding nvmlShutdown are found.
    tags = [
        "manual",
        "multi_gpu",
        "no_oss",
        "noguitar",
        "notap",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":framework_test_lib",
        ":nccl_ops",
        ":platform_test",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "decode_proto_ops_gen",
    deps = [
        "//tensorflow/core:decode_proto_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "encode_proto_ops_gen",
    deps = [
        "//tensorflow/core:encode_proto_ops_op_lib",
    ],
)

tf_python_pybind_extension(
    name = "_proto_comparators",
    srcs = ["framework/proto_comparators.cc"],
    module_name = "_proto_comparators",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_library(
    name = "proto_ops",
    srcs = ["ops/proto_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":decode_proto_ops_gen",
        ":encode_proto_ops_gen",
        ":framework_ops",
    ],
)

py_library(
    name = "pywrap_mlir",
    srcs = ["pywrap_mlir.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_mlir",
        ":pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_mlir",
    srcs = ["mlir_wrapper.cc"],
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/compiler/mlir/python:pywrap_mlir_hdrs",
        "//tensorflow/python/lib/core:safe_ptr_hdr",
    ],
    module_name = "_pywrap_mlir",
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:fixed_array",
        "@pybind11",
    ],
)

cc_library(
    name = "unified_api_pywrap_required_headers",
    textual_hdrs = [
        "//tensorflow/python/lib/core:basic_hdrs",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/c/experimental/gradients:pywrap_required_hdrs",
        "//tensorflow/c/experimental/gradients/tape:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime:core_cpu_lib_headers",
        "//tensorflow/core/public:session.h",
        "//tensorflow/core/public:session_options.h",
    ],
    visibility = ["//tensorflow/python/framework/experimental:__pkg__"],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

py_library(
    name = "pywrap_tfe",
    srcs = ["pywrap_tfe.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe",
        ":pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfe",
    srcs = ["tfe_wrapper.cc"],
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:numpy_hdr",
        "//tensorflow/python/lib/core:py_exception_registry_hdr",
        "//tensorflow/python/lib/core:safe_ptr_hdr",
        "//tensorflow/python/util:util_hdr",
    ],
    module_name = "_pywrap_tfe",
    # Only include TensorFlow header-only targets here.
    # If a cc_library needs to depend on TensorFlow .cc files through srcs or
    # deps, then you can use cc_header_only_library to keep only headers.
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/py/numpy:headers",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/c:pywrap_required_hdrs",
        "@pybind11",
        "//third_party/python_runtime:headers",
        "//tensorflow/compiler/jit:flags_headers_only",
        "//tensorflow/compiler/jit:get_compiler_ir_hdrs_only",
        "//tensorflow/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform",
        "//tensorflow/core/lib/llvm_rtti",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_pywrap_graph_analyzer",
    srcs = ["grappler/graph_analyzer_tool_wrapper.cc"],
    module_name = "_pywrap_graph_analyzer",
    deps = [
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
        "@pybind11",
    ],
)

py_binary(
    name = "graph_analyzer",
    srcs = [
        "grappler/graph_analyzer.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":_pywrap_graph_analyzer",
        ":framework_for_generated_wrappers",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_parallel_device",
    srcs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        "//tensorflow/c/eager/parallel_device:headers",
        "//tensorflow/c/eager/parallel_device:sources",
        "//tensorflow/python/distribute/parallel_device:pywrap_parallel_device.cc",
        "//tensorflow/python/lib/core:safe_ptr_hdr",
    ],
    module_name = "_pywrap_parallel_device",
    visibility = ["//tensorflow/python/distribute/parallel_device:__pkg__"],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/c/eager:tfe_cancellationmanager_internal_hdrs_only",
        "//tensorflow/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

pyx_library(
    name = "framework_fast_tensor_util",
    srcs = ["framework/fast_tensor_util.pyx"],
    py_deps = ["//tensorflow/python:util"],
    deps = ["//third_party/py/numpy:headers"],
)

py_library(
    name = "tf2",
    srcs = ["tf2.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "tf2_test",
    srcs = ["framework/tf2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":framework_combinations",
        ":tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

cuda_py_test(
    name = "raw_ops_test",
    srcs = ["ops/raw_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
    ],
)

alias(
    name = "platform_benchmark",
    actual = "//tensorflow/python/platform:benchmark",
)

alias(
    name = "platform_analytics",
    actual = "//tensorflow/python/platform:analytics",
)

py_library(
    name = "platform_test",
    srcs_version = "PY3",
    deps = ["//tensorflow/python/platform:test"],
)

alias(
    name = "platform",
    actual = "//tensorflow/python/platform:platform",
    visibility = ["//visibility:public"],
)

alias(
    name = "client_testlib",
    actual = "//tensorflow/python/platform:client_testlib",
)

alias(
    name = "pybind11_absl",
    actual = "//tensorflow/python/lib/core:pybind11_absl",
)

alias(
    name = "pybind11_proto",
    actual = "//tensorflow/python/lib/core:pybind11_proto",
)

alias(
    name = "py_func_lib",
    actual = "//tensorflow/python/lib/core:py_func_lib",
)

alias(
    name = "py_seq_tensor",
    actual = "//tensorflow/python/lib/core:py_seq_tensor",
)

alias(
    name = "py_util",
    actual = "//tensorflow/python/lib/core:py_util",
)

alias(
    name = "py_record_reader_lib",
    actual = "//tensorflow/python/lib/io:py_record_reader_lib",
)

alias(
    name = "numpy_lib",
    actual = "//tensorflow/python/lib/core:numpy_lib",
)

alias(
    name = "py_exception_registry",
    actual = "//tensorflow/python/lib/core:py_exception_registry",
)

alias(
    name = "pybind11_lib",
    actual = "//tensorflow/python/lib/core:pybind11_lib",
)

alias(
    name = "pybind11_status_headers",
    actual = "//tensorflow/python/lib/core:pybind11_status_headers",
)

alias(
    name = "pybind11_status",
    actual = "//tensorflow/python/lib/core:pybind11_status",
)

alias(
    name = "lib",
    actual = "//tensorflow/python/lib/io:lib",
)

alias(
    name = "safe_ptr",
    actual = "//tensorflow/python/lib/core:safe_ptr",
)

alias(
    name = "ndarray_tensor",
    actual = "//tensorflow/python/lib/core:ndarray_tensor",
)

alias(
    name = "ndarray_tensor_bridge",
    actual = "//tensorflow/python/lib/core:ndarray_tensor_bridge",
)

alias(
    name = "ndarray_tensor_headers",
    actual = "//tensorflow/python/lib/core:ndarray_tensor_headers",
)

alias(
    name = "basic_session_run_hooks",
    actual = "//tensorflow/python/training:basic_session_run_hooks",
)

alias(
    name = "checkpoint_management",
    actual = "//tensorflow/python/training:checkpoint_management",
)

alias(
    name = "distribute",
    actual = "//tensorflow/python/training:distribute",
)

alias(
    name = "py_checkpoint_reader",
    actual = "//tensorflow/python/training:py_checkpoint_reader",
)

alias(
    name = "saver",
    actual = "//tensorflow/python/training:saver",
)

alias(
    name = "session_run_hook",
    actual = "//tensorflow/python/training:session_run_hook",
)

alias(
    name = "training",
    actual = "//tensorflow/python/training:training",
)

alias(
    name = "training_lib",
    actual = "//tensorflow/python/training:training_lib",
)

alias(
    name = "training_server_lib",
    actual = "//tensorflow/python/training:server_lib",
)

alias(
    name = "training_util",
    actual = "//tensorflow/python/training:training_util",
)

alias(
    name = "loss_scale",
    actual = "//tensorflow/python/training/experimental:loss_scale",
)

alias(
    name = "loss_scale_optimizer",
    actual = "//tensorflow/python/training/experimental:loss_scale_optimizer",
)

alias(
    name = "mixed_precision",
    actual = "//tensorflow/python/training/experimental:mixed_precision",
)

alias(
    name = "mixed_precision_global_state",
    actual = "//tensorflow/python/training/experimental:mixed_precision_global_state",
)

alias(
    name = "loss_scaling_gradient_tape",
    actual = "//tensorflow/python/training/experimental:loss_scaling_gradient_tape",
)

py_library(
    name = "learning_rate_decay",
    # This rule depends on a target that only python:__pkg__ has visibility for.
    srcs = ["//tensorflow/python/training:learning_rate_decay.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/keras/optimizer_v2:legacy_learning_rate_decay"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "protos_all_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     deps = [":protos_all"],
# )
#
# py_proto_library(
#     name = "cpp_shape_inference_proto_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     deps = [":cpp_shape_inference_proto"],
# )
# copybara:uncomment_end
