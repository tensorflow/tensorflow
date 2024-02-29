#
# Public targets:
#  ":platform" - Low-level and platform-specific Python code.

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//rules_pywrap:pywrap.bzl", "pybind_extension", "pywrap_library", "pywrap_common_library")

# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.bzl", "VERSION", "cc_header_only_library", "clean_dep", "if_google", "if_mlir", "if_oss", "if_windows", "if_xla_available", "tf_enable_mlir_bridge", "tsl_async_value_deps")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pywrap_tensorflow_macro", "tf_external_workspace_visible", "tf_monitoring_python_deps", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_binary_deps",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_additional_plugin_deps",
    "tf_additional_profiler_deps",
)
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

# TODO(mdan): Break into per-directory files.

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//tensorflow:internal",
    "//tensorflow/dtensor:dtensor-internal",
    "//tensorflow/lite/toco/python:__pkg__",
    "//tensorflow_models:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/courier:__subpackages__",
    "//third_party/py/courier:__subpackages__",
    "//third_party/py/lingvo:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/tensorfn:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/tensorflow_docs:__subpackages__",
    "//third_party/py/tensorflow_examples:__subpackages__",
    "//third_party/py/tensorflow_recommenders:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/tf_keras:__subpackages__",
    "//third_party/py/starcraft2:__subpackages__",
    "//third_party/py/tensorflow_gnn:__subpackages__",
    "//third_party/py/tensorflow_numerics:__subpackages__",
    "//third_party/py/tensorflow_privacy:__subpackages__",
    "//third_party/reverb:__subpackages__",
    "//tensorflow_minigo:__subpackages__",
    "//research/graph/fairness/inproc_fair_reg:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

# Description:

py_strict_library(
    name = "python",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/compiler/aot/tests:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/lite/toco/python:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python/debug:__subpackages__",
        "//tensorflow/python/tools:__pkg__",  # TODO(b/34059704): remove when fixed
        "//tensorflow/tools/quantization:__pkg__",  # TODO(b/34059704): remove when fixed
    ],
    deps = [
        ":no_contrib",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/ops:gradient_checker_v2",
        "//tensorflow/python/ops:stateful_random_ops",
        "//tensorflow/python/ops/structured:structured_ops",
        "//tensorflow/python/tpu:tpu_noestimator",
    ],
)

# Some of the dependencies in keras_lib are possibly outdated or not generally
# useful, so a future task is to determine a good set of dependencies.
# keras_lib may be removed in the future, it should NOT be used as
# a direct dependency.
# TODO(b/199420795) investigate minimal set of dependencies ...
py_strict_library(
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
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/keras",
        "//tensorflow/python/layers",
        "//tensorflow/python/ops:rnn",
    ],
)

py_strict_library(
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
        "//third_party/py/tensorflow_core:__subpackages__",
    ],
    deps = [
        ":_pywrap_py_exception_registry",
        ":_pywrap_quantize_training",
        ":distributed_framework_test_lib",
        ":keras_lib",
        ":proto_exports",
        ":pywrap_tensorflow",
        ":pywrap_tfe",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/dtensor/python:dtensor",
        "//tensorflow/python/autograph",
        "//tensorflow/python/autograph/utils:testing",
        "//tensorflow/python/client",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/client:timeline",
        "//tensorflow/python/compat",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/compiler/mlir",
        "//tensorflow/python/compiler/tensorrt:init_py",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/data",
        "//tensorflow/python/debug:debug_py",
        "//tensorflow/python/distribute",
        "//tensorflow/python/distribute:combinations",  # For tf.__internal__ API.
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:estimator_training",
        "//tensorflow/python/distribute:strategy_combinations",  # For tf.__internal__,
        "//tensorflow/python/distribute/experimental/rpc:rpc_ops",
        "//tensorflow/python/dlpack",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:profiler",
        "//tensorflow/python/eager:profiler_client",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:_pywrap_python_op_gen",
        "//tensorflow/python/framework:_test_metrics_util",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:framework_lib",
        "//tensorflow/python/framework:graph_io",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:load_library",
        "//tensorflow/python/framework:meta_graph",
        "//tensorflow/python/framework:subscribe",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_ops",  # TODO(b/183988750): Break testing code out into separate rule.
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/grappler:tf_cluster",
        "//tensorflow/python/grappler:tf_item",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:audio_ops_gen",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:bitwise_ops",
        "//tensorflow/python/ops:boosted_trees_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:clustering_ops",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:composite_tensor_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:cond_v2",
        "//tensorflow/python/ops:confusion_matrix",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_case",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_switch_case",
        "//tensorflow/python/ops:cudnn_rnn_ops_gen",
        "//tensorflow/python/ops:filesystem_ops",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/ops:gradient_checker_v2",
        "//tensorflow/python/ops:histogram_ops",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:initializers_ns",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/ops:list_ops",
        "//tensorflow/python/ops:manip_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:map_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:nccl_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:proto_ops",
        "//tensorflow/python/ops:random_crop_ops",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:rnn_cell",
        "//tensorflow/python/ops:rnn_ops_gen",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:sendrecv_ops_gen",
        "//tensorflow/python/ops:session_ops",
        "//tensorflow/python/ops:sets",
        "//tensorflow/python/ops:sets_impl",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:standard_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:stateful_random_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:uniform_quant_ops_gen",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:weak_tensor_ops",
        "//tensorflow/python/ops:weak_tensor_test_util",
        "//tensorflow/python/ops:weights_broadcast_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops:while_v2",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/distributions:bijector_test_util",
        "//tensorflow/python/ops/distributions:identity_bijector",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg:linear_operator_test_util",
        "//tensorflow/python/ops/linalg/sparse:sparse_py",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:np_random",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/ops/parallel_for",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python/ops/signal:dct_ops",
        "//tensorflow/python/ops/signal:fft_ops",
        "//tensorflow/python/ops/signal:mel_ops",
        "//tensorflow/python/ops/signal:mfcc_ops",
        "//tensorflow/python/ops/signal:reconstruction_ops",
        "//tensorflow/python/ops/signal:shape_ops",
        "//tensorflow/python/ops/signal:spectral_ops",
        "//tensorflow/python/ops/signal:util_ops",
        "//tensorflow/python/ops/signal:window_ops",
        "//tensorflow/python/ops/structured:structured_ops",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:app",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler",
        "//tensorflow/python/profiler:profiler_client",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/tools:module_util",
        "//tensorflow/python/tools/api/generator:create_python_api",
        "//tensorflow/python/tpu:datasets",
        "//tensorflow/python/tpu:functional",
        "//tensorflow/python/tpu:preempted_hook_py",
        "//tensorflow/python/tpu:tpu_noestimator",
        "//tensorflow/python/training:evaluation",
        "//tensorflow/python/training:quantize_training",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:saver_test_utils",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:data",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/types:trace",
        "//tensorflow/python/user_ops:ops",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:all_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:module_wrapper",
        "//tensorflow/python/util:tf_decorator_export",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "core",
    visibility = [
        "//tensorflow:__pkg__",
    ],
    deps = [
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:core",
    ],
)

# This target should only be used for API generation.
py_strict_library(
    name = "modules_with_exports",
    srcs = ["modules_with_exports.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//tensorflow/tools/compatibility/update:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
    ],
    deps = [
        ":no_contrib",
        ":proto_exports",
        ":pywrap_tensorflow",
        ":tf2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/client",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/client:timeline",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/compiler/mlir",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/data",
        "//tensorflow/python/debug/lib:check_numerics_callback",
        "//tensorflow/python/debug/lib:dumping_callback",
        "//tensorflow/python/distribute",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:merge_call_interim",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/distribute/experimental/rpc:rpc_ops",
        "//tensorflow/python/distribute/failure_handling:failure_handling_lib",
        "//tensorflow/python/distribute/failure_handling:preemption_watcher",
        "//tensorflow/python/dlpack",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:framework_lib",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_combinations_lib",
        "//tensorflow/python/framework:versions",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:audio_ops_gen",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:bitwise_ops",
        "//tensorflow/python/ops:boosted_trees_ops_gen",
        "//tensorflow/python/ops:clustering_ops_gen",
        "//tensorflow/python/ops:composite_tensor_ops",
        "//tensorflow/python/ops:cond_v2",
        "//tensorflow/python/ops:cudnn_rnn_ops_gen",
        "//tensorflow/python/ops:debug_ops_gen",
        "//tensorflow/python/ops:filesystem_ops_gen",
        "//tensorflow/python/ops:gradient_checker_v2",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:initializers_ns",
        "//tensorflow/python/ops:manip_ops",
        "//tensorflow/python/ops:map_ops_gen",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/ops:random_crop_ops",
        "//tensorflow/python/ops:rnn",
        "//tensorflow/python/ops:rnn_cell",
        "//tensorflow/python/ops:rnn_ops_gen",
        "//tensorflow/python/ops:sendrecv_ops_gen",
        "//tensorflow/python/ops:sets",
        "//tensorflow/python/ops:standard_ops",
        "//tensorflow/python/ops:stateful_random_ops",
        "//tensorflow/python/ops:tpu_ops_gen",
        "//tensorflow/python/ops:uniform_quant_ops_gen",
        "//tensorflow/python/ops:while_v2",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg/sparse:sparse_py",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_arrays",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:np_dtypes",
        "//tensorflow/python/ops/numpy_ops:np_math_ops",
        "//tensorflow/python/ops/numpy_ops:np_random",
        "//tensorflow/python/ops/numpy_ops:np_utils",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/ragged:ragged_ops",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python/ops/structured:structured_ops",
        "//tensorflow/python/platform:app",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler",
        "//tensorflow/python/profiler:profiler_client",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/summary:tb_summary",
        "//tensorflow/python/tpu:tpu_noestimator",
        "//tensorflow/python/training",
        "//tensorflow/python/training:quantize_training",
        "//tensorflow/python/user_ops:ops",
        "//tensorflow/python/util:all_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_decorator_export",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
    ],
)

pybind_extension(
    name = "_pywrap_tfcompile",
    srcs = ["tfcompile_wrapper.cc"],
    data = [
        "_pywrap_tfcompile.pyi",
    ],
    deps = [
        "//tensorflow/compiler/aot:tfcompile_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        # The headers here cannot be brought in via cc_header_only_library
        "//tensorflow/compiler/aot:llvm_targets",
    ],
)

pybind_extension(
    name = "_pywrap_quantize_training",
    srcs = [
        "//tensorflow/python/training:quantize_training_wrapper.cc",
        "//tensorflow/core/common_runtime:quantize_training_hdrs"
    ],
    data = [
        "_pywrap_quantize_training.pyi",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:reader_base",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
    ],
)

# TODO(yanhuasun): Move this back and the source file back to lib/core directory.
pybind_extension(
    name = "_pywrap_py_exception_registry",
    srcs = ["py_exception_registry_wrapper.cc"],
    data = [
        "_pywrap_py_exception_registry.pyi",
    ],
    # Do not sort: core:py_exception_registry must come before platform:status
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "//third_party/python_runtime:headers",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:py_exception_registry",
        "//tensorflow/core/platform:status",
    ],
)

pybind_extension(
    name = "_pywrap_toco_api",
    srcs = [
        "lite/toco_python_api_wrapper.cc",
    ],
    data = [
        "_pywrap_toco_api.pyi",
    ],
    deps = [
        "//tensorflow/lite/toco/python:toco_python_api",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
    ],
)

py_strict_library(
    name = "extra_py_tests_deps",
    srcs_version = "PY3",
    # extra_py_tests_deps is public to allow the use of tf_py_test or
    # cuda_py_test build rules for out-of-tree custom ops.
    # extra_py_tests_deps should NOT be used as dependency directly
    # in out-of-tree BUILD files.
    # Some of the dependencies in :keras_lib are possibly outdated
    # or not generally useful, so a future task is to determine a good
    # set of dependencies.
    # TODO(b/199420795) investigate minimal set of dependencies ...
    visibility = ["//visibility:public"],
    deps = [
        ":keras_lib",
        "//third_party/py/numpy",
        "@pypi_scipy//:pkg",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "distributed_framework_test_lib",
    srcs_version = "PY3",
    deps = ["//tensorflow/python/framework:test_lib"],
)

# Note: this is a heavyweight library specialized for TensorFlow graphs. Do not use for
# other purposes.

py_strict_library(
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

py_strict_library(
    name = "pywrap_tensorflow",
    srcs = [
        "pywrap_tensorflow.py",
    ] + if_static(
        ["pywrap_dlopen_global_flags.py"],
        # Import will fail, indicating no global dlopen flags
        otherwise = [],
    ),  # b/153585257
    srcs_version = "PY3",
    deps = [
#        ":pywrap_tensorflow_internal",
        "//tensorflow/python/platform:self_check",
    ],
)

#pywrap_tensorflow_macro(
#    name = "pywrap_tensorflow_internal",
#    srcs = ["pywrap_tensorflow_internal.cc"],
#    exports_filter = [
#        "@pybind11",
#        "@local_config_python//:python_headers",
#    ] + if_windows([
#        "//:__subpackages__",
#        "@nsync//:__subpackages__",
#        "@com_google_absl//:__subpackages__",
#        "@com_google_protobuf//:__subpackages__",
#        "@double_conversion//:__subpackages__",
#        "@eigen_archive//:__subpackages__",
#        "@local_tsl//tsl:__subpackages__",
#        "@local_xla//xla:__subpackages__",
#    ]),
#    roots = [
#        "//tensorflow/python/lib/core:py_exception_registry",
#    ],
#    win_def_file = ":pywrap_tensorflow_filtered_def_file",
#    deps = [
#        "//tensorflow/c:c_api",
#        "//tensorflow/c:c_api_experimental",
#        "//tensorflow/c:checkpoint_reader",
#        "//tensorflow/c:env",
#        "//tensorflow/c:kernels",
#        "//tensorflow/c:kernels_experimental",
#        "//tensorflow/c:logging",
#        "//tensorflow/c:ops",
#        "//tensorflow/c:python_api",
#        "//tensorflow/c:safe_ptr",
#        "//tensorflow/c:tf_status_helper",
#        "//tensorflow/c/eager:c_api",
#        "//tensorflow/c/eager:c_api_experimental",
#        "//tensorflow/c/experimental/filesystem:filesystem_interface",
#        "//tensorflow/c/experimental/gradients",
#        "//tensorflow/c/experimental/gradients/tape",
#        "//tensorflow/c/experimental/ops",
#        "//tensorflow/c/experimental/stream_executor",
#        "//tensorflow/cc/saved_model:fingerprinting_impl",
#        "//tensorflow/cc/saved_model:loader_lite_impl",
#        "//tensorflow/cc/saved_model:metrics_impl",
#        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_cc_impl",
#        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
#        "//tensorflow/compiler/tf2tensorrt:op_converter_registry_impl",
#        "//tensorflow/compiler/tf2xla:tf2xla_opset",
#        "//tensorflow/core:framework_internal_impl",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:lib_internal_impl",
#        "//tensorflow/core:reader_base",
#        "//tensorflow/core/common_runtime:core_cpu_impl",
#        "//tensorflow/core/common_runtime/gpu:gpu_runtime_impl",
#        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
#        "//tensorflow/core/config:flag_defs",
#        "//tensorflow/core/config:flags",
#        "//tensorflow/core/data/service:dispatcher_client",
#        "//tensorflow/core/data/service:grpc_util",
#        "//tensorflow/core/data/service:py_utils",
#        "//tensorflow/core/data/service:server_lib",
#        "//tensorflow/core/debug",
#        "//tensorflow/core/distributed_runtime:server_lib",
#        "//tensorflow/core/function/runtime_client:runtime_client_cc",
#        "//tensorflow/core/grappler:grappler_item",
#        "//tensorflow/core/grappler:grappler_item_builder",
#        "//tensorflow/core/grappler/clusters:cluster",
#        "//tensorflow/core/grappler/clusters:single_machine",
#        "//tensorflow/core/grappler/clusters:virtual_cluster",
#        "//tensorflow/core/grappler/costs:graph_memory",
#        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
#        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
#        "//tensorflow/core/grappler/optimizers:meta_optimizer",
#        "//tensorflow/core/kernels:data_service_ops",
#        "//tensorflow/core/platform:stacktrace_handler",
#        "//tensorflow/core/profiler:profiler_impl",
#        "//tensorflow/core/profiler/backends/cpu:python_tracer",
#        "//tensorflow/core/profiler/internal:print_model_analysis",
#        "//tensorflow/core/util:determinism",
#        "//tensorflow/distribute/experimental/rpc/kernels:rpc_ops",
#        "//tensorflow/dtensor/cc:dtensor_device_cc",
#        "//tensorflow/dtensor/cc:tensor_layout",
#        "//tensorflow/lite/delegates/flex:delegate",
#        "//tensorflow/lite/kernels/shim:tf_kernel_shim",
#        "//tensorflow/lite/toco/python:toco_python_api",
#        "//tensorflow/python/client:tf_session_helper",
#        "//tensorflow/python/eager:pywrap_tfe_lib",
#        "//tensorflow/python/framework:op_def_util_cc",
#        "//tensorflow/python/framework:python_api_dispatcher",
#        "//tensorflow/python/framework:python_api_info",
#        "//tensorflow/python/framework:python_api_parameter_converter",
#        "//tensorflow/python/framework:python_op_gen",
#        "//tensorflow/python/framework:python_tensor_converter",
#        "//tensorflow/python/grappler:cost_analyzer_lib",
#        "//tensorflow/python/grappler:model_analyzer_lib",
#        "//tensorflow/python/lib/core:py_func_lib",
#        "//tensorflow/python/lib/core:pybind11_absl",
#        "//tensorflow/python/lib/core:pybind11_lib",
#        "//tensorflow/python/lib/core:pybind11_proto",
#        "//tensorflow/python/lib/core:pybind11_status",
#        "//tensorflow/python/lib/core:safe_pyobject_ptr",
#        "//tensorflow/python/util:cpp_nest",
#        "//tensorflow/python/util:cpp_python_util",
#        "//tensorflow/python/util:function_parameter_canonicalizer",
#        "//tensorflow/python/util:kernel_registry",
#        "//tensorflow/tools/graph_transforms:transform_graph_lib",
#        "//third_party/python_runtime:headers",
#        "@com_google_absl//absl/base",
#        "@com_google_absl//absl/container:inlined_vector",
#        "@com_google_absl//absl/types:optional",
#        "@com_google_absl//absl/types:span",
#        "@local_tsl//tsl/profiler/rpc:profiler_server_impl",
#        "@local_tsl//tsl/profiler/rpc/client:profiler_client_impl",
#        "@local_tsl//tsl/python/lib/core:numpy",
#        "@local_xla//xla/stream_executor:stream_executor_impl",
#    ] + select({
#        "//tensorflow/compiler/mlir/python:disable_mlir_config": [],
#        "//conditions:default": [
#            "//tensorflow/compiler/mlir/python:mlir",
#        ],
#    }) + if_static([
#        "//tensorflow/core/platform:tensor_float_32_utils",
#        "//tensorflow/core/platform:enable_tf2_utils",
#    ]) + if_google([
#        "//base",
#        "//tensorflow/core/util/tensor_bundle",
#    ]) + if_oss(if_static(
#        [clean_dep("@nsync//:nsync_headers")],
#    )) + if_oss([
#        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
#        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
#    ]) + if_oss(if_cuda_is_configured([
#        # TODO(tmorris): These dependencies are added to get the RPATHs for
#        # nvidia standalone wheels into pywrap_tensorflow_internal. We might be
#        # able to remove this in the future, as these stubs should already
#        # be brought in via other dependencies.
#        "@local_tsl//tsl/cuda:cudnn",
#        "@local_tsl//tsl/cuda:cufft",
#        "@local_tsl//tsl/cuda:nccl_rpath",
#    ])) + if_xla_available([
#        "//tensorflow/compiler/aot:tfcompile_lib",
#    ]) + tf_monitoring_python_deps() + tf_additional_plugin_deps() + tf_additional_profiler_deps() + tf_additional_binary_deps(),
#)

py_strict_library(
    name = "pywrap_mlir",
    srcs = ["pywrap_mlir.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_mlir",
        ":pywrap_tensorflow",
        "//tensorflow/python/eager:context",
    ],
)

pybind_extension(
    name = "_pywrap_mlir",
    srcs = [
        "mlir_wrapper.cc",
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    data = [
        "_pywrap_mlir.pyi",
    ],
    deps = [
        "//tensorflow/compiler/mlir/python:mlir",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
    ],
)

py_strict_library(
    name = "pywrap_sanitizers",
    srcs = ["pywrap_sanitizers.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_sanitizers",
        ":pywrap_tensorflow",
    ],
)

pybind_extension(
    name = "flags_pybind",
    srcs = [
        "//tensorflow/core/config:flags_api_wrapper.cc"
    ],
    data = [
        "flags_pybind.pyi",
    ],
    visibility = ["//tensorflow/core/config:__subpackages__"],
    deps = [
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/types:optional",
    ],
)

pybind_extension(
    name = "_pywrap_sanitizers",
    srcs = ["sanitizers_wrapper.cc"],
    data = [
        "_pywrap_sanitizers.pyi",
    ],
    deps = [
    ],
)

cc_library(
    name = "unified_api_pywrap_required_headers",
    textual_hdrs = [
        "//tensorflow/python/lib/core:basic_hdrs",
        "@local_tsl//tsl/python/lib/core:basic_hdrs",
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
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
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
    ],
)

py_strict_library(
    name = "pywrap_tfe",
    srcs = ["pywrap_tfe.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe",
        ":pywrap_tensorflow",
    ],
)

pybind_extension(
    name = "_pywrap_tfe",
    srcs = ["tfe_wrapper.cc",
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:py_exception_registry_hdr",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "//tensorflow/python/util:util_hdr",
        "@local_tsl//tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_tsl//tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_tsl//tsl/python/lib/core:numpy_hdr",
    ],
    data = [
        "_pywrap_tfe.pyi",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit:get_compiler_ir_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:reader_base",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        # copybara:uncomment "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
            "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
            "@local_tsl//tsl/protobuf:coordination_service_proto_cc",
        ],
    ),
)

py_strict_library(
    name = "pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = ["pywrap_tfe_monitoring_reader.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe_monitoring_reader",
    ],
)

pybind_extension(
    name = "_pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = [
        "tfe_wrapper_monitoring_reader.cc",
        "//tensorflow/c/eager:headers_monitoring_reader",
        "//tensorflow/c/eager:pywrap_headers_monitoring_reader",
        "//tensorflow/python/util:util_hdr",
    ],
    data = [
        "_pywrap_tfe_monitoring_reader.pyi",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental_reader",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit:get_compiler_ir_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:reader_base",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

pybind_extension(
    name = "_pywrap_parallel_device",
    srcs = [
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "//tensorflow/c/eager/parallel_device:headers",
        "//tensorflow/c/eager/parallel_device:sources",
        "//tensorflow/python/distribute/parallel_device:pywrap_parallel_device.cc",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    data = [
        "_pywrap_parallel_device.pyi",
    ],
    visibility = ["//tensorflow/python/distribute/parallel_device:__pkg__"],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/c/eager:tfe_cancellation_manager_internal",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:reader_base",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
    ],
)

pybind_extension(
    name = "_pywrap_dtensor_device",
    srcs = ["pywrap_dtensor_device.cc"],
    data = [
        "_pywrap_dtensor_device.pyi",
    ],
    deps = [
        "//tensorflow/dtensor/cc:dtensor_device_cc",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status_headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
    private_deps = [
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ]
)

py_strict_library(
    name = "tf2",
    srcs = ["tf2.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/util:tf_export",
    ],
)

# copybara:uncomment_begin(google-only)
# # TODO(b/287346201): Look into replacing usages of this target.
# py_proto_library(
#     name = "protos_all_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     deps = [":protos_all"],
# )
#
# # Special build rule for jax2tf.
# # TODO(b/239052279): remove once TF->TFLite dependency cycle is resolved
# py_strict_library(
#     name = "jax2tf_support",
#     visibility = ["//third_party/py/jax/experimental/jax2tf:__subpackages__"],
#     deps = [
#         "//tensorflow/python/dlpack",
#         "//tensorflow/python/ops/ragged:ragged_array_ops",
#         "//tensorflow/python/saved_model",
#     ],
# )
# copybara:uncomment_end

pytype_strict_library(
    name = "proto_exports",
    srcs = ["proto_exports.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:tf_export",
    ],
)

pywrap_library(
    name = "_pywrap_tensorflow",
    win_def_file = "_pywrap_tensorflow.def",
    deps = [
        ":_pywrap_quantize_training",
        ":_pywrap_tensorflow_cc_only",
        "//tensorflow/python/client:_pywrap_device_lib",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/client:_pywrap_tf_session",
        "//tensorflow/python/grappler:_pywrap_tf_item",
        "//tensorflow/python/data/experimental/service:_pywrap_server_lib",
        "//tensorflow/python/data/experimental/service:_pywrap_snapshot_utils",
        "//tensorflow/python/data/experimental/service:_pywrap_utils",
        "//tensorflow/python:_pywrap_mlir",
        "//tensorflow/compiler/mlir/quantization/tensorflow/calibrator:pywrap_calibration",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:pywrap_function_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:pywrap_quantize_model",
        "//tensorflow/python:_pywrap_toco_api",
        "//tensorflow/python:flags_pybind",
        "//tensorflow/python/framework:_dtypes",
        "//tensorflow/python/framework:_pywrap_python_op_gen",
        "//tensorflow/python/framework:_op_def_library_pybind",
        "//tensorflow/python/framework:_op_def_registry",
        "//tensorflow/python/framework:_proto_comparators",
        "//tensorflow/python/framework:_pywrap_python_api_dispatcher",
        "//tensorflow/python/framework:_test_metrics_util",
        "//tensorflow/python/grappler:_pywrap_tf_cluster",
        "//tensorflow/python/grappler:_pywrap_tf_optimizer",
        "//tensorflow/python/lib/core:_pywrap_py_func",
        "//tensorflow/python/lib/io:_pywrap_file_io",
        "//tensorflow/python/lib/io:_pywrap_record_io",
        "//tensorflow/python/platform:_pywrap_cpu_feature_guard",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/profiler/internal:_pywrap_profiler",
        "//tensorflow/python/profiler/internal:_pywrap_traceme",
        "//tensorflow/python:_pywrap_dtensor_device",
        "//tensorflow/python:_pywrap_parallel_device",
        "//tensorflow/python:_pywrap_py_exception_registry",
        "//tensorflow/python:_pywrap_sanitizers",
        "//tensorflow/python:_pywrap_tfcompile",
        "//tensorflow/python:_pywrap_tfe",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/util:fast_module_type",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_determinism",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_tensor_float_32_execution",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:_tf_stack",
        "//tensorflow/python/util:pywrap_xla_ops",
        "//tensorflow/python/util:_function_parameter_canonicalizer_binding_for_test",
        "//tensorflow/python/client:_pywrap_debug_events_writer",
        "//tensorflow/python/tpu:_pywrap_tpu_embedding",
        "//tensorflow/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//tensorflow/python/framework:_errors_test_helper",
        "//tensorflow/python/framework:_python_memory_checker_helper",
        "//tensorflow/python/grappler:_pywrap_cost_analyzer",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python/grappler:_pywrap_model_analyzer",
        "//tensorflow/python/framework:_pywrap_python_api_info",
        "//tensorflow/python/framework:_pywrap_python_tensor_converter",
        "//tensorflow/python/framework/experimental:_unified_api",
        "//tensorflow/python/framework:_pywrap_python_api_parameter_converter",
        "//tensorflow/python/framework:_op_def_util",
        "//tensorflow/compiler/mlir/quantization/stablehlo/python:pywrap_quantization",
        "//tensorflow/compiler/mlir/tfr:tfr_wrapper",
        "//tensorflow/python/framework:_py_context_manager",
        "//tensorflow/python/framework/experimental:_math_ops",
        "//tensorflow/python/framework/experimental:_nn_ops",
        "//tensorflow/python/framework/experimental:_tape",
    ],
    linkopts = select({
        "//tensorflow:windows": [
            "-DEFAULTLIB:ws2_32.lib",
            "-DEFAULTLIB:advapi32.lib",
            "-DEFAULTLIB:crypt32.lib",
            "-DEFAULTLIB:Normaliz.lib",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

pybind_extension(
    name = "_pywrap_tensorflow_cc_only",
    deps = [
#        "//tensorflow/core:framework_internal",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/kernels:data_service_ops",
        "//tensorflow/core/kernels:reader_ops",
    ]
)

pywrap_common_library(
    name = "_pywrap_tensorflow_common",
    dep = ":_pywrap_tensorflow",
)