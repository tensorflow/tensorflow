#
# Public targets:
#  ":platform" - Low-level and platform-specific Python code.

load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow:tensorflow.bzl",
    "VERSION",
    "cc_header_only_library",
    "if_google",
    "if_nccl",
    "if_oss",
    "if_windows",
    "if_xla_available",
    "tf_enable_mlir_bridge",
    "tf_python_pybind_static_deps",
    "tsl_async_value_deps",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
    "pybind_extension",
    "pywrap_aware_cc_import",
    "pywrap_aware_filegroup",
    "pywrap_aware_genrule",
    "pywrap_binaries",
    "pywrap_common_library",
    "pywrap_library",
    "pywrap_tensorflow_macro",
    "tf_external_workspace_visible",
    "tf_monitoring_python_deps",
    "tf_pybind_cc_library_wrapper",
    "tf_python_pybind_extension",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_binary_deps",
    "tf_proto_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_pywrap",
    "if_static",
    "tf_additional_plugin_deps",
    "tf_additional_profiler_deps",
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
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:internal",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/python/tools/api/generator:__pkg__",
        "//tensorflow/tools/api/tests:__pkg__",
        "//tensorflow/tools/compatibility/update:__pkg__",
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
    visibility = [
        "//tensorflow:__pkg__",
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
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo/python:pywrap_tensorflow_to_stablehlo",
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
        "//tensorflow/python/distribute:strategy_combinations",  # For tf.__internal__,
        "//tensorflow/python/distribute/experimental/rpc:rpc_ops",
        "//tensorflow/python/dlpack",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:_pywrap_python_op_gen",
        "//tensorflow/python/framework:_test_metrics_util",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
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
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/summary/writer",
        "//tensorflow/python/tools:module_util",
        "//tensorflow/python/tools/api/generator:create_python_api",
        "//tensorflow/python/tpu:_pywrap_sparse_core_layout",
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
        "//tensorflow/python/checkpoint/sharding:sharding_policies",
        "//tensorflow/python/checkpoint/sharding:sharding_util",
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
        "//tensorflow/python/ops:array_ops",
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

tf_python_pybind_extension(
    name = "_pywrap_tfcompile",
    srcs = ["tfcompile_wrapper.cc"],
    dynamic_deps = [":_pywrap_tensorflow_internal.so"] + select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        "//tensorflow:windows": [],
    }),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tfcompile.pyi",
    ],
    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "@pybind11",
        "//tensorflow/compiler/aot:tfcompile_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        # The headers here cannot be brought in via cc_header_only_library
        "//tensorflow/compiler/aot:llvm_targets",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_quantize_training",
    srcs = [
        "//tensorflow/python/training:quantize_training_wrapper.cc",
    ],
    hdrs = ["//tensorflow/core/common_runtime:quantize_training_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_quantize_training.pyi",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

# TODO(yanhuasun): Move this back and the source file back to lib/core directory.
tf_python_pybind_extension(
    name = "_pywrap_py_exception_registry",
    srcs = ["py_exception_registry_wrapper.cc"],
    dynamic_deps = [":_pywrap_tensorflow_internal.so"] + select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        "//tensorflow:windows": [],
    }),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_py_exception_registry.pyi",
    ],
    static_deps = tf_python_pybind_static_deps(),
    # Do not sort: core:py_exception_registry must come before platform:status
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@pybind11",
        "@local_xla//third_party/python_runtime:headers",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:py_exception_registry",
        "//tensorflow/core/platform:status",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_toco_api",
    srcs = [
        "lite/toco_python_api_wrapper.cc",
    ],
    hdrs = ["//tensorflow/lite/toco/python:toco_python_api_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_toco_api.pyi",
    ],
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap(["//tensorflow/lite/toco/python:toco_python_api"]),
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

cc_header_only_library(
    name = "python_op_gen_headers_lib",
    extra_deps = [
        "//tensorflow/core:protos_all_cc",
    ],
    features = ["-parse_headers"],
    tags = ["no-ide"],
    deps = [
        "//tensorflow/python/framework:python_op_gen",
    ],
)

py_strict_library(
    name = "extra_py_tests_deps",
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
    deps = ["//tensorflow/python/framework:test_lib"],
)

# Note: this is a heavyweight library specialized for TensorFlow graphs. Do not use for
# other purposes.

py_strict_library(
    name = "global_test_configuration",
    compatible_with = get_compatible_with_portable(),
    deps =
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
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_tensorflow_internal",
        "//tensorflow/python/platform:self_check",
    ],
)

pywrap_tensorflow_macro(
    name = "pywrap_tensorflow_internal",
    srcs = ["pywrap_tensorflow_internal.cc"],
    dynamic_deps = select({
        "//tensorflow:macos": [
            "//tensorflow:libtensorflow_cc.%s.dylib" % VERSION,
            "//tensorflow:libtensorflow_framework.%s.dylib" % VERSION,
        ],
        "//conditions:default": [
            "//tensorflow:libtensorflow_cc.so.%s" % VERSION,
            "//tensorflow:libtensorflow_framework.so.%s" % VERSION,
        ],
        "//tensorflow:windows": [],
    }),
    exports_filter = [
        "@pybind11",
        "@local_config_python//:python_headers",
        "@com_googlesource_code_re2//:__subpackages__",
    ] + if_windows([
        "//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "@local_xla//xla:__subpackages__",
    ]),
    roots = [
        "//tensorflow/python/lib/core:py_exception_registry",
    ],
    static_deps = [
        "@arm_neon_2_x86_sse//:__subpackages__",
        "@bazel_tools//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@clog//:__subpackages__",
        "@com_github_cares_cares//:__subpackages__",
        "@com_github_googlecloudplatform_tensorflow_gcp_tools//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_googleapis//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@compute_library//:__subpackages__",
        "@cpuinfo//:__subpackages__",
        "@curl//:__subpackages__",
        "@dlpack//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@farmhash_gpu_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@flatbuffers//:__subpackages__",
        "@FP16//:__subpackages__",
        "@FXdiv//:__subpackages__",
        "@gemmlowp//:__subpackages__",
        "@gif//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@hwloc//:__subpackages__",
        "@icu//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@llvm_openmp//:__subpackages__",
        "@llvm-project//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_git//:__subpackages__",
        "@local_config_nccl//:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        "@local_execution_config_platform//:__subpackages__",
        "@mkl_dnn_acl_compatible//:__subpackages__",
        "@nccl_archive//:__subpackages__",
        "@nvtx_archive//:__subpackages__",
        "@onednn//:__subpackages__",
        "@org_sqlite//:__subpackages__",
        "@platforms//:__subpackages__",
        "@png//:__subpackages__",
        "@pthreadpool//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@ruy//:__subpackages__",
        "@snappy//:__subpackages__",
        "@sobol_data//:__subpackages__",
        "@stablehlo//:__subpackages__",
        "@tf_runtime//:__subpackages__",
        "//:__subpackages__",
        "@upb//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@zlib//:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "@local_xla//xla:__subpackages__",
    ] + tsl_async_value_deps(),
    win_def_file = ":pywrap_tensorflow_filtered_def_file",
    deps = [
        ":_pywrap_library_dependency_enforcer",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/c:env",
        "//tensorflow/c:kernels",
        "//tensorflow/c:kernels_experimental",
        "//tensorflow/c:logging",
        "//tensorflow/c:ops",
        "//tensorflow/c:python_api",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/experimental/filesystem:filesystem_interface",
        "//tensorflow/c/experimental/gradients",
        "//tensorflow/c/experimental/gradients/tape",
        "//tensorflow/c/experimental/ops",
        "//tensorflow/c/experimental/stream_executor",
        "//tensorflow/cc/saved_model:fingerprinting_impl",
        "//tensorflow/cc/saved_model:loader_lite_impl",
        "//tensorflow/cc/saved_model:metrics_impl",
        "//tensorflow/compiler/mlir/lite/python:converter_python_api",
        "//tensorflow/compiler/mlir/quantization/stablehlo/python:pywrap_quantization_lib_impl",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_cc_impl",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo/python:pywrap_tensorflow_to_stablehlo_lib_impl",
        "//tensorflow/compiler/tf2tensorrt:op_converter_registry_impl",
        "//tensorflow/compiler/tf2xla:tf2xla_opset",
        "//tensorflow/core:framework_internal_impl",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal_impl",
        "//tensorflow/core:reader_base",
        "//tensorflow/core/common_runtime:core_cpu_impl",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime_impl",
        "//tensorflow/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
        "//tensorflow/core/config:flag_defs",
        "//tensorflow/core/config:flags",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/core/data/service:py_utils",
        "//tensorflow/core/data/service:server_lib",
        "//tensorflow/core/debug",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/function/runtime_client:runtime_client_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/costs:graph_memory",
        "//tensorflow/core/grappler/graph_analyzer:graph_analyzer_tool",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/kernels:data_service_ops",
        "//tensorflow/core/platform:stacktrace_handler",
        "//tensorflow/core/profiler:profiler_impl",
        "//tensorflow/core/profiler/internal:print_model_analysis",
        "//tensorflow/core/tpu/kernels:sparse_core_layout",
        "//tensorflow/core/util:determinism",
        "//tensorflow/distribute/experimental/rpc/kernels:rpc_ops",
        "//tensorflow/dtensor/cc:dtensor_device_cc",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/kernels/shim:tf_kernel_shim",
        "//tensorflow/lite/toco/python:toco_python_api",
        "//tensorflow/python/client:tf_session_helper",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/framework:op_def_util_cc",
        "//tensorflow/python/framework:python_api_dispatcher",
        "//tensorflow/python/framework:python_api_info",
        "//tensorflow/python/framework:python_api_parameter_converter",
        "//tensorflow/python/framework:python_op_gen",
        "//tensorflow/python/framework:python_tensor_converter",
        "//tensorflow/python/grappler:cost_analyzer_lib",
        "//tensorflow/python/grappler:model_analyzer_lib",
        "//tensorflow/python/lib/core:py_func_lib",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_proto",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_nest",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:function_parameter_canonicalizer",
        "//tensorflow/python/util:kernel_registry",
        "//tensorflow/tools/graph_transforms:transform_graph_lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",
        "@local_xla//xla/backends/profiler/cpu:python_tracer",
        "@local_xla//xla/stream_executor:stream_executor_impl",
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ] + select({
        "//tensorflow/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//tensorflow/compiler/mlir/python:mlir",
        ],
    }) + if_static([
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core/platform:enable_tf2_utils",
    ]) + if_google([
        "//base",
        "//tensorflow/core/util/tensor_bundle",
    ]) + if_oss([
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
    ]) + if_oss(if_cuda_is_configured([
        # TODO(tmorris): These dependencies are added to get the RPATHs for
        # nvidia standalone wheels into pywrap_tensorflow_internal. We might be
        # able to remove this in the future, as these stubs should already
        # be brought in via other dependencies.
        "@local_xla//xla/tsl/cuda:cudnn",
        "@local_xla//xla/tsl/cuda:cufft",
        "@local_xla//xla/tsl/cuda:cupti",
        "@local_xla//xla/tsl/cuda:nccl_rpath",
    ])) + if_xla_available([
        "//tensorflow/compiler/aot:tfcompile_lib",
    ]) + tf_monitoring_python_deps() + tf_additional_plugin_deps() + tf_additional_profiler_deps() + tf_additional_binary_deps(),
)

# ** Targets for Windows build (start) **
# We need the following targets to expose symbols from _pywrap_tensorflow.dll

pywrap_aware_filegroup(
    name = "win_lib_files_for_exported_symbols",
    srcs = [
        "//tensorflow/c:checkpoint_reader",  # checkpoint_reader
        "//tensorflow/c:python_api",  # tf_session
        "//tensorflow/c:safe_ptr",  # checkpoint_reader
        "//tensorflow/c:tf_status_helper",  # tfe
        "//tensorflow/cc/saved_model:fingerprinting_impl",  # SavedModel fingerprinting
        "//tensorflow/cc/saved_model:metrics_impl",  # SavedModel metrics
        "//tensorflow/compiler/jit:flags",  # tfe
        "//tensorflow/compiler/jit:get_compiler_ir",  # tfe
        "//tensorflow/compiler/mlir/lite/python:converter_python_api",  # converter
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_cc_impl",  # quantization
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo/python:pywrap_tensorflow_to_stablehlo_lib_impl",  # tensorflow_to_stablehlo
        "//tensorflow/compiler/tf2xla:tf2xla_opset",  # pywrap_xla_ops
        "//tensorflow/core:framework_internal_impl",  # op_def_registry
        "//tensorflow/core:lib_internal_impl",  # device_lib
        "//tensorflow/core:op_gen_lib",  # tf_session
        "//tensorflow/core/common_runtime:graph_constructor",  # tf_session
        "//tensorflow/core/common_runtime:quantize_training",  # quantize_training
        "//tensorflow/core/common_runtime:session_options",  # device_lib, tfe, tf_session
        "//tensorflow/core/common_runtime:session_state",  # tf_session
        "//tensorflow/core/common_runtime/eager:context",  # tfe
        "//tensorflow/core/common_runtime/eager:eager_executor",  # tfe
        "//tensorflow/core/common_runtime/eager:tensor_handle",  # tfe
        "//tensorflow/core/config:flag_defs",  # flags_api
        "//tensorflow/core/config:flags",  # flags_api
        "//tensorflow/core/data/service:dispatcher_client",  # dispatcher_client
        "//tensorflow/core/data/service:grpc_util",  # grpc_util
        "//tensorflow/core/data/service:server_lib",  # server_lib
        "//tensorflow/core/framework:attr_value_proto_cc",  # tf_text
        "//tensorflow/core/framework:op_def_proto_cc",  # tf_text
        "//tensorflow/core/function/runtime_client:runtime_client_cc",  # runtime_client_pybind
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
        "//tensorflow/core/platform:cpu_feature_guard",  # cpu_feature_guard
        "//tensorflow/core/platform:statusor",  # tfe
        "//tensorflow/core/profiler/internal:print_model_analysis",  # tfprof
        "//tensorflow/core/profiler/rpc/client:profiler_client_impl",  # profiler
        "//tensorflow/core/tpu/kernels:sparse_core_layout",  # sparse_core_layouts
        "//tensorflow/core/util:determinism",  # determinism
        "//tensorflow/core/util:port",  # util_port
        "//tensorflow/core/util/tensor_bundle",  # checkpoint_reader
        "//tensorflow/dtensor/cc:dtensor_device_cc",  # DTensor
        "//tensorflow/dtensor/cc:tensor_layout",  # DTensor
        "//tensorflow/lite/toco/python:toco_python_api",  # toco
        "//tensorflow/python/client:tf_session_helper",  # tf_session
        "//tensorflow/python/eager:pywrap_tfe_lib",  # pywrap_tfe_lib
        "//tensorflow/python/framework:op_def_util_cc",  # op_def_util
        "//tensorflow/python/framework:python_api_dispatcher",  # python_api_dispatcher
        "//tensorflow/python/framework:python_api_info",  # python_api_info
        "//tensorflow/python/framework:python_api_parameter_converter",  # python_api_parameter_converter
        "//tensorflow/python/framework:python_op_gen",  # python_op_gen
        "//tensorflow/python/framework:python_tensor_converter",  # python_tensor_converter
        "//tensorflow/python/grappler:cost_analyzer_lib",
        "//tensorflow/python/grappler:model_analyzer_lib",  # model_analyzer
        "//tensorflow/python/lib/core:ndarray_tensor",  # checkpoint_reader
        "//tensorflow/python/lib/core:py_exception_registry",  # py_exception_registry
        "//tensorflow/python/lib/core:py_func_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",  # checkpoint_reader
        "//tensorflow/python/util:cpp_nest",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:kernel_registry",
        "//tensorflow/tools/graph_transforms:transform_graph_lib",  # transform_graph
        "@local_tsl//tsl/platform:tensor_float_32_utils",  # tensor_float_32
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",  # profiler
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",  # profiler
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",  # profiler
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/python/lib/core:ml_dtypes_lib",  # bfloat16, float8_e4m3fn, float8_e5m2
        "@local_xla//xla/tsl/python/lib/core:numpy",  # checkpoint_reader
    ] + select({
        "//tensorflow/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//tensorflow/compiler/mlir/python:mlir",  # mlir
        ],
    }) + if_xla_available([
        "//tensorflow/compiler/aot:tfcompile_lib",  # tfcompile
        "@local_xla//xla:status_macros",  # tfcompile
        "@local_xla//xla/hlo/ir:hlo",  # tfcompile
        "@local_xla//xla/tsl/concurrency:async_value",  # tfcompile
    ]),
    visibility = ["//visibility:private"],
)

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
pywrap_aware_genrule(
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
pywrap_aware_genrule(
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

# Get the import library of _pywrap_tensorflow_internal.pyd, platform-specific to Windows.
pywrap_aware_filegroup(
    name = "get_pywrap_tensorflow_import_lib_file",
    srcs = [":_pywrap_tensorflow_internal.so"],
    output_group = "interface_library",
)

pywrap_aware_cc_import(
    name = "_pywrap_tensorflow_internal_linux",
    shared_library = "//tensorflow/python:lib_pywrap_tensorflow_internal.so",
    visibility = tf_external_workspace_visible(visibility),
)

pywrap_aware_cc_import(
    name = "_pywrap_tensorflow_internal_macos",
    shared_library = "//tensorflow/python:lib_pywrap_tensorflow_internal.dylib",
    visibility = tf_external_workspace_visible(visibility),
)

pywrap_aware_cc_import(
    name = "_pywrap_tensorflow_internal_windows",
    interface_library = "//tensorflow/python:pywrap_tensorflow_import_lib_file",
    shared_library = "//tensorflow/python:_pywrap_tensorflow_internal.dll",
    visibility = tf_external_workspace_visible(visibility),
)

# Rename the import library for _pywrap_tensorflow_internal.pyd to _pywrap_tensorflow_internal.lib
# (It was _pywrap_tensorflow_internal.so.if.lib).
pywrap_aware_genrule(
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
pywrap_aware_cc_import(
    name = "pywrap_tensorflow_import_lib",
    interface_library = select({
        "//tensorflow:windows": ":pywrap_tensorflow_import_lib_file",
        "//conditions:default": "not_existing_on_unix.lib",  # Just a placeholder for Unix platforms
    }),
    system_provided = 1,
)

# ** Targets for Windows build (end) **

py_strict_library(
    name = "pywrap_mlir",
    srcs = ["pywrap_mlir.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_mlir",
        ":pywrap_tensorflow",
        "//tensorflow/python/eager:context",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_mlir",
    srcs = ["mlir_wrapper.cc"],
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "//tensorflow/compiler/mlir/python:pywrap_mlir_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_mlir.pyi",
    ],
    deps = [
        "//tensorflow/compiler/tf2tensorrt:common_utils",
        "//tensorflow/compiler/tf2tensorrt:trt_parameters",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap(["//tensorflow/compiler/mlir/python:mlir"]),
)

py_strict_library(
    name = "pywrap_sanitizers",
    srcs = ["pywrap_sanitizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_sanitizers",
        ":pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "flags_pybind",
    srcs = ["//tensorflow/core/config:flags_api_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "flags_pybind.pyi",
    ],
    visibility = [
        "//tensorflow/core/config:__subpackages__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:optional",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_sanitizers",
    srcs = ["sanitizers_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_sanitizers.pyi",
    ],
    deps = [
        "@pybind11",
    ],
)

cc_library(
    name = "unified_api_pywrap_required_headers",
    textual_hdrs = [
        "//tensorflow/python/lib/core:basic_hdrs",
        "@local_xla//xla/tsl/python/lib/core:basic_hdrs",
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
        "@local_xla//third_party/python_runtime:headers",
    ],
)

py_strict_library(
    name = "pywrap_tfe",
    srcs = ["pywrap_tfe.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe",
        ":pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfe",
    srcs = ["tfe_wrapper.cc"],
    hdrs = if_pywrap(
        if_false = [
            "//tensorflow/c:headers",
            "//tensorflow/c:safe_ptr_hdr",
            "//tensorflow/c/eager:headers",
            "//tensorflow/c/eager:pywrap_required_hdrs",
            "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
            "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
            "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
            "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
            "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
            "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
            "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
            "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
            "//tensorflow/python/eager:pywrap_required_hdrs",
            "//tensorflow/python/lib/core:py_exception_registry_hdr",
            "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
            "//tensorflow/python/util:util_hdr",
        ],
    ),
    dynamic_deps = [":_pywrap_tensorflow_internal.so"] + select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        "//tensorflow:windows": [],
    }),
    enable_stub_generation = True,
    linkopts = [],
    pytype_srcs = [
        "_pywrap_tfe.pyi",
    ],
    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//tensorflow/compiler/jit:flags_headers_only",
        "//tensorflow/compiler/jit:get_compiler_ir_hdrs_only",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@local_xla//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@dlpack",
        "@pybind11",
        # copybara:uncomment "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        if_true = ["//tensorflow/python/eager:pywrap_tfe_lib"],
    ) + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
        ],
    ),
)

py_strict_library(
    name = "pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = ["pywrap_tfe_monitoring_reader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe_monitoring_reader",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = ["tfe_wrapper_monitoring_reader.cc"],
    hdrs = [
        "//tensorflow/c/eager:headers_monitoring_reader",
        "//tensorflow/c/eager:pywrap_headers_monitoring_reader",
        "//tensorflow/python/util:util_hdr",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tfe_monitoring_reader.pyi",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental_reader",
        "//tensorflow/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//tensorflow/compiler/jit:flags_headers_only",
        "//tensorflow/compiler/jit:get_compiler_ir_hdrs_only",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_parallel_device.pyi",
    ],
    visibility = [
        "//tensorflow/python/distribute/parallel_device:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/c/eager:tfe_cancellationmanager_internal_hdrs_only",
        "//tensorflow/c/eager:tfe_op_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_dtensor_device",
    srcs = ["pywrap_dtensor_device.cc"],
    data = [
        "_pywrap_dtensor_device.pyi",
    ],
    enable_stub_generation = True,
    features = ["-layering_check"],
    deps = [
        ":pywrap_densor_device_headers",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status_headers",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_densor_device_headers",
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/dtensor/cc:dtensor_device_cc",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

py_strict_library(
    name = "tf2",
    srcs = ["tf2.py"],
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

pybind_extension(
    name = "_pywrap_tensorflow_internal",
    srcs = ["pywrap_tensorflow_internal.cc"],
    pywrap_only = True,
    deps = [
        ":_pywrap_library_dependency_enforcer",
        "@pybind11",
    ],
)

pybind_extension(
    name = "_pywrap_tensorflow_cc_only",
    srcs = [],
    deps = [
        ":_protobuf_inline_symbols_enforcer",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
        "//tensorflow/core/kernels:data_service_ops",
        "//tensorflow/core/kernels:reader_ops",
        "//tensorflow/distribute/experimental/rpc/kernels:rpc_ops",
        "//tensorflow/dtensor/cc:tensor_layout",
        "@local_xla//xla/backends/profiler/cpu:python_tracer",
    ],
)

cc_library(
    name = "_protobuf_inline_symbols_enforcer",
    srcs = ["protobuf_inline_symbols_enforcer.cc"],
    local_defines = select({
        "//tensorflow:windows": [
            "PROTOBUF_USE_DLLS",
            "LIBPROTOBUF_EXPORTS",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/grappler/costs:op_performance_data_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "_pywrap_library_dependency_enforcer",
    srcs = ["pywrap_library_dependency_enforcer.cc"],
    hdrs = ["pywrap_library_dependency_enforcer.h"],
    alwayslink = True,
)

cc_library(
    name = "_cuda_deps_filter",
    srcs = [],
    deps = if_pywrap(
        if_true = (
            if_cuda_is_configured([
                "@local_xla//xla/tsl/cuda:cupti",
                "@local_xla//xla/tsl/cuda:nvshmem_stub",
            ]) + if_nccl([
                "@local_xla//xla/tsl/cuda:nccl",
            ])
        ),
    ),
)

pywrap_library(
    name = "_pywrap_tensorflow",
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_def_files_or_filters = {
        "tensorflow/python/_pywrap_tensorflow_common": ":_pywrap_tensorflow.def",
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_filters = {
        "tensorflow/tensorflow_framework": select({
            "//tensorflow:windows": [],
            "//conditions:default": [
                "//tensorflow:tensorflow_framework_pywrap_filter",
                ":_cuda_deps_filter",
                "@com_google_absl//absl/debugging:leak_check",
                "@com_google_absl//absl/flags:flag",
                "@com_google_absl//absl/log:die_if_null",
                "@com_google_absl//absl/random",
            ],
        }),
        "tensorflow/tensorflow_cc": select({
            "//tensorflow:windows": [],
            "//conditions:default": ["//tensorflow:tensorflow_cc_pywrap_filter"],
        }),
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_linkopts = {
        "tensorflow/tensorflow_framework": select({
            "//tensorflow:windows": [],
            "//tensorflow:macos": [
                "-lpthread",
                "-ldl",
                "-lm",
            ],
            "//conditions:default": [
                "-z defs",
                "-lpthread",
                "-ldl",
                "-lm",
            ],
        }),
        "tensorflow/tensorflow_cc": select({
            "//tensorflow:windows": [
                "-DEFAULTLIB:ws2_32.lib",
                "-DEFAULTLIB:advapi32.lib",
                "-DEFAULTLIB:crypt32.lib",
                "-DEFAULTLIB:Normaliz.lib",
                "-DEFAULTLIB:ntdll.lib",
            ],
            "//tensorflow:macos": [
                "-lpthread",
                "-ldl",
                "-lm",
            ],
            "//conditions:default": [
                "-z defs",
                "-lpthread",
                "-ldl",
                "-lm",
            ],
        }),
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_version_scripts = {
        "tensorflow/tensorflow_cc": select({
            "//tensorflow:windows": None,
            "//tensorflow:macos": "//tensorflow:tf_exported_symbols.lds",
            "//conditions:default": "//tensorflow:tf_version_script.lds",
        }),
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_versions = {
        "tensorflow/tensorflow_framework": "2",
        "tensorflow/tensorflow_cc": "2",
    },
    enable_common_lib_starlark_only_filter = select({
        "//tensorflow:windows": False,
        "//tensorflow:macos": False,
        "//conditions:default": True,
    }),
    starlark_only_deps = [
        "//tensorflow/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//tensorflow/compiler/mlir/quantization/stablehlo/python:pywrap_quantization",
        "//tensorflow/compiler/mlir/tfr:tfr_wrapper",
        "//tensorflow/python/framework:_errors_test_helper",
        "//tensorflow/python/framework/experimental:_math_ops",
        "//tensorflow/python/framework/experimental:_nn_ops",
        "//tensorflow/python/framework/experimental:_tape",
        "//tensorflow/python/framework/experimental:_unified_api",
        "//tensorflow/python/framework:_op_def_util",
        "//tensorflow/python/framework:_py_context_manager",
        "//tensorflow/python/framework:_pywrap_python_api_info",
        "//tensorflow/python/framework:_pywrap_python_api_parameter_converter",
        "//tensorflow/python/framework:_pywrap_python_tensor_converter",
        "//tensorflow/python/grappler:_pywrap_cost_analyzer",
        "//tensorflow/python/grappler:_pywrap_model_analyzer",
        "//tensorflow/python/util:_function_parameter_canonicalizer_binding_for_test",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite/python:_pywrap_converter_api",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:pywrap_function_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:pywrap_quantize_model",
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo/python:pywrap_tensorflow_to_stablehlo",
        "//tensorflow/compiler/tf2tensorrt:_pywrap_py_utils",
        "//tensorflow/python:_pywrap_dtensor_device",
        "//tensorflow/python:_pywrap_mlir",
        "//tensorflow/python:_pywrap_parallel_device",
        "//tensorflow/python:_pywrap_py_exception_registry",
        "//tensorflow/python:_pywrap_quantize_training",
        "//tensorflow/python:_pywrap_sanitizers",
        "//tensorflow/python:_pywrap_tensorflow_cc_only",
        "//tensorflow/python:_pywrap_tensorflow_internal",
        "//tensorflow/python:_pywrap_tfcompile",
        "//tensorflow/python:_pywrap_tfe",
        "//tensorflow/python:_pywrap_toco_api",
        "//tensorflow/python:flags_pybind",
        "//tensorflow/python/autograph/impl/testing:pybind_for_testing",
        "//tensorflow/python/client:_pywrap_debug_events_writer",
        "//tensorflow/python/client:_pywrap_device_lib",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/client:_pywrap_tf_session",
        "//tensorflow/python/data/experimental/service:_pywrap_server_lib",
        "//tensorflow/python/data/experimental/service:_pywrap_snapshot_utils",
        "//tensorflow/python/data/experimental/service:_pywrap_utils_exp",
        "//tensorflow/python/framework:_dtypes",
        "//tensorflow/python/framework:_op_def_library_pybind",
        "//tensorflow/python/framework:_op_def_registry",
        "//tensorflow/python/framework:_proto_comparators",
        "//tensorflow/python/framework:_python_memory_checker_helper",
        "//tensorflow/python/framework:_pywrap_python_api_dispatcher",
        "//tensorflow/python/framework:_pywrap_python_op_gen",
        "//tensorflow/python/framework:_test_metrics_util",
        "//tensorflow/python/grappler:_pywrap_tf_cluster",
        "//tensorflow/python/grappler:_pywrap_tf_item",
        "//tensorflow/python/grappler:_pywrap_tf_optimizer",
        "//tensorflow/python/lib/core:_pywrap_py_func",
        "//tensorflow/python/lib/io:_pywrap_file_io",
        "//tensorflow/python/lib/io:_pywrap_record_io",
        "//tensorflow/python/platform:_pywrap_cpu_feature_guard",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/profiler/internal:_pywrap_profiler",
        "//tensorflow/python/profiler/internal:_pywrap_traceme",
        "//tensorflow/python/saved_model:pywrap_saved_model",
        "//tensorflow/python/tpu:_pywrap_sparse_core_layout",
        "//tensorflow/python/tpu:_pywrap_tpu_embedding",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_determinism",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tensor_float_32_execution",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:_tf_stack",
        "//tensorflow/python/util:fast_module_type",
        "//tensorflow/python/util:pywrap_xla_ops",
    ],
)

pywrap_common_library(
    name = "pywrap_tensorflow_framework",
    dep = ":_pywrap_tensorflow",
    filter_name = "tensorflow_framework",
    visibility = ["//visibility:public"],
)

pywrap_common_library(
    name = "pywrap_tensorflow_common",
    dep = ":_pywrap_tensorflow",
    filter_name = "_pywrap_tensorflow_common",
    visibility = ["//visibility:public"],
)

alias(
    name = "tensorflow_common_framework",
    actual = select({
        "//conditions:default": ":pywrap_tensorflow_framework",
        "//tensorflow:macos": ":pywrap_tensorflow_framework",
        "//tensorflow:windows": ":pywrap_tensorflow_common",
    }),
    visibility = ["//visibility:public"],
)

pywrap_binaries(
    name = "pywrap_tensorflow_binaries",
    dep = ":_pywrap_tensorflow",
)
