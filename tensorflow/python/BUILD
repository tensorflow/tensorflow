#
# Public targets:
#  ":platform" - Low-level and platform-specific Python code.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "VERSION", "cc_header_only_library", "clean_dep", "if_google", "if_mlir", "if_oss", "if_windows", "if_xla_available", "py_test", "tf_enable_mlir_bridge", "tf_python_pybind_static_deps", "tsl_async_value_deps")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "get_compatible_with_portable", "pywrap_tensorflow_macro", "tf_external_workspace_visible", "tf_monitoring_python_deps", "tf_py_test", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")
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
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")
load("//third_party/mkl_dnn:build_defs.bzl", "if_onednn_v3")

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
    "//third_party/py/keras:__subpackages__",
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

py_library(
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
        ":gradient_checker_v2",
        ":no_contrib",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/platform:app",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/platform:sysconfig",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/summary:summary_py",
        "//tensorflow/python/tpu:tpu_estimator",
        "//tensorflow/python/util:all_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

# Some of the dependencies in keras_lib are possibly outdated or not generally
# useful, so a future task is to determine a good set of dependencies.
# keras_lib may be removed in the future, it should NOT be used as
# a direct dependency.
# TODO(b/199420795) investigate minimal set of dependencies ...
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
        ":_pywrap_py_exception_registry",
        ":_pywrap_quantize_training",
        ":_pywrap_utils",
        ":array_ops",
        ":array_ops_stack",
        ":audio_ops_gen",
        ":bincount_ops",
        ":bitwise_ops",
        ":boosted_trees_ops",
        ":check_ops",
        ":client_testlib",
        ":clustering_ops",
        ":collective_ops",
        ":composite_tensor_ops",
        ":cond",
        ":cond_v2",
        ":confusion_matrix",
        ":control_flow_assert",
        ":control_flow_case",
        ":control_flow_ops",
        ":control_flow_switch_case",
        ":cudnn_rnn_ops_gen",
        ":distributed_framework_test_lib",
        ":dtensor",
        ":filesystem_ops",
        ":functional_ops",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":histogram_ops",
        ":image_ops",
        ":initializers_ns",
        ":io_ops",
        ":keras_lib",
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
        ":proto_ops",
        ":pywrap_tensorflow",
        ":pywrap_tfe",
        ":random_crop_ops",
        ":ref_variable",
        ":rnn_ops_gen",
        ":script_ops",
        ":sendrecv_ops_gen",
        ":session_ops",
        ":sets",
        ":sparse_ops",
        ":standard_ops",
        ":state_ops",
        ":string_ops",
        ":tensor_array_ops",
        ":training",
        ":uniform_quant_ops_gen",
        ":variable_v1",
        ":weights_broadcast_ops",
        ":while_loop",
        ":while_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/autograph",
        "//tensorflow/python/autograph/utils:testing",
        "//tensorflow/python/client",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/client:pywrap_tf_session",
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
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:_pywrap_python_op_gen",
        "//tensorflow/python/framework:_test_metrics_util",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:subscribe",
        "//tensorflow/python/framework:test_ops",  # TODO(b/183988750): Break testing code out into separate rule.
        "//tensorflow/python/grappler:tf_cluster",
        "//tensorflow/python/grappler:tf_item",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg:linear_operator_test_util",
        "//tensorflow/python/ops/linalg/sparse:sparse_py",
        "//tensorflow/python/ops/losses",
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
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:app",
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
        "//tensorflow/python/training:saver_test_utils",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:data",
        "//tensorflow/python/types:distribute",
        "//tensorflow/python/types:trace",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:all_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:dispatch",
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
py_library(
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
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/distribute:merge_call_interim",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute/failure_handling:failure_handling_lib",
        "//tensorflow/python/distribute/failure_handling:preemption_watcher",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

alias(
    name = "tf_decorator",
    actual = "//tensorflow/python/util:tf_decorator",
)

# Deprecated.  Use
#   //third_party/tensorflow/tsl/python/lib/core:bfloat16_lib
# for the bfloat16-specific dependency, or
#   //third_party/tensorflow/tsl/python/lib/core
# for all custom types and python utilities.
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
    dynamic_deps = [":_pywrap_tensorflow_internal.so"] + select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        "//tensorflow:windows": [],
    }),
    features = ["-layering_check"],
    static_deps = tf_python_pybind_static_deps(),
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

alias(
    name = "pywrap_tf_session",
    actual = "//tensorflow/python/client:pywrap_tf_session",
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

# TODO(yanhuasun): Move this back and the source file back to lib/core directory.
tf_python_pybind_extension(
    name = "_pywrap_py_exception_registry",
    srcs = ["py_exception_registry_wrapper.cc"],
    dynamic_deps = [":_pywrap_tensorflow_internal.so"] + select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        "//tensorflow:windows": [],
    }),
    static_deps = tf_python_pybind_static_deps(),
    deps = [
        ":py_exception_registry",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:py_exception_registry",
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

py_library(
    name = "map_fn",
    srcs = ["ops/map_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":tensor_array_ops",
        ":variable_scope",
        ":while_loop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_library(
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
        "@six_archive//:six",
    ],
)

py_library(
    name = "distributed_framework_test_lib",
    srcs_version = "PY3",
    deps = ["//tensorflow/python/framework:test_lib"],
)

tf_gen_op_wrapper_private_py(
    name = "functional_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/eager/polymorphic_function:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["ops/functional_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":functional_ops_gen",
        ":tensor_array_ops",
        ":variable_scope",
        ":while_loop",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "functional_ops_test",
    srcs = ["ops/functional_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":functional_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:tensor_spec",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "debug_ops_gen",
    out = "ops/gen_debug_ops.py",
    visibility = [
        "//tensorflow/python/debug:__subpackages__",
    ],
    deps = ["//tensorflow/core:debug_ops_op_lib"],
)

tf_gen_op_wrapper_private_py(
    name = "array_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/array_ops:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
        "//tensorflow/python/kernel_tests/v1_compat_tests:__pkg__",
        "//tensorflow/python/training:__pkg__",
        "//tensorflow/security/fuzzing/google:__pkg__",
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
        "//tensorflow/dtensor:dtensor-internal",
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
        "//tensorflow/python/kernel_tests/io_ops:__pkg__",
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
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
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
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
        "//tensorflow/python/kernel_tests/data_structures:__pkg__",
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
    name = "optional_ops_gen",
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
    name = "filesystem_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "image_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests/image_ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "io_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/checkpoint:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/array_ops:__pkg__",
        "//tensorflow/python/kernel_tests/io_ops:__pkg__",
        "//tensorflow/python/trackable:__pkg__",
        "//tensorflow/python/training:__pkg__",
        "//tensorflow/python/training/tracking:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "linalg_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "logging_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/framework:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
        "//tensorflow/python/summary:__pkg__",
        "//tensorflow/python/summary/writer:__pkg__",
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
        "//tensorflow/python/kernel_tests/data_structures:__pkg__",
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
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_test(
    name = "batch_ops_test",
    size = "small",
    srcs = ["ops/batch_ops_test.py"],
    main = "ops/batch_ops_test.py",
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
        ":gradients",
        ":script_ops",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:dtypes",
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
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/math_ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "nn_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/nn_ops:__pkg__",
        "//tensorflow/python/kernel_tests/sparse_ops:__pkg__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/security/fuzzing/google:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "count_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/security/fuzzing/google:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "parsing_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/autograph/operators:__pkg__",
        "//tensorflow/python/data/ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "random_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/eager/polymorphic_function:__pkg__",
        "//tensorflow/python/kernel_tests/random:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "special_math_ops_gen",
    visibility = ["//learning/brain/python/ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "stateful_random_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/kernel_tests/random:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "resource_variable_ops_gen",
    visibility = [
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/distribute:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "stateless_random_ops_gen",
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/data/experimental/ops:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "stateless_random_ops_v2_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
)

tf_gen_op_wrapper_private_py(
    name = "random_index_shuffle_ops_gen",
    visibility = visibility + ["//tensorflow/python/kernel_tests/random:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "list_ops_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
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
    visibility = visibility + ["//tensorflow/python/kernel_tests/math_ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "state_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python/framework:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
        "//tensorflow/python/kernel_tests/variables:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "sparse_ops_gen",
    visibility = visibility + ["//tensorflow/python/kernel_tests/sparse_ops:__pkg__"],
)

tf_gen_op_wrapper_private_py(
    name = "spectral_ops_gen",
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python/ops/signal:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "string_ops_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
)

tf_gen_op_wrapper_private_py(
    name = "training_ops_gen",
    visibility = [
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "composite_tensor_ops_gen",
    visibility = [],
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

tf_gen_op_wrapper_private_py(
    name = "rnn_ops_gen",
    visibility = ["//tensorflow/python/kernel_tests/nn_ops:__pkg__"],
)

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
        "//tensorflow/python/tpu/ops:__pkg__",
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
        "//tensorflow/core:tpu_sharding_util_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "tpu_partition_ops_gen",
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/python/tpu:__pkg__",
        "//tensorflow/python/tpu/ops:__pkg__",
    ],
    deps = [
        "//tensorflow/core/tpu/ops:tpu_partitioned_ops",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "uniform_quant_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [
        "//tensorflow/core:uniform_quant_ops_op_lib",
    ],
)

tf_gen_op_wrapper_private_py(
    name = "sync_ops_gen",
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:sync_ops_op_lib",
    ],
)

py_library(
    name = "array_grad",
    srcs = ["ops/array_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":cond",
        ":control_flow_ops",
        ":control_flow_util",
        ":math_ops",
        ":math_ops_gen",
        ":pywrap_tfe",
        ":resource_variable_ops_gen",
        ":sparse_ops",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
    ],
)

cuda_py_test(
    name = "array_grad_test",
    srcs = ["ops/array_grad_test.py"],
    main = "ops/array_grad_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradient_checker_v2",
        ":gradients",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
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
        ":array_ops_stack",
        ":control_flow_util",
        ":math_ops_gen",
        ":shape_util",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "array_ops_stack",
    srcs = ["ops/array_ops_stack.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops_gen",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "bitwise_ops",
    srcs = ["ops/bitwise_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":bitwise_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "boosted_trees_ops",
    srcs = ["ops/boosted_trees_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":boosted_trees_ops_gen",
        ":ops",
        ":training",
        "//tensorflow/python/framework",
    ],
)

py_library(
    name = "optional_grad",
    srcs = ["ops/optional_grad.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:ops",
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
        ":set_ops_gen",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "candidate_sampling_ops",
    srcs = ["ops/candidate_sampling_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops_gen",
        ":math_ops",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "check_ops",
    srcs = ["ops/check_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clip_ops",
    srcs = ["ops/clip_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops_gen",
        ":numerics",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "clustering_ops",
    srcs = ["ops/clustering_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":clustering_ops_gen",
        ":cond",
        ":ops",
        ":training",
        ":variable_v1",
        "//tensorflow/python/framework",
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
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "collective_ops",
    srcs = ["ops/collective_ops.py"],
    srcs_version = "PY3",
    deps = [":collective_ops_gen"],
)

tf_py_test(
    name = "collective_ops_test",
    size = "small",
    srcs = ["ops/collective_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
    ],
    deps = [
        ":client_testlib",
        ":collective_ops",
        ":variable_v1",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "collective_ops_gpu_test",
    size = "small",
    srcs = ["ops/collective_ops_gpu_test.py"],
    main = "ops/collective_ops_gpu_test.py",
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_gpu",
        "no_windows",
    ],
    deps = [
        ":client_testlib",
        ":collective_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "compiled_collective_ops_gpu_test",
    size = "small",
    srcs = ["ops/compiled_collective_ops_gpu_test.py"],
    main = "ops/compiled_collective_ops_gpu_test.py",
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_gpu",
        "no_windows",
    ],
    deps = [
        ":client_testlib",
        ":collective_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "control_flow_assert",
    srcs = ["ops/control_flow_assert.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_ops_gen",
        ":logging_ops_gen",
        ":math_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
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
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
    ],
)

py_strict_library(
    name = "control_flow_ops",
    srcs = ["ops/control_flow_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":control_flow_ops_gen",
        ":control_flow_util",
        ":math_ops",
        ":tensor_array_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "control_flow_case",
    srcs = ["ops/control_flow_case.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops_stack",
        ":cond",
        ":control_flow_assert",
        ":math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "control_flow_switch_case",
    srcs = ["ops/control_flow_switch_case.py"],
    deps = [
        ":array_ops",
        ":cond_v2",
        ":control_flow_util",
        ":functional_ops_gen",
        ":math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "cond",
    srcs = ["ops/cond.py"],
    deps = [
        ":array_ops",
        ":cond_v2",
        ":control_flow_ops",
        ":control_flow_util",
        ":math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function:eager_function_run",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "while_loop",
    srcs = ["ops/while_loop.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":control_flow_util",
        ":math_ops",
        ":tensor_array_ops",
        ":while_v2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_library(
    name = "control_flow_util",
    srcs = ["ops/control_flow_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "control_flow_util_v2",
    srcs = ["ops/control_flow_util_v2.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//waymo/ml:__subpackages__"],
    deps = [
        ":control_flow_util",
        ":control_flow_v2_func_graphs",
        ":gradients_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function:atomic_function",
        "//tensorflow/python/eager/polymorphic_function:monomorphic_function",
        "//tensorflow/python/eager/polymorphic_function:tracing_compiler",
        "//tensorflow/python/eager/polymorphic_function:transform",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:keras_deps",
        "//tensorflow/python/util:tf_decorator",
    ],
)

py_library(
    name = "control_flow_v2_func_graphs",
    srcs = ["ops/control_flow_v2_func_graphs.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:func_graph",
    ],
)

py_library(
    name = "control_flow_v2_toggles",
    srcs = ["ops/control_flow_v2_toggles.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        ":control_flow_util_v2",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
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
        ":control_flow_util_v2",
        ":functional_ops_gen",
        ":gradients_util",
        ":handle_data_util",
        ":optional_ops_gen",
        ":pywrap_tensorflow",
        "//tensorflow/python/compat",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:c_api_util",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:nest",
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
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":functional_ops_gen",
        ":gradients_util",
        ":handle_data_util",
        ":list_ops",
        ":math_ops",
        ":tensor_array_ops",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_library(
    name = "bincount_ops",
    srcs = ["ops/bincount_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":count_ops_gen",
        "//tensorflow/python/compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_test(
    name = "bincount_ops_test",
    size = "small",
    srcs = ["ops/bincount_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":bincount_ops",
        ":platform_test",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "ctc_ops",
    srcs = ["ops/ctc_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":ctc_ops_gen",
        ":functional_ops",
        ":nn_grad",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "cudnn_rnn_grad",
    srcs = ["ops/cudnn_rnn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":cudnn_rnn_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "data_flow_grad",
    srcs = ["ops/data_flow_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":data_flow_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
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
        ":math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
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
        ":math_ops",
        ":resource_variable_ops",
        ":sparse_ops",
        ":variables",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_functional_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "handle_data_util",
    srcs = [
        "ops/handle_data_util.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
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
        ":math_ops",
        ":unconnected_gradients",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:forwardprop",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
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
        ":control_flow_util",
        ":image_grad",
        ":linalg_grad",
        ":linalg_ops",
        ":logging_ops",
        ":manip_grad",
        ":manip_ops",
        ":math_grad",
        ":math_ops",
        ":optional_grad",
        ":random_grad",
        ":tensor_array_ops",
        ":unconnected_gradients",
        ":while_loop",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
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
        ":functional_ops_gen",
        ":math_ops",
        ":resource_variable_ops",
        ":unconnected_gradients",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "default_gradient",
    srcs = [
        "ops/default_gradient.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":resource_variable_ops",
        "//tensorflow/python/framework:dtypes",
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
        ":control_flow_ops",
        ":control_flow_util",
        ":data_flow_ops_gen",
        ":resource_variable_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_library(
    name = "unconnected_gradients",
    srcs = ["ops/unconnected_gradients.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "filesystem_ops",
    srcs = ["ops/filesystem_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":filesystem_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "histogram_ops",
    srcs = ["ops/histogram_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "image_grad",
    srcs = ["ops/image_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":image_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
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
        ":cond",
        ":control_flow_case",
        ":control_flow_ops",
        ":gradients",
        ":image_ops_gen",
        ":math_ops",
        ":nn",
        ":nn_ops_gen",
        ":random_ops",
        ":string_ops",
        ":variables",
        ":while_loop",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "init_ops",
    srcs = ["ops/init_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":math_ops",
        ":random_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "init_ops_v2",
    srcs = ["ops/init_ops_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":math_ops",
        ":random_ops",
        ":stateless_random_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/util:tf_export",
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

py_strict_library(
    name = "io_ops",
    srcs = ["ops/io_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":data_flow_ops_gen",
        ":io_ops_gen",
        ":lib",
        ":parsing_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "linalg_grad",
    srcs = ["ops/linalg_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_ops",
        ":linalg_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/linalg:linalg_impl",
    ],
)

py_library(
    name = "linalg_ops",
    srcs = ["ops/linalg_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":map_fn",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "linalg_ops_impl",
    srcs = ["ops/linalg_ops_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "manip_grad",
    srcs = ["ops/manip_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":manip_ops",
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "manip_ops",
    srcs = ["ops/manip_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":manip_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "logging_ops",
    srcs = ["ops/logging_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":logging_ops_gen",
        ":string_ops",
        "//tensorflow/python/compat",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "lookup_ops",
    srcs = ["ops/lookup_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":lookup_ops_gen",
        ":math_ops",
        ":string_ops",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "math_grad",
    srcs = ["ops/math_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":math_ops",
        ":math_ops_gen",
        ":special_math_ops",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "op_selector",
    srcs = ["ops/op_selector.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:object_identity",
    ],
)

py_library(
    name = "math_ops",
    srcs = ["ops/math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":bitwise_ops_gen",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":math_ops_gen",
        ":nn_ops_gen",
        ":sparse_ops_gen",
        ":state_ops",
        ":state_ops_gen",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:traceback_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "math_ops_extra",
    srcs = [
        "ops/math_ops_extra.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":math_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "resources",
    srcs = ["ops/resources.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
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
        ":handle_data_util",
        ":resource_variable_ops_gen",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/checkpoint:tensor_callable",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "critical_section_ops",
    srcs = ["ops/critical_section_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":resource_variable_ops_gen",
        ":tensor_array_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/util:lazy_loader",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "map_ops",
    srcs = ["ops/map_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
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
        ":cond",
        ":ctc_ops",
        ":embedding_ops",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops",
        ":variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:device_context",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "nn_grad",
    srcs = ["ops/nn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients",
        ":math_ops",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_library(
    name = "nn_ops",
    srcs = ["ops/nn_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops_gen",
        ":random_ops",
        ":stateless_random_ops",
        ":variables",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:device_context",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "parsing_config",
    srcs = ["ops/parsing_config.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_math_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "parsing_ops",
    srcs = ["ops/parsing_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":math_ops",
        ":parsing_config",
        ":parsing_ops_gen",
        ":sparse_ops",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "partitioned_variables",
    srcs = ["ops/partitioned_variables.py"],
    srcs_version = "PY3",
    deps = [
        ":variable_scope",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "random_grad",
    srcs = ["ops/random_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":random_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "random_ops",
    srcs = ["ops/random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":math_ops",
        ":random_ops_gen",
        ":shape_util",
        ":stateless_random_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "random_crop_ops",
    srcs = ["ops/random_crop_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_assert",
        ":control_flow_ops",
        ":math_ops",
        ":random_ops",
        ":stateless_random_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "stateful_random_ops",
    srcs = ["ops/stateful_random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":math_ops",
        ":stateful_random_ops_gen",
        ":stateless_random_ops",
        ":stateless_random_ops_v2_gen",
        ":variables",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "stateless_random_ops",
    srcs = ["ops/stateless_random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":bitwise_ops",
        ":math_ops",
        ":random_index_shuffle_ops_gen",
        ":shape_util",
        ":stateless_random_ops_gen",
        ":stateless_random_ops_v2_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "rnn",
    srcs = ["ops/rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":math_ops",
        ":rnn_cell",
        ":tensor_array_ops",
        ":variable_scope",
        ":while_loop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_impl",
        "//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_wrapper_impl",
    ],
)

py_library(
    name = "script_ops",
    srcs = [
        "ops/autograph_ops.py",
        "ops/script_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":script_ops_gen",
        "//tensorflow/python/autograph/operators:py_builtins",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/lib/core:_pywrap_py_func",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sdca_ops",
    srcs = ["ops/sdca_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":sdca_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "session_ops",
    srcs = ["ops/session_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "shape_util",
    srcs = ["ops/shape_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_library(
    name = "sparse_grad",
    srcs = ["ops/sparse_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":sparse_ops",
        ":sparse_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
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
        ":math_ops",
        ":sparse_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_ops_test",
    srcs = ["ops/sparse_ops_test.py"],
    main = "ops/sparse_ops_test.py",
    python_version = "PY3",
    deps = [
        ":array_grad",
        ":gradient_checker_v2",
        ":sparse_grad",
        ":sparse_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "sort_ops",
    srcs = ["ops/sort_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sort_ops_test",
    srcs = ["ops/sort_ops_test.py"],
    main = "ops/sort_ops_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":random_ops",
        ":sort_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
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
        ":cond",
        ":control_flow_ops",
        ":math_ops",
        ":sparse_ops",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
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
        ":cond",
        ":control_flow_ops",
        ":math_ops",
        ":sets",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:tf_export",
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
        ":cond",
        ":confusion_matrix",
        ":control_flow_ops",
        ":math_ops",
        ":nn",
        ":sets",
        ":sparse_ops",
        ":state_ops",
        ":variable_scope",
        ":variable_v1",
        ":variables",
        ":weights_broadcast_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
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
        ":math_ops",
        ":special_math_ops_gen",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "@opt_einsum_archive//:opt_einsum",
    ],
)

py_library(
    name = "rnn_grad",
    srcs = ["ops/rnn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_test(
    name = "rnn_grad_test",
    srcs = ["ops/rnn_grad_test.py"],
    main = "ops/rnn_grad_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradients",
        ":math_ops",
        ":rnn_grad",
        ":rnn_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "script_ops_test",
    srcs = ["ops/script_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":script_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "autograph_ops_test",
    srcs = ["ops/autograph_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":script_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
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
        ":cond",
        ":confusion_matrix",
        ":control_flow_case",
        ":control_flow_ops",
        ":critical_section_ops",
        ":cudnn_rnn_grad",
        ":data_flow_grad",
        ":data_flow_ops",
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
        ":math_ops_extra",
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
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/structured",
    ],
)

py_library(
    name = "state_grad",
    srcs = ["ops/state_grad.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/framework:ops"],
)

py_library(
    name = "state_ops",
    srcs = ["ops/state_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops_gen",
        ":resource_variable_ops_gen",
        ":state_ops_gen",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "string_ops",
    srcs = ["ops/string_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":string_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "summary_ops_v2",
    srcs = ["ops/summary_ops_v2.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":math_ops",
        ":resource_variable_ops",
        ":resource_variable_ops_gen",
        ":summary_op_util",
        ":summary_ops_gen",
        ":training_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:profiler",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:smart_cond",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "template",
    srcs = ["ops/template.py"],
    srcs_version = "PY3",
    deps = [
        ":variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "tensor_array_grad",
    srcs = ["ops/tensor_array_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_array_ops",
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "tensor_array_ops",
    srcs = ["ops/tensor_array_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops_gen",
        ":data_flow_ops_gen",
        ":list_ops",
        ":math_ops",
        ":tf2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
    ],
)

py_library(
    name = "composite_tensor_ops",
    srcs = ["ops/composite_tensor_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "composite_tensor_ops_gen",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "variable_scope",
    srcs = ["ops/variable_scope.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":init_ops",
        ":resource_variable_ops",
        ":tf2",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "variables",
    srcs = ["ops/variables.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":control_flow_ops",
        ":math_ops",
        ":math_ops_gen",
        ":pywrap_tensorflow",
        ":state_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
        "//tensorflow/python/util:traceback_utils",
    ],
)

py_library(
    name = "ref_variable",
    srcs = ["ops/ref_variable.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":resource_variable_ops",
        ":state_ops",
        ":state_ops_gen",
        ":variable_scope",
        ":variable_v1",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
    ],
)

py_library(
    name = "variable_v1",
    srcs = ["ops/variable_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":cond",
        ":state_ops",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
    ],
)

py_library(
    name = "gradient_checker",
    srcs = ["ops/gradient_checker.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "gradient_checker_v2",
    srcs = ["ops/gradient_checker_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "bitwise_ops_test",
    size = "small",
    srcs = ["ops/bitwise_ops_test.py"],
    main = "ops/bitwise_ops_test.py",
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":bitwise_ops",
        ":bitwise_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

cuda_py_test(
    name = "control_flow_ops_test",
    size = "medium",  # Contains artificial waits.
    srcs = ["ops/control_flow_ops_test.py"],
    main = "ops/control_flow_ops_test.py",
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
        ":cond",
        ":cond_v2",
        ":control_flow_case",
        ":control_flow_ops",
        ":control_flow_v2_toggles",
        ":embedding_ops",
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
        ":while_loop",
        ":while_v2",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/util:nest",
    ],
)

py_test(
    name = "op_selector_test",
    srcs = ["ops/op_selector_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":math_ops",
        ":op_selector",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_test(
    name = "embedding_ops_test",
    srcs = ["ops/embedding_ops_test.py"],
    main = "ops/embedding_ops_test.py",
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    deps = [
        ":embedding_ops",
        ":gradients",
        ":resource_variable_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gradient_checker_v2_test",
    size = "medium",
    srcs = ["ops/gradient_checker_v2_test.py"],
    main = "ops/gradient_checker_v2_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradient_checker_v2",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["ops/gradients_test.py"],
    main = "ops/gradients_test.py",
    python_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":cond",
        ":control_flow_grad",
        ":control_flow_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":functional_ops",
        ":gradient_checker_v2",
        ":gradients",
        ":init_ops",
        ":list_ops",
        ":map_ops",
        ":math_grad",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":platform_test",
        ":ref_variable",
        ":state_grad",
        ":state_ops",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":unconnected_gradients",
        ":variable_scope",
        ":variable_v1",
        ":while_loop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "image_grad_d9m_test",
    size = "large",
    srcs = ["ops/image_grad_d9m_test.py"],
    main = "ops/image_grad_d9m_test.py",
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":image_grad_test_base",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_test(
    name = "image_grad_test",
    size = "large",
    srcs = ["ops/image_grad_test.py"],
    main = "ops/image_grad_test.py",
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":image_grad_test_base",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "image_grad_test_base",
    srcs = ["ops/image_grad_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":errors",
        ":gradient_checker_v2",
        ":gradients",
        ":image_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["ops/image_ops_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    main = "ops/image_ops_test.py",
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "no_cuda_asan",  # TODO(b/171511582): re-enable.
        "no_windows",  #TODO(b/207035199): re-enable
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":config",
        ":control_flow_ops",
        ":image_ops",
        ":io_ops",
        ":math_ops",
        ":platform_test",
        ":random_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "init_ops_test",
    size = "small",
    srcs = ["ops/init_ops_test.py"],
    main = "ops/init_ops_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":init_ops",
        ":resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "init_ops_v2_test",
    size = "medium",
    srcs = ["ops/init_ops_v2_test.py"],
    main = "ops/init_ops_v2_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":init_ops_v2",
        ":random_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_grad_test",
    size = "small",
    srcs = ["ops/math_grad_test.py"],
    main = "ops/math_grad_test.py",
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradient_checker_v2",
        ":math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_ops_test",
    size = "medium",
    srcs = ["ops/math_ops_test.py"],
    main = "ops/math_ops_test.py",
    python_version = "PY3",
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        ":array_ops",
        ":gradients",
        ":math_ops",
        ":math_ops_extra",
        ":platform_test",
        ":variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "math_ops_linspace_test",
    size = "medium",
    srcs = ["ops/math_ops_linspace_test.py"],
    main = "ops/math_ops_linspace_test.py",
    python_version = "PY3",
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        ":math_ops",
        ":platform_test",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_batchnorm_test",
    size = "medium",
    srcs = ["ops/nn_batchnorm_test.py"],
    main = "ops/nn_batchnorm_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_mac_arm64",
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_fused_batchnorm_d9m_test",
    size = "medium",
    srcs = ["ops/nn_fused_batchnorm_d9m_test.py"],
    main = "ops/nn_fused_batchnorm_d9m_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
    ],
    deps = [
        ":client_testlib",
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":nn",
        ":nn_grad",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_fused_batchnorm_test",
    size = "large",
    srcs = ["ops/nn_fused_batchnorm_test.py"],
    main = "ops/nn_fused_batchnorm_test.py",
    python_version = "PY3",
    shard_count = 24,
    tags = ["no_rocm"],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_test",
    size = "medium",
    srcs = ["ops/nn_test.py"],
    main = "ops/nn_test.py",
    python_version = "PY3",
    tags = ["no_windows"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradient_checker_v2",
        ":nn",
        ":nn_grad",
        ":nn_ops",
        ":partitioned_variables",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "nn_xent_test",
    size = "medium",
    srcs = ["ops/nn_xent_test.py"],
    main = "ops/nn_xent_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":gradients",
        ":nn",
        ":nn_grad",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
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
        ":client_testlib",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
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
    main = "ops/sobol_ops_test.py",
    tags = [
        "no_oss",  # TODO(b/149565560)
        "no_windows_gpu",
    ],
    deps = [
        ":math_ops",
        ":platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "special_math_ops_test",
    size = "medium",
    srcs = ["ops/special_math_ops_test.py"],
    main = "ops/special_math_ops_test.py",
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradient_checker_v2",
        ":math_ops",
        ":special_math_ops",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:benchmark",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "variable_spec_test",
    size = "small",
    srcs = ["ops/variable_spec_test.py"],
    python_version = "PY3",
    deps = [
        ":platform_test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:fast_tensor_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

alias(
    name = "client",
    actual = "//tensorflow/python/client:client",
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

alias(
    name = "device_lib",
    actual = "//tensorflow/python/client:device_lib",
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
    deps = [
        ":pywrap_tensorflow_internal",
        "//tensorflow/python/platform:self_check",
    ],
)

pywrap_tensorflow_macro(
    name = "pywrap_tensorflow_internal",
    srcs = ["pywrap_tensorflow_internal.cc"],
    dynamic_deps = ["//tensorflow/tsl/python/lib/core:bfloat16.so"] + select({
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
    ] + if_windows([
        "//:__subpackages__",
        "@nsync//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@double_conversion//:__subpackages__",
        "@eigen_archive//:__subpackages__",
    ]),
    roots = [
        ":py_exception_registry",
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
        "@cudnn_frontend_archive//:__subpackages__",
        "@curl//:__subpackages__",
        "@dlpack//:__subpackages__",
        "@double_conversion//:__subpackages__",
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
        "@mkl_dnn_v1//:__subpackages__",
        "@nccl_archive//:__subpackages__",
        "@nsync//:__subpackages__",
        "@nvtx_archive//:__subpackages__",
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
    ] + tsl_async_value_deps() + if_onednn_v3(["@onednn_v3//:__subpackages__"]),
    win_def_file = ":pywrap_tensorflow_filtered_def_file",
    deps = [
        ":safe_pyobject_ptr",
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
        "//tensorflow/compiler/mlir/python:mlir",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_cc",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/compiler/tf2tensorrt:op_converter_registry_impl",
        "//tensorflow/compiler/tf2xla:tf2xla_opset",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_impl",
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
        "//tensorflow/core/profiler/backends/cpu:python_tracer",
        "//tensorflow/core/profiler/internal:print_model_analysis",
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
        "//tensorflow/tsl/profiler/rpc:profiler_server_impl",
        "//tensorflow/tsl/profiler/rpc/client:profiler_client_impl",
        "//tensorflow/tsl/python/lib/core",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ] + if_static([
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core/platform:enable_tf2_utils",
    ]) + if_google([
        "//base",
        "//tensorflow/core/util/tensor_bundle",
    ]) + if_oss(if_static(
        [clean_dep("@nsync//:nsync_headers")],
    )) + if_oss([
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_session",
    ]) + if_xla_available([
        "//tensorflow/compiler/aot:tfcompile_lib",
    ]) + tf_monitoring_python_deps() + tf_additional_plugin_deps() + tf_additional_profiler_deps() + tf_additional_binary_deps(),
)

# ** Targets for Windows build (start) **
# We need the following targets to expose symbols from _pywrap_tensorflow.dll

filegroup(
    name = "win_lib_files_for_exported_symbols",
    srcs = [
        ":ndarray_tensor",  # checkpoint_reader
        ":py_exception_registry",  # py_exception_registry
        ":safe_pyobject_ptr",  # checkpoint_reader
        "//tensorflow/c:checkpoint_reader",  # checkpoint_reader
        "//tensorflow/c:python_api",  # tf_session
        "//tensorflow/c:safe_ptr",  # checkpoint_reader
        "//tensorflow/c:tf_status_helper",  # tfe
        "//tensorflow/cc/saved_model:fingerprinting_impl",  # SavedModel fingerprinting
        "//tensorflow/cc/saved_model:metrics_impl",  # SavedModel metrics
        "//tensorflow/compiler/jit:flags",  # tfe
        "//tensorflow/compiler/jit:get_compiler_ir",  # tfe
        "//tensorflow/compiler/mlir/python:mlir",  # mlir
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:quantize_model_cc_impl",  # quantization
        "//tensorflow/compiler/tf2xla:tf2xla_opset",  # pywrap_xla_ops
        "//tensorflow/compiler/xla/stream_executor:stream_executor_pimpl",  # stat_summarizer
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
        "//tensorflow/core/profiler/backends/cpu:traceme_recorder_impl",  # profiler
        "//tensorflow/core/profiler/internal:print_model_analysis",  # tfprof
        "//tensorflow/core/profiler/rpc/client:profiler_client_impl",  # profiler
        "//tensorflow/core/util:determinism",  # determinism
        "//tensorflow/core/util:port",  # util_port
        "//tensorflow/core/util/tensor_bundle",  # checkpoint_reader
        "//tensorflow/dtensor/cc:dtensor_device_cc",  # DTensor
        "//tensorflow/dtensor/cc:tensor_layout",  # DTensor
        "//tensorflow/lite/kernels/shim:shape",  # tf_text
        "//tensorflow/lite/kernels/shim:tf_op_shim",  # tf_text
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
        "//tensorflow/python/lib/core:py_func_lib",
        "//tensorflow/python/util:cpp_nest",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:kernel_registry",
        "//tensorflow/tools/graph_transforms:transform_graph_lib",  # transform_graph
        "//tensorflow/tsl/platform:tensor_float_32_utils",  # tensor_float_32
        "//tensorflow/tsl/profiler/lib:profiler_session_impl",  # profiler
        "//tensorflow/tsl/profiler/rpc:profiler_server_impl",  # profiler
        "//tensorflow/tsl/profiler/rpc/client:profiler_client_impl",
        "//tensorflow/tsl/python/lib/core:bfloat16_lib",  # bfloat16
        "//tensorflow/tsl/python/lib/core:custom_casts_lib",
        "//tensorflow/tsl/python/lib/core:float8_lib",  # float8_e4m3fn, float8_e5m2
        "//tensorflow/tsl/python/lib/core:numpy",  # checkpoint_reader
    ] + if_xla_available([
        "//tensorflow/compiler/aot:tfcompile_lib",  # tfcompile
        "//tensorflow/compiler/xla:status_macros",  # tfcompile
        "//tensorflow/compiler/xla/hlo/ir:hlo",  # tfcompile
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

# Get the import library of _pywrap_tensorflow_internal.pyd, platform-specific to Windows.
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

alias(
    name = "session",
    actual = "//tensorflow/python/client:session",
)

alias(
    name = "timeline",
    actual = "//tensorflow/python/client:timeline",
    visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "summary_op_util",
    srcs = ["ops/summary_op_util.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
    ],
)

alias(
    name = "fake_summary_writer",
    actual = "//tensorflow/python/summary/writer:fake_summary_writer",
    visibility = ["//visibility:public"],
)

# -----------------------------------------------------------------------------
# Quantization

tf_py_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["ops/dequantize_op_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "quantized_ops_test",
    size = "small",
    srcs = ["ops/quantized_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["ops/quantized_conv_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":client_testlib",
        ":nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
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
        ":gradients",
        ":math_ops",
        ":random_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
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
        ":client_testlib",
        ":control_flow_ops_gen",
        ":data_flow_ops",
        ":math_ops",
        ":random_ops",
        ":state_ops",
        ":state_ops_gen",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

cuda_py_test(
    name = "batch_norm_benchmark",
    srcs = ["ops/batch_norm_benchmark.py"],
    main = "ops/batch_norm_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":gradients",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_ops_gen",
        ":random_ops",
        ":variables",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
    ],
)

cuda_py_test(
    name = "collective_ops_benchmark",
    srcs = ["ops/collective_ops_benchmark.py"],
    main = "ops/collective_ops_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":collective_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_test(
    name = "concat_benchmark",
    srcs = ["ops/concat_benchmark.py"],
    main = "ops/concat_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":client_testlib",
        ":control_flow_ops",
        ":gradients",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_test(
    name = "control_flow_ops_benchmark",
    srcs = ["ops/control_flow_ops_benchmark.py"],
    main = "ops/control_flow_ops_benchmark.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":cond",
        ":control_flow_ops",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_test(
    name = "conv2d_benchmark",
    size = "large",
    srcs = ["ops/conv2d_benchmark.py"],
    main = "ops/conv2d_benchmark.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":control_flow_ops",
        ":nn_ops",
        ":platform_benchmark",
        ":random_ops",
        ":variable_v1",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:flags",
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
        ":client_testlib",
        ":control_flow_ops",
        ":platform_benchmark",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
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
        ":client_testlib",
        ":control_flow_ops",
        ":platform_benchmark",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "matmul_benchmark",
    size = "medium",
    srcs = ["ops/matmul_benchmark.py"],
    main = "ops/matmul_benchmark.py",
    python_version = "PY3",
    deps = [
        ":matmul_benchmark_main_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_library(
    name = "matmul_benchmark_main_lib",
    testonly = True,
    srcs = ["ops/matmul_benchmark.py"],
    srcs_version = "PY3",
    deps = [
        ":client_testlib",
        ":control_flow_ops",
        ":math_ops",
        ":platform_benchmark",
        ":random_ops",
        ":variable_v1",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "nn_grad_test",
    size = "medium",
    srcs = ["ops/nn_grad_test.py"],
    main = "ops/nn_grad_test.py",
    python_version = "PY3",
    deps = [
        ":client_testlib",
        ":gradient_checker_v2",
        ":nn_grad",
        ":nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
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
        ":nccl_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_test(
    name = "nccl_ops_test",
    size = "small",
    srcs = ["ops/nccl_ops_test.py"],
    main = "ops/nccl_ops_test.py",
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
        ":nccl_ops",
        ":platform_test",
        "//tensorflow/python/framework:test_lib",
    ],
)

cuda_py_test(
    name = "factory_ops_test",
    size = "small",
    srcs = ["ops/factory_ops_test.py"],
    main = "ops/factory_ops_test.py",
    deps = [
        ":platform_test",
        ":sparse_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
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

py_library(
    name = "proto_ops",
    srcs = ["ops/proto_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":decode_proto_ops_gen",
        ":encode_proto_ops_gen",
        "//tensorflow/python/framework:ops",
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
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "//tensorflow/compiler/mlir/python:pywrap_mlir_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

py_library(
    name = "pywrap_sanitizers",
    srcs = ["pywrap_sanitizers.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_sanitizers",
        ":pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "flags_pybind",
    srcs = ["//tensorflow/core/config:flags_api_wrapper.cc"],
    visibility = ["//tensorflow/core/config:__subpackages__"],
    deps = [
        ":pybind11_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/config:flags_headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_sanitizers",
    srcs = ["sanitizers_wrapper.cc"],
    deps = [
        "@pybind11",
    ],
)

cc_library(
    name = "unified_api_pywrap_required_headers",
    textual_hdrs = [
        "//tensorflow/python/lib/core:basic_hdrs",
        "//tensorflow/tsl/python/lib/core:basic_hdrs",
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
        "//tensorflow/tsl/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/tsl/python/lib/core:numpy_hdr",
    ],
    dynamic_deps = [":_pywrap_tensorflow_internal.so"] + select({
        "//tensorflow:macos": ["//tensorflow:libtensorflow_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//tensorflow:libtensorflow_framework.so.%s" % VERSION],
        "//tensorflow:windows": [],
    }),
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
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
            "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
            "//tensorflow/tsl/protobuf:coordination_service_proto_cc_headers_only",
        ],
    ),
)

py_library(
    name = "pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = ["pywrap_tfe_monitoring_reader.py"],
    srcs_version = "PY3",
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
    visibility = ["//tensorflow/python/distribute/parallel_device:__pkg__"],
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
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_dtensor_device",
    srcs = ["pywrap_dtensor_device.cc"],
    features = ["-layering_check"],
    deps = [
        ":pywrap_densor_device_headers",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status_headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
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
        "//tensorflow/dtensor/cc:dtensor_device_cc",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/python/eager:pywrap_tfe_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
    ],
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

cuda_py_test(
    name = "raw_ops_test",
    srcs = ["ops/raw_ops_test.py"],
    main = "ops/raw_ops_test.py",
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
    name = "safe_pyobject_ptr",
    actual = "//tensorflow/python/lib/core:safe_pyobject_ptr",
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
    name = "python_op_gen_main",
    actual = "//tensorflow/python/framework:python_op_gen_main",
    visibility = ["//visibility:public"],
)

alias(
    name = "framework",
    actual = "//tensorflow/python/framework",
)

alias(
    name = "c_api_util",
    actual = "//tensorflow/python/framework:c_api_util",
)

alias(
    name = "constant_op",
    actual = "//tensorflow/python/framework:constant_op",
)

alias(
    name = "device_spec",
    actual = "//tensorflow/python/framework:device_spec",
)

alias(
    name = "device",
    actual = "//tensorflow/python/framework:device",
)

alias(
    name = "_dtypes",
    actual = "//tensorflow/python/framework:_dtypes",
)

alias(
    name = "dtypes",
    actual = "//tensorflow/python/framework:dtypes",
    visibility = visibility + ["//smartass:__subpackages__"],
)

alias(
    name = "errors",
    actual = "//tensorflow/python/framework:errors",
)

alias(
    name = "error_interpolation",
    actual = "//tensorflow/python/framework:error_interpolation",
)

alias(
    name = "function",
    actual = "//tensorflow/python/framework:function",
    visibility = visibility + ["//smartass/brain:__subpackages__"],
)

alias(
    name = "graph_to_function_def",
    actual = "//tensorflow/python/framework:graph_to_function_def",
)

alias(
    name = "function_def_to_graph",
    actual = "//tensorflow/python/framework:function_def_to_graph",
    visibility = visibility + ["//waymo/ml:__subpackages__"],
)

alias(
    name = "graph_util",
    actual = "//tensorflow/python/framework:graph_util",
)

alias(
    name = "convert_to_constants",
    actual = "//tensorflow/python/framework:convert_to_constants",
    visibility = visibility + ["//waymo/ml:__subpackages__"],
)

alias(
    name = "kernels",
    actual = "//tensorflow/python/framework:kernels",
)

alias(
    name = "op_def_registry",
    actual = "//tensorflow/python/framework:op_def_registry",
)

alias(
    name = "framework_ops",
    actual = "//tensorflow/python/framework:ops",
)

alias(
    name = "framework_indexed_slices",
    actual = "//tensorflow/python/framework:indexed_slices",
)

alias(
    name = "framework_for_generated_wrappers",
    actual = "//tensorflow/python/framework:for_generated_wrappers",
    visibility = ["//visibility:public"],
)

alias(
    name = "framework_for_generated_wrappers_v2",
    actual = "//tensorflow/python/framework:for_generated_wrappers_v2",
    visibility = ["//visibility:public"],
)

alias(
    name = "op_callbacks",
    actual = "//tensorflow/python/framework:op_callbacks",
)

alias(
    name = "indexed_slices",
    actual = "//tensorflow/python/framework:indexed_slices",
)

alias(
    name = "func_graph",
    actual = "//tensorflow/python/framework:func_graph",
)

alias(
    name = "auto_control_deps",
    actual = "//tensorflow/python/framework:auto_control_deps",
)

alias(
    name = "config",
    actual = "//tensorflow/python/framework:config",
)

alias(
    name = "random_seed",
    actual = "//tensorflow/python/framework:random_seed",
)

alias(
    name = "smart_cond",
    actual = "//tensorflow/python/framework:smart_cond",
)

alias(
    name = "sparse_tensor",
    actual = "//tensorflow/python/framework:sparse_tensor",
)

alias(
    name = "composite_tensor",
    actual = "//tensorflow/python/framework:composite_tensor",
    visibility = visibility,
)

alias(
    name = "graph_building_benchmark",
    actual = "//tensorflow/python/framework:graph_building_benchmark",
)

alias(
    name = "tensor_shape",
    actual = "//tensorflow/python/framework:tensor_shape",
    visibility = visibility + ["//smartass/brain/configure/python:__pkg__"],
)

alias(
    name = "type_spec",
    actual = "//tensorflow/python/framework:type_spec",
    visibility = visibility,
)

alias(
    name = "tensor_spec",
    actual = "//tensorflow/python/framework:tensor_spec",
)

alias(
    name = "tensor_util",
    actual = "//tensorflow/python/framework:tensor_util",
)

alias(
    name = "versions",
    actual = "//tensorflow/python/framework:versions",
)

alias(
    name = "framework_test_lib",
    actual = "//tensorflow/python/framework:test_lib",
    visibility = visibility + [
        "//tensorflow_estimator/python/estimator:__subpackages__",
    ],
)

alias(
    name = "is_mlir_bridge_test_true",
    actual = "//tensorflow/python/framework:is_mlir_bridge_test_true",
    visibility = visibility + ["//tensorflow/compiler/tests:friends"],
)

alias(
    name = "tfrt_utils",
    actual = "//tensorflow/python/framework:tfrt_utils",
)

alias(
    name = "framework_combinations",
    actual = "//tensorflow/python/framework:combinations",
)

alias(
    name = "framework_importer_test",
    actual = "//tensorflow/python/framework:importer_test",
)

alias(
    name = "framework_test_combinations_lib",
    actual = "//tensorflow/python/framework:test_combinations_lib",
)

alias(
    name = "framework_kernels_test",
    actual = "//tensorflow/python/framework:kernels_test",
)

alias(
    name = "framework_ops_test",
    actual = "//tensorflow/python/framework:ops_test",
)

alias(
    name = "framework_test_util_test",
    actual = "//tensorflow/python/framework:test_util_test",
)

alias(
    name = "function_def_to_graph_test",
    actual = "//tensorflow/python/framework:function_def_to_graph_test",
)

alias(
    name = "op_def_library_test",
    actual = "//tensorflow/python/framework:op_def_library_test",
)

alias(
    name = "config_test",
    actual = "//tensorflow/python/framework:config_test",
)

alias(
    name = "is_mlir_bridge_test_false",
    actual = "//tensorflow/python/framework:is_mlir_bridge_test_false",
    visibility = visibility + ["//tensorflow/compiler/tests:friends"],
)

alias(
    name = "memory_checker",
    actual = "//tensorflow/python/framework:memory_checker",
)

alias(
    name = "test_ops",
    actual = "//tensorflow/python/framework:test_ops",
)

alias(
    name = "test_ops_kernels",
    actual = "//tensorflow/python/framework:test_ops_kernels",
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
    name = "checkpoint",
    actual = "//tensorflow/python/checkpoint:checkpoint",
)

alias(
    name = "checkpoint_lib",
    actual = "//tensorflow/python/checkpoint:checkpoint_lib",
)

alias(
    name = "trackable",
    actual = "//tensorflow/python/trackable:trackable",
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
    name = "tf_optimizer",
    actual = "//tensorflow/python/grappler:tf_optimizer",
    visibility = ["//visibility:public"],
)

alias(
    name = "ops",
    actual = "//tensorflow/python/user_ops:ops",
)

alias(
    name = "layers_base",
    actual = "//tensorflow/python/layers:layers_base",
)

alias(
    name = "layers_util",
    actual = "//tensorflow/python/layers:layers_util",
)

alias(
    name = "layers",
    actual = "//tensorflow/python/layers:layers",
)

py_library(
    name = "dtensor",
    deps = [
        "//tensorflow/dtensor/python:dtensor",
    ],
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
# # Special build rule for jax2tf.
# # TODO(b/239052279): remove once TF->TFLite dependency cycle is resolved
# py_library(
#     name = "jax2tf_support",
#     visibility = ["//third_party/py/jax/experimental/jax2tf:__subpackages__"],
#     deps = [
#         "//tensorflow/python/dlpack",
#         "//tensorflow/python/ops/ragged:ragged_array_ops",
#         "//tensorflow/python/saved_model",
#     ],
# )
# copybara:uncomment_end
