load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_strict_wrapper_private_py")
load(
    "//tensorflow/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

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
    "//third_party/py/scenic:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

py_strict_library(
    name = "map_fn",
    srcs = ["map_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":tensor_array_ops",
        ":variable_scope",
        ":while_loop",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "functional_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/eager/polymorphic_function:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
    ],
)

py_strict_library(
    name = "functional_ops",
    srcs = ["functional_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":functional_ops_gen",
        ":math_ops",
        ":tensor_array_ops",
        ":variable_scope",
        ":while_loop",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_test(
    name = "functional_ops_test",
    srcs = ["functional_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":functional_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "debug_ops_gen",
    out = "gen_debug_ops.py",
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/debug:__subpackages__",
    ],
    deps = ["//tensorflow/core:debug_ops_op_lib"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "array_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/mlir/tfr:__pkg__",
        "//tensorflow/compiler/mlir/tfr/examples/customization:__pkg__",
        "//tensorflow/compiler/mlir/tfr/examples/pad:__pkg__",
        "//tensorflow/compiler/mlir/tfr/python/experimental:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/opt:__pkg__",
        "//tensorflow/contrib/recurrent:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
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

tf_gen_op_strict_wrapper_private_py(
    name = "bitwise_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/quantization:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "boosted_trees_ops_gen",
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:boosted_trees_ops_op_lib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "summary_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = ["//tensorflow/core:summary_ops_op_lib"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "audio_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/examples/speech_commands:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "cudnn_rnn_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "candidate_sampling_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "checkpoint_ops_gen",
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/io_ops:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "clustering_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:clustering_ops_op_lib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "collective_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:collective_ops_op_lib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
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

tf_gen_op_strict_wrapper_private_py(
    name = "ctc_ops_gen",
    visibility = ["//tensorflow/python:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "data_flow_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
        "//tensorflow/python/kernel_tests/data_structures:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "dataset_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "optional_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "experimental_dataset_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "filesystem_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "image_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/dtensor/python/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests/image_ops:__pkg__",
        "//tensorflow/python/ops/parallel_for:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "io_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/checkpoint:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/array_ops:__pkg__",
        "//tensorflow/python/kernel_tests/io_ops:__pkg__",
        "//tensorflow/python/trackable:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "linalg_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "logging_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python:__pkg__",
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

tf_gen_op_strict_wrapper_private_py(
    name = "lookup_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/data_structures:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "batch_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_strict_library(
    name = "batch_ops",
    srcs = [
        "batch_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":batch_ops_gen",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "batch_ops_test",
    size = "small",
    srcs = ["batch_ops_test.py"],
    main = "batch_ops_test.py",
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
        ":batch_ops_gen",
        ":functional_ops_gen",
        ":math_ops",
        ":random_ops",
        ":resource_variable_ops",
        ":script_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "manip_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "math_ops_gen",
    visibility = [
        "//learning/brain/google/python/ops:__pkg__",
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/mlir/tfr:__pkg__",
        "//tensorflow/compiler/mlir/tfr/examples/mnist:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/opt:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/math_ops:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "nn_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/mlir/tfr/examples/mnist:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/opt:__pkg__",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/nn_ops:__pkg__",
        "//tensorflow/python/kernel_tests/sparse_ops:__pkg__",
        "//tensorflow/python/tools:__pkg__",
        "//tensorflow/security/fuzzing/google:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "count_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/ops:__subpackages__",
        "//tensorflow/security/fuzzing/google:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "parsing_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/dtensor/python/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/autograph/operators:__pkg__",
        "//tensorflow/python/data/ops:__pkg__",
        "//tensorflow/python/ops/parallel_for:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "random_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/eager/polymorphic_function:__pkg__",
        "//tensorflow/python/kernel_tests/random:__pkg__",
        "//tensorflow/python/ops/parallel_for:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "special_math_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "stateful_random_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/dtensor/python/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests/random:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "resource_variable_ops_gen",
    visibility = [
        "//tensorflow/compiler/mlir/tfr:__pkg__",
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/distribute:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "stateless_random_ops_gen",
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/data/experimental/ops:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "stateless_random_ops_v2_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "stochastic_cast_op_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "random_index_shuffle_ops_gen",
    visibility = visibility + ["//tensorflow/python/kernel_tests/random:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "list_ops_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "map_ops_gen",
    visibility = ["//tensorflow/python:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "script_ops_gen",
    visibility = ["//tensorflow/python:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "sdca_ops_gen",
    visibility = ["//tensorflow/python:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "set_ops_gen",
    visibility = visibility + ["//tensorflow/python/kernel_tests/math_ops:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "state_ops_gen",
    visibility = [
        "//learning/brain/python/ops:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/framework:__pkg__",
        "//tensorflow/python/kernel_tests:__pkg__",
        "//tensorflow/python/kernel_tests/array_ops:__pkg__",
        "//tensorflow/python/kernel_tests/control_flow:__pkg__",
        "//tensorflow/python/kernel_tests/variables:__pkg__",
        "//tensorflow/python/tpu/google/sparse_core:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "sparse_ops_gen",
    visibility = visibility + ["//tensorflow/python/kernel_tests/sparse_ops:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "spectral_ops_gen",
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/ops/signal:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "string_ops_gen",
    visibility = visibility + ["//tensorflow/dtensor:dtensor-internal"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "training_ops_gen",
    visibility = [
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/contrib/opt:__pkg__",
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/eager/polymorphic_function:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "composite_tensor_ops_gen",
    visibility = visibility + ["//tensorflow/python/kernel_tests:__pkg__"],
)

tf_gen_op_strict_wrapper_private_py(
    name = "ragged_array_ops_gen",
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "ragged_math_ops_gen",
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "ragged_conversion_ops_gen",
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "rnn_ops_gen",
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/kernel_tests/nn_ops:__pkg__",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "sendrecv_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:sendrecv_ops_op_lib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "tpu_ops_gen",
    visibility = [
        "//smartass/brain/configure/python:__pkg__",
        "//tensorflow/compiler/tests:__pkg__",
        "//tensorflow/python:__pkg__",
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

tf_gen_op_strict_wrapper_private_py(
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

tf_gen_op_strict_wrapper_private_py(
    name = "uniform_quant_ops_gen",
    visibility = [
        "//tensorflow:__subpackages__",
    ],
    deps = [
        "//tensorflow/core:uniform_quant_ops_op_lib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "sync_ops_gen",
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/core:sync_ops_op_lib",
    ],
)

py_strict_library(
    name = "array_grad",
    srcs = ["array_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":array_ops_stack",
        ":cond",
        ":control_flow_util",
        ":math_ops",
        ":math_ops_gen",
        ":resource_variable_ops_gen",
        ":sparse_ops",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
    ],
)

cuda_py_strict_test(
    name = "array_grad_test",
    srcs = ["array_grad_test.py"],
    main = "array_grad_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":gradient_checker_v2",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "array_ops",
    srcs = [
        "array_ops.py",
        "tensor_getitem_override.py",
    ],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":array_ops_gen",
        ":array_ops_stack",
        ":math_ops_gen",
        ":shape_util",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/dtensor/python:api",  # buildcleaner: keep
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "array_ops_stack",
    srcs = ["array_ops_stack.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops_gen",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "inplace_ops",
    srcs = ["inplace_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
    ],
)

py_strict_library(
    name = "bitwise_ops",
    srcs = ["bitwise_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":bitwise_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "boosted_trees_ops",
    srcs = ["boosted_trees_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":boosted_trees_ops_gen",
        ":resources",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/training:saver",
    ],
)

py_strict_library(
    name = "optional_grad",
    srcs = ["optional_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":optional_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "sets",
    srcs = [
        "sets.py",
    ],
    srcs_version = "PY3",
    deps = [":sets_impl"],
)

py_strict_library(
    name = "candidate_sampling_ops",
    srcs = ["candidate_sampling_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":candidate_sampling_ops_gen",
        ":math_ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "check_ops",
    srcs = ["check_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_assert",
        ":control_flow_ops",
        ":math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "clip_ops",
    srcs = ["clip_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":array_ops_stack",
        ":math_ops",
        ":nn_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "clip_ops_test",
    size = "small",
    srcs = ["clip_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":clip_ops",
        ":numerics",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "clustering_ops",
    srcs = ["clustering_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":clustering_ops_gen",
        ":cond",
        ":control_flow_ops",
        ":embedding_ops",
        ":math_ops",
        ":nn_impl",
        ":random_ops",
        ":state_ops",
        ":variable_v1",
        ":while_loop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
    ],
)

tf_py_strict_test(
    name = "clustering_ops_test",
    size = "medium",
    srcs = ["clustering_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":clustering_ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "collective_ops",
    srcs = ["collective_ops.py"],
    srcs_version = "PY3",
    deps = [":collective_ops_gen"],
)

tf_py_strict_test(
    name = "collective_ops_test",
    size = "small",
    srcs = ["collective_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
    ],
    deps = [
        ":array_ops",
        ":collective_ops",
        ":math_ops",
        ":variable_v1",
        ":variables",
        ":while_loop",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "collective_ops_xla_test",
    size = "small",
    srcs = ["collective_ops_xla_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_rocm",
        "no_windows",
        "nomac",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        ":array_ops",
        ":collective_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "collective_ops_gpu_test",
    size = "small",
    srcs = ["collective_ops_gpu_test.py"],
    main = "collective_ops_gpu_test.py",
    python_version = "PY3",
    tags = [
        "guitar",
        # "multi_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
        "no_windows",
    ],
    deps = [
        ":collective_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "compiled_collective_ops_gpu_test",
    size = "small",
    srcs = ["compiled_collective_ops_gpu_test.py"],
    main = "compiled_collective_ops_gpu_test.py",
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_gpu",
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":collective_ops",
        ":while_loop",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "control_flow_assert",
    srcs = ["control_flow_assert.py"],
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

py_strict_library(
    name = "control_flow_grad",
    srcs =
        ["control_flow_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":control_flow_util",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
    ],
)

py_strict_library(
    name = "control_flow_ops",
    srcs = ["control_flow_ops.py"],
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
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
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
    srcs = ["control_flow_case.py"],
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
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "control_flow_switch_case",
    srcs = ["control_flow_switch_case.py"],
    deps = [
        ":array_ops",
        ":cond_v2",
        ":control_flow_util",
        ":functional_ops_gen",
        ":math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "cond",
    srcs = ["cond.py"],
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
        "//tensorflow/python/framework:tensor",
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
    srcs = ["while_loop.py"],
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

py_strict_library(
    name = "control_flow_util",
    srcs = ["control_flow_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "control_flow_util_v2",
    srcs = ["control_flow_util_v2.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//waymo/ml:__subpackages__"],
    deps = [
        ":control_flow_util",
        ":control_flow_v2_func_graphs",
        ":gradients_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function:atomic_function",
        "//tensorflow/python/eager/polymorphic_function:concrete_function",
        "//tensorflow/python/eager/polymorphic_function:tracing_compilation",
        "//tensorflow/python/eager/polymorphic_function:transform",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function_def_to_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:keras_deps",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "control_flow_v2_func_graphs",
    srcs = ["control_flow_v2_func_graphs.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "control_flow_v2_toggles",
    srcs = ["control_flow_v2_toggles.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "control_flow_v2_toggles_test",
    size = "small",
    srcs = ["control_flow_v2_toggles_test.py"],
    python_version = "PY3",
    deps = [
        ":control_flow_util_v2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "control_flow_v2_enable_test",
    size = "small",
    srcs = ["control_flow_v2_enable_test.py"],
    python_version = "PY3",
    deps = [
        ":control_flow_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "control_flow_v2_disable_test",
    size = "small",
    srcs = ["control_flow_v2_disable_test.py"],
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
        ":control_flow_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_library(
    name = "cond_v2",
    srcs = [
        "cond_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":default_gradient",
        ":functional_ops_gen",
        ":gradients_util",
        ":handle_data_util",
        ":math_ops",
        ":optional_ops_gen",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/framework:auto_control_deps",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:none_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "while_v2",
    srcs = [
        "while_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":default_gradient",
        ":functional_ops_gen",
        ":gradients_util",
        ":handle_data_util",
        ":list_ops",
        ":math_ops",
        ":resource_variable_ops_gen",
        ":tensor_array_ops",
        ":while_v2_indexed_slices_rewriter",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "bincount_ops",
    srcs = ["bincount_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":math_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "bincount_ops_test",
    size = "small",
    srcs = ["bincount_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":bincount_ops",
        ":count_ops_gen",
        ":sparse_ops",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "ctc_ops",
    srcs = ["ctc_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":array_ops_stack",
        ":ctc_ops_gen",
        ":custom_gradient",
        ":functional_ops",
        ":inplace_ops",
        ":linalg_ops",
        ":map_fn",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":sparse_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
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

py_strict_library(
    name = "cudnn_rnn_grad",
    srcs = ["cudnn_rnn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":cudnn_rnn_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "data_flow_grad",
    srcs = ["data_flow_grad.py"],
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

py_strict_library(
    name = "data_flow_ops",
    srcs = ["data_flow_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":control_flow_ops",
        ":data_flow_ops_gen",
        ":math_ops",
        ":resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "embedding_ops",
    srcs = ["embedding_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":clip_ops",
        ":data_flow_grad",
        ":data_flow_ops",
        ":math_ops",
        ":resource_variable_ops",
        ":sparse_ops",
        ":variables",
        "//tensorflow/python/compat",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "handle_data_util",
    srcs = [
        "handle_data_util.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "gradients",
    srcs = [
        "gradients.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":custom_gradient",
        ":gradients_impl",
        ":gradients_util",
        ":unconnected_gradients",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:forwardprop",
        "//tensorflow/python/eager:function",
    ],
)

py_strict_library(
    name = "gradients_impl",
    srcs = [
        "gradients_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":check_ops",
        ":control_flow_grad",
        ":cudnn_rnn_grad",
        ":gradients_util",
        ":image_grad",
        ":io_ops",
        ":linalg_grad",
        ":linalg_ops",
        ":logging_ops",
        ":lookup_grad",
        ":manip_grad",
        ":math_grad",
        ":math_ops",
        ":nccl_ops",
        ":nn_grad",
        ":optional_grad",
        ":parsing_grad",
        ":proto_ops",
        ":random_grad",
        ":rnn_grad",
        ":sdca_ops",
        ":sets",
        ":sparse_grad",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":unconnected_gradients",
        ":while_loop",
        "//tensorflow/compiler/jit/ops:xla_ops_grad",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//tensorflow/python/ops/signal:fft_ops",
        "//tensorflow/python/training:checkpoint_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "gradients_util",
    srcs = [
        "gradients_util.py",
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
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "default_gradient",
    srcs = [
        "default_gradient.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":resource_variable_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

py_strict_library(
    name = "control_flow_state",
    srcs = [
        "control_flow_state.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":control_flow_util",
        ":control_flow_v2_func_graphs",
        ":data_flow_ops_gen",
        ":default_gradient",
        ":resource_variable_ops",
        ":resource_variable_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "unconnected_gradients",
    srcs = ["unconnected_gradients.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "filesystem_ops",
    srcs = ["filesystem_ops.py"],
    srcs_version = "PY3",
    deps = [":filesystem_ops_gen"],
)

py_strict_library(
    name = "histogram_ops",
    srcs = ["histogram_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":clip_ops",
        ":control_flow_assert",
        ":control_flow_ops",
        ":math_ops",
        ":math_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "image_grad",
    srcs = ["image_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":image_ops_gen",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "image_ops",
    srcs = [
        "image_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":image_ops_gen",
        ":image_ops_impl",
        ":linalg_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "init_ops",
    srcs = ["init_ops.py"],
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
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "init_ops_v2",
    srcs = ["init_ops_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":init_ops",
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

py_strict_library(
    name = "initializers_ns",
    srcs = ["initializers_ns.py"],
    srcs_version = "PY3",
    deps = [
        ":init_ops",
        ":variables",
    ],
)

py_strict_library(
    name = "io_ops",
    srcs = ["io_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":data_flow_ops_gen",
        ":io_ops_gen",
        ":parsing_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/lib/io:python_io",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linalg_grad",
    srcs = ["linalg_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":cond",
        ":linalg_ops",
        ":linalg_ops_gen",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/linalg:linalg_impl",
    ],
)

py_strict_library(
    name = "linalg_ops",
    srcs = ["linalg_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":cond",
        ":linalg_ops_gen",
        ":linalg_ops_impl",
        ":map_fn",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linalg_ops_impl",
    srcs = ["linalg_ops_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "manip_grad",
    srcs = ["manip_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":manip_ops",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "manip_ops",
    srcs = ["manip_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":manip_ops_gen",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "logging_ops",
    srcs = ["logging_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":logging_ops_gen",
        ":string_ops",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "lookup_ops",
    srcs = ["lookup_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":lookup_grad",
        ":lookup_ops_gen",
        ":math_ops",
        ":string_ops",
        "//tensorflow/python/checkpoint:saveable_compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/saved_model/registration",
        "//tensorflow/python/trackable:asset",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "lookup_grad",
    srcs = ["lookup_grad.py"],
    deps = [
        "//tensorflow/python/framework:ops",
    ],
)

tf_py_strict_test(
    name = "lookup_ops_async_checkpoint_test",
    srcs = ["lookup_ops_async_checkpoint_test.py"],
    srcs_version = "PY3",
    deps = [
        ":lookup_ops",
        ":variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "math_grad",
    srcs = ["math_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":math_ops",
        ":math_ops_gen",
        ":special_math_ops",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "op_selector",
    srcs = ["op_selector.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:object_identity",
    ],
)

py_strict_library(
    name = "math_ops",
    srcs = [
        "math_ops.py",
        "tensor_math_operator_overrides.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":array_ops_stack",
        ":bitwise_ops_gen",
        ":data_flow_ops_gen",
        ":math_ops_gen",
        ":nn_ops_gen",
        ":sparse_ops_gen",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:override_binary_operator",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "resources",
    srcs = ["resources.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":control_flow_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:tf_should_use",
    ],
)

alias(
    name = "resource_variable_ops",
    actual = ":variables",
)

py_strict_library(
    name = "critical_section_ops",
    srcs = ["critical_section_ops.py"],
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

py_strict_library(
    name = "list_ops",
    srcs = ["list_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":handle_data_util",
        ":list_ops_gen",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "map_ops",
    srcs = ["map_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":map_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "nn",
    srcs = ["nn.py"],
    srcs_version = "PY3",
    deps = [
        ":candidate_sampling_ops",
        ":ctc_ops",
        ":embedding_ops",
        ":math_ops",
        ":nn_fused_batch_norm_grad",
        ":nn_grad",
        ":nn_impl",
        ":nn_impl_distribute",
        ":nn_ops",
    ],
)

py_strict_library(
    name = "nn_grad",
    srcs = ["nn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":math_ops",
        ":nn_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "nn_fused_batch_norm_grad",
    srcs = ["nn_fused_batch_norm_grad.py"],
    deps = [
        ":array_ops",
        ":math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "nn_ops",
    srcs = ["nn_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":check_ops",
        ":math_ops",
        ":math_ops_gen",
        ":nn_grad",
        ":nn_ops_gen",
        ":random_ops",
        ":stateless_random_ops",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:graph_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor",
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

py_strict_library(
    name = "numerics",
    srcs = ["numerics.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "parsing_config",
    srcs = ["parsing_config.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":math_ops",
        ":sparse_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_math_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "parsing_ops",
    srcs = ["parsing_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_assert",
        ":control_flow_ops",
        ":math_ops",
        ":parsing_config",
        ":parsing_grad",
        ":parsing_ops_gen",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "parsing_grad",
    srcs = ["parsing_grad.py"],
    deps = ["//tensorflow/python/framework:ops"],
)

py_strict_library(
    name = "partitioned_variables",
    srcs = ["partitioned_variables.py"],
    srcs_version = "PY3",
    deps = [
        ":variable_scope",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "random_grad",
    srcs = ["random_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":clip_ops",
        ":math_ops",
        ":random_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "random_ops",
    srcs = ["random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":random_ops_gen",
        ":shape_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "random_crop_ops",
    srcs = ["random_crop_ops.py"],
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

py_strict_library(
    name = "stateful_random_ops",
    srcs = ["stateful_random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":math_ops",
        ":random_ops_util",
        ":resource_variable_ops",
        ":stateful_random_ops_gen",
        ":stateless_random_ops",
        ":stateless_random_ops_v2_gen",
        ":variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute:values_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "stateless_random_ops",
    srcs = ["stateless_random_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":bitwise_ops",
        ":math_ops",
        ":random_index_shuffle_ops_gen",
        ":random_ops_util",
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

py_strict_library(
    name = "stochastic_cast_op",
    srcs = ["stochastic_cast_op.py"],
    srcs_version = "PY3",
    deps = [
        ":random_ops_util",
        ":stochastic_cast_op_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "rnn",
    srcs = ["rnn.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":cond",
        ":control_flow_assert",
        ":control_flow_util",
        ":control_flow_util_v2",
        ":math_ops",
        ":rnn_cell_impl",
        ":tensor_array_ops",
        ":variable_scope",
        ":while_loop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "rnn_cell",
    srcs = [
        "rnn_cell.py",
    ],
    srcs_version = "PY3",
    deps = [":rnn_cell_impl"],
)

py_strict_library(
    name = "script_ops",
    srcs = [
        "autograph_ops.py",
        "script_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":check_ops",
        ":resource_variable_ops",
        ":script_ops_gen",
        ":sort_ops",
        "//tensorflow/python/autograph/impl:api",
        "//tensorflow/python/autograph/operators:py_builtins",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:backprop_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/lib/core:_pywrap_py_func",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "sdca_ops",
    srcs = ["sdca_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":sdca_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "session_ops",
    srcs = ["session_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":data_flow_ops_gen",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "shape_util",
    srcs = ["shape_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "random_ops_util",
    srcs = ["random_ops_util.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":bitwise_ops",
        ":math_ops",
        ":stateless_random_ops_v2_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "sparse_grad",
    srcs = ["sparse_grad.py"],
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

py_strict_library(
    name = "sparse_ops",
    srcs = ["sparse_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":bincount_ops",
        ":check_ops",
        ":control_flow_ops",
        ":count_ops_gen",
        ":math_ops",
        ":math_ops_gen",
        ":sparse_ops_gen",
        ":special_math_ops",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sparse_bincount_ops_test",
    srcs = ["sparse_bincount_ops_test.py"],
    deps = [
        ":sparse_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sparse_ops_test",
    srcs = ["sparse_ops_test.py"],
    main = "sparse_ops_test.py",
    python_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":gradient_checker_v2",
        ":math_ops",
        ":sparse_grad",
        ":sparse_ops",
        ":sparse_ops_gen",
        "//tensorflow/python/eager:context",
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

py_strict_library(
    name = "sort_ops",
    srcs = ["sort_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sort_ops_test",
    srcs = ["sort_ops_test.py"],
    main = "sort_ops_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":random_ops",
        ":sort_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "confusion_matrix",
    srcs = ["confusion_matrix.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":check_ops",
        ":cond",
        ":control_flow_ops",
        ":math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "weights_broadcast_ops",
    srcs = [
        "weights_broadcast_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_assert",
        ":control_flow_ops",
        ":math_ops",
        ":sets",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "metrics",
    srcs = [
        "metrics.py",
    ],
    srcs_version = "PY3",
    deps = [":metrics_impl"],
)

py_strict_library(
    name = "special_math_ops",
    srcs = ["special_math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":linalg_ops_gen",
        ":math_ops",
        ":special_math_ops_gen",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
        "@pypi_opt_einsum//:pkg",
    ],
)

py_strict_library(
    name = "rnn_grad",
    srcs = ["rnn_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_strict_test(
    name = "rnn_grad_test",
    srcs = ["rnn_grad_test.py"],
    main = "rnn_grad_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients",
        ":math_ops",
        ":rnn_grad",
        ":rnn_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "script_ops_test",
    srcs = ["script_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":resource_variable_ops",
        ":script_ops",
        ":script_ops_gen",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "autograph_ops_test",
    srcs = ["autograph_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":script_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "standard_ops",
    srcs = ["standard_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":batch_ops",
        ":check_ops",
        ":clip_ops",
        ":cond",
        ":confusion_matrix",
        ":control_flow_assert",
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
        ":numerics",
        ":parsing_ops",
        ":partitioned_variables",
        ":proto_ops",
        ":random_grad",
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
        ":stateless_random_ops",
        ":string_ops",
        ":template",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":variable_scope",
        ":variables",
        ":while_loop",
        "//tensorflow/python/autograph",
        "//tensorflow/python/compiler/tensorrt:trt_convert_py",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/ops/ragged:ragged_batch_gather_ops",
        "//tensorflow/python/ops/ragged:ragged_batch_gather_with_default_op",
        "//tensorflow/python/ops/ragged:ragged_bincount_ops",
        "//tensorflow/python/ops/ragged:ragged_check_ops",
        "//tensorflow/python/ops/ragged:ragged_conversion_ops",
        "//tensorflow/python/ops/ragged:ragged_dispatch",
        "//tensorflow/python/ops/ragged:ragged_embedding_ops",
        "//tensorflow/python/ops/ragged:ragged_image_ops",
        "//tensorflow/python/ops/ragged:ragged_operators",
        "//tensorflow/python/ops/ragged:ragged_squeeze_op",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
    ],
)

py_strict_library(
    name = "state_grad",
    srcs = ["state_grad.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/framework:ops"],
)

py_strict_library(
    name = "state_ops",
    srcs = ["state_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops_gen",
        ":resource_variable_ops_gen",
        ":state_grad",
        ":state_ops_gen",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "string_ops",
    srcs = ["string_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":math_ops",
        ":parsing_ops_gen",
        ":string_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "summary_ops_v2",
    srcs = ["summary_ops_v2.py"],
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:smart_cond",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/trackable:resource",
        "//tensorflow/python/training:training_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "template",
    srcs = ["template.py"],
    srcs_version = "PY3",
    deps = [
        ":variable_scope",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tensor_array_grad",
    srcs = ["tensor_array_grad.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":tensor_array_ops",
        "//tensorflow/python/framework:ops",
    ],
)

py_strict_library(
    name = "tensor_array_ops",
    srcs = ["tensor_array_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":control_flow_ops_gen",
        ":control_flow_util",
        ":data_flow_ops_gen",
        ":list_ops",
        ":math_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:trace",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "composite_tensor_ops",
    srcs = ["composite_tensor_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "composite_tensor_ops_gen",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "variable_scope",
    srcs = ["variable_scope.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":init_ops",
        ":resource_variable_ops",
        ":resource_variables_toggle",
        ":variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "resource_variables_toggle",
    srcs = ["resource_variables_toggle.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "variables",
    srcs = [
        "resource_variable_ops.py",
        "variables.py",
    ],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":array_ops_stack",
        ":control_flow_ops",
        ":handle_data_util",
        ":math_ops",
        ":math_ops_gen",
        ":resource_variable_ops_gen",
        ":state_ops",
        ":state_ops_gen",
        "//tensorflow/compiler/tf2xla/ops:gen_xla_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/checkpoint:tensor_callable",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/framework:auto_control_deps_utils",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
        "//tensorflow/python/util:traceback_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
        "@pypi_typing_extensions//:pkg",
    ],
)

alias(
    name = "ref_variable",
    actual = ":variable_v1",
)

py_strict_library(
    name = "variable_v1",
    srcs = [
        "ref_variable.py",
        "variable_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":cond",
        ":resource_variable_ops",
        ":resource_variables_toggle",
        ":state_ops",
        ":state_ops_gen",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_should_use",
    ],
)

py_strict_library(
    name = "gradient_checker",
    srcs = ["gradient_checker.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients",
        ":math_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "gradient_checker_v2",
    srcs = ["gradient_checker_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients_impl",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "bitwise_ops_test",
    size = "medium",
    srcs = ["bitwise_ops_test.py"],
    main = "bitwise_ops_test.py",
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

cuda_py_strict_test(
    name = "control_flow_ops_test",
    size = "medium",  # Contains artificial waits.
    srcs = ["control_flow_ops_test.py"],
    main = "control_flow_ops_test.py",
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
        ":array_ops_stack",
        ":check_ops",
        ":cond",
        ":control_flow_assert",
        ":control_flow_case",
        ":control_flow_ops",
        ":control_flow_switch_case",
        ":control_flow_util_v2",
        ":control_flow_v2_toggles",
        ":custom_gradient",
        ":embedding_ops",
        ":gradients_impl",
        ":init_ops",
        ":linalg_ops",
        ":math_ops",
        ":random_ops",
        ":script_ops",
        ":state_ops",
        ":summary_ops_v2",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":variable_scope",
        ":variables",
        ":while_loop",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/autograph/lang:directives",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/training:momentum",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "op_selector_test",
    srcs = ["op_selector_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":math_ops",
        ":op_selector",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "embedding_ops_test",
    srcs = ["embedding_ops_test.py"],
    main = "embedding_ops_test.py",
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    deps = [
        ":embedding_ops",
        ":gradients",
        ":math_ops",
        ":resource_variable_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gradient_checker_v2_test",
    size = "medium",
    srcs = ["gradient_checker_v2_test.py"],
    main = "gradient_checker_v2_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":custom_gradient",
        ":gradient_checker_v2",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":sparse_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["gradients_test.py"],
    main = "gradients_test.py",
    python_version = "PY3",
    deps = [
        ":array_grad",
        ":array_ops",
        ":array_ops_stack",
        ":cond",
        ":control_flow_grad",
        ":custom_gradient",
        ":data_flow_grad",
        ":data_flow_ops",
        ":functional_ops",
        ":gradient_checker_v2",
        ":gradients",
        ":gradients_impl",
        ":init_ops",
        ":list_ops",
        ":math_grad",
        ":math_ops",
        ":nn_grad",
        ":nn_ops",
        ":ref_variable",
        ":resource_variable_ops",
        ":state_grad",
        ":state_ops",
        ":tensor_array_grad",
        ":tensor_array_ops",
        ":unconnected_gradients",
        ":variable_scope",
        ":variable_v1",
        ":variables",
        ":while_loop",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:function",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "image_grad_d9m_test",
    size = "large",
    srcs = ["image_grad_d9m_test.py"],
    main = "image_grad_d9m_test.py",
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":array_ops",
        ":gradients_impl",
        ":image_grad_test_base",
        ":image_ops",
        ":random_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "image_grad_test",
    size = "large",
    srcs = ["image_grad_test.py"],
    main = "image_grad_test.py",
    python_version = "PY3",
    shard_count = 5,
    deps = [
        ":image_grad_test_base",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "image_grad_test_base",
    srcs = ["image_grad_test_base.py"],
    srcs_version = "PY3",
    deps = [
        ":array_ops_stack",
        ":gradient_checker_v2",
        ":image_ops",
        ":image_ops_gen",
        ":math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "image_ops_test",
    size = "medium",
    srcs = ["image_ops_test.py"],
    data = ["//tensorflow/core:image_testdata"],
    main = "image_ops_test.py",
    python_version = "PY3",
    shard_count = 16,
    tags = [
        "no_cuda_asan",  # TODO(b/171511582): re-enable.
        "no_windows",  #TODO(b/207035199): re-enable
    ],
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":control_flow_ops",
        ":image_ops",
        ":image_ops_gen",
        ":image_ops_impl",
        ":io_ops",
        ":math_ops",
        ":random_ops",
        ":stateless_random_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/experimental/ops:get_single_element",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "init_ops_test",
    size = "small",
    srcs = ["init_ops_test.py"],
    main = "init_ops_test.py",
    python_version = "PY3",
    deps = [
        ":init_ops",
        ":variable_scope",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "init_ops_v2_test",
    size = "medium",
    srcs = ["init_ops_v2_test.py"],
    main = "init_ops_v2_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":init_ops_v2",
        ":random_ops",
        ":variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "math_grad_test",
    size = "medium",
    srcs = ["math_grad_test.py"],
    main = "math_grad_test.py",
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":gradients",
        ":math_grad",
        ":math_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "math_ops_test",
    size = "medium",
    srcs = ["math_ops_test.py"],
    main = "math_ops_test.py",
    python_version = "PY3",
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        ":array_ops",
        ":gradients",
        ":math_ops",
        ":resource_variable_ops",
        ":tensor_array_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "math_ops_linspace_test",
    size = "medium",
    srcs = ["math_ops_linspace_test.py"],
    main = "math_ops_linspace_test.py",
    python_version = "PY3",
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        ":math_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "nn_batchnorm_test",
    size = "medium",
    srcs = ["nn_batchnorm_test.py"],
    main = "nn_batchnorm_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_mac_arm64",
        "no_windows",
    ],
    deps = [
        ":array_ops",
        ":gradient_checker",
        ":gradients_impl",
        ":math_ops",
        ":nn_grad",
        ":nn_impl",
        ":nn_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "nn_fused_batchnorm_d9m_test",
    size = "medium",
    srcs = ["nn_fused_batchnorm_d9m_test.py"],
    main = "nn_fused_batchnorm_d9m_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
    ],
    deps = [
        ":nn_grad",
        ":nn_impl",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "nn_fused_batchnorm_test",
    size = "large",
    srcs = ["nn_fused_batchnorm_test.py"],
    main = "nn_fused_batchnorm_test.py",
    python_version = "PY3",
    shard_count = 24,
    tags = [],
    deps = [
        ":array_ops",
        ":gradient_checker",
        ":gradients_impl",
        ":math_ops",
        ":nn_fused_batch_norm_grad",
        ":nn_impl",
        ":nn_ops",
        ":nn_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "nn_test",
    size = "medium",
    srcs = ["nn_test.py"],
    main = "nn_test.py",
    python_version = "PY3",
    tags = [
        "no_windows",
        "notap",  # TODO(b/290819913)
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":array_ops",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":math_ops",
        ":nn",
        ":nn_grad",
        ":nn_impl",
        ":nn_ops",
        ":partitioned_variables",
        ":stateful_random_ops",
        ":variable_scope",
        ":variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "nn_loss_scaling_utilities_test",
    size = "small",
    srcs = ["nn_loss_scaling_utilities_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":nn_impl_distribute",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "nn_xent_test",
    size = "medium",
    srcs = ["nn_xent_test.py"],
    main = "nn_xent_test.py",
    python_version = "PY3",
    deps = [
        ":gradient_checker",
        ":gradients_impl",
        ":nn_grad",
        ":nn_impl",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["tensor_array_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":tensor_array_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "sobol_ops_test",
    size = "small",
    srcs = ["sobol_ops_test.py"],
    kernels = [
        "//tensorflow/core/kernels:libtfkernel_sobol_op.so",
    ],
    main = "sobol_ops_test.py",
    tags = [
        "no_oss",  # TODO(b/149565560)
        "no_windows_gpu",
    ],
    deps = [
        ":math_ops",
        ":math_ops_gen",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "special_math_ops_test",
    size = "medium",
    srcs = ["special_math_ops_test.py"],
    main = "special_math_ops_test.py",
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":array_ops",
        ":gradient_checker_v2",
        ":math_ops",
        ":special_math_ops",
        ":variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:benchmark",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@pypi_opt_einsum//:pkg",
    ],
)

tf_py_strict_test(
    name = "variable_spec_test",
    size = "small",
    srcs = ["variable_spec_test.py"],
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "summary_op_util",
    srcs = ["summary_op_util.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
    ],
)

# -----------------------------------------------------------------------------
# Quantization

tf_py_strict_test(
    name = "dequantize_op_test",
    size = "small",
    srcs = ["dequantize_op_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":array_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "quantized_ops_test",
    size = "small",
    srcs = ["quantized_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":array_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "quantized_conv_ops_test",
    size = "small",
    srcs = ["quantized_conv_ops_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":math_ops",
        ":nn_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "array_ops_test",
    srcs = ["array_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":math_ops",
        ":random_ops",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "array_ops_shape_test",
    srcs = ["array_ops_shape_test.py"],
    env = {"TF_FLAG_TF_SHAPE_DEFAULT_INT64": "true"},
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":array_ops",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_benchmark_test(
    name = "accumulate_n_benchmark",
    size = "medium",
    srcs = ["accumulate_n_benchmark.py"],
    main = "accumulate_n_benchmark.py",
    python_version = "PY3",
    shard_count = 6,
    deps = [
        ":array_ops",
        ":control_flow_ops_gen",
        ":data_flow_ops",
        ":math_ops",
        ":random_ops",
        ":state_ops",
        ":state_ops_gen",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_benchmark_test(
    name = "batch_norm_benchmark",
    srcs = ["batch_norm_benchmark.py"],
    main = "batch_norm_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":gradients_impl",
        ":math_ops",
        ":nn_grad",
        ":nn_impl",
        ":nn_ops_gen",
        ":random_ops",
        ":variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_benchmark_test(
    name = "collective_ops_benchmark",
    srcs = ["collective_ops_benchmark.py"],
    main = "collective_ops_benchmark.py",
    python_version = "PY3",
    deps = [
        ":collective_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_benchmark_test(
    name = "concat_benchmark",
    srcs = ["concat_benchmark.py"],
    main = "concat_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":gradients_impl",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_benchmark_test(
    name = "control_flow_ops_benchmark",
    srcs = ["control_flow_ops_benchmark.py"],
    main = "control_flow_ops_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":cond",
        ":control_flow_util",
        ":math_ops",
        ":random_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_benchmark_test(
    name = "conv2d_benchmark",
    size = "large",
    srcs = ["conv2d_benchmark.py"],
    main = "conv2d_benchmark.py",
    python_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":nn_ops",
        ":random_ops",
        ":variable_v1",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:flags",
    ],
)

cuda_py_benchmark_test(
    name = "split_benchmark",
    srcs = ["split_benchmark.py"],
    main = "split_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":variables",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:benchmark",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_benchmark_test(
    name = "transpose_benchmark",
    size = "medium",
    srcs = ["transpose_benchmark.py"],
    main = "transpose_benchmark.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":control_flow_ops",
        ":variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_benchmark_test(
    name = "matmul_benchmark",
    size = "medium",
    srcs = ["matmul_benchmark.py"],
    main = "matmul_benchmark.py",
    python_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":math_ops",
        ":random_ops",
        ":variable_v1",
        ":variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "matmul_benchmark_main_lib",
    testonly = True,
    srcs = ["matmul_benchmark.py"],
    srcs_version = "PY3",
    deps = [
        ":control_flow_ops",
        ":math_ops",
        ":random_ops",
        ":variable_v1",
        ":variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "nn_grad_test",
    size = "medium",
    srcs = ["nn_grad_test.py"],
    main = "nn_grad_test.py",
    python_version = "PY3",
    deps = [
        ":array_ops",
        ":gradient_checker",
        ":gradient_checker_v2",
        ":gradients_impl",
        ":nn_grad",
        ":nn_impl",
        ":nn_ops",
        ":nn_ops_gen",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "nccl_ops_gen",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:nccl_ops_op_lib",
    ],
)

py_strict_library(
    name = "nccl_ops",
    srcs = ["nccl_ops.py"],
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

cuda_py_strict_test(
    name = "nccl_ops_test",
    size = "small",
    srcs = ["nccl_ops_test.py"],
    main = "nccl_ops_test.py",
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
        ":gradients",
        ":nccl_ops",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "factory_ops_test",
    size = "small",
    srcs = ["factory_ops_test.py"],
    main = "factory_ops_test.py",
    deps = [
        ":sparse_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "decode_proto_ops_gen",
    visibility = ["//tensorflow/python:__pkg__"],
    deps = [
        "//tensorflow/core:decode_proto_ops_op_lib",
    ],
)

tf_gen_op_strict_wrapper_private_py(
    name = "encode_proto_ops_gen",
    visibility = ["//tensorflow/python:__pkg__"],
    deps = [
        "//tensorflow/core:encode_proto_ops_op_lib",
    ],
)

py_strict_library(
    name = "proto_ops",
    srcs = ["proto_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":decode_proto_ops_gen",
        ":encode_proto_ops_gen",
        "//tensorflow/python/framework:ops",
    ],
)

cuda_py_strict_test(
    name = "raw_ops_test",
    srcs = ["raw_ops_test.py"],
    main = "raw_ops_test.py",
    python_version = "PY3",
    deps = [
        ":data_flow_ops_gen",
        ":math_ops_gen",
        ":string_ops_gen",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "nn_impl_distribute",
    srcs = ["nn_impl_distribute.py"],
    deps = [
        ":array_ops",
        ":check_ops",
        ":math_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/losses:util",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "nn_impl",
    srcs = ["nn_impl.py"],
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":candidate_sampling_ops",
        ":cond",
        ":ctc_ops",
        ":custom_gradient",
        ":embedding_ops",
        ":linalg_ops",
        ":math_ops",
        ":nn_fused_batch_norm_grad",
        ":nn_ops",
        ":nn_ops_gen",
        ":sparse_ops_gen",
        ":variables",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:device_context",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "rnn_cell_wrapper_impl",
    srcs = ["rnn_cell_wrapper_impl.py"],
    deps = ["//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_wrapper_impl"],
)

py_strict_library(
    name = "rnn_cell_impl",
    srcs = ["rnn_cell_impl.py"],
    deps = [
        ":array_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_impl",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "image_ops_impl",
    srcs = ["image_ops_impl.py"],
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":check_ops",
        ":cond",
        ":control_flow_assert",
        ":control_flow_case",
        ":control_flow_ops",
        ":image_ops_gen",
        ":math_ops",
        ":nn_impl",
        ":nn_ops",
        ":random_ops",
        ":ref_variable",
        ":sort_ops",
        ":stateless_random_ops",
        ":string_ops",
        ":variables",
        ":while_loop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "custom_gradient",
    srcs = ["custom_gradient.py"],
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":handle_data_util",
        ":math_ops",
        ":op_selector",
        ":resource_variable_ops",
        ":unconnected_gradients",
        ":variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "while_v2_indexed_slices_rewriter",
    srcs = ["while_v2_indexed_slices_rewriter.py"],
    deps = [
        ":array_ops",
        ":resource_variable_ops_gen",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "metrics_impl",
    srcs = ["metrics_impl.py"],
    deps = [
        ":array_ops",
        ":array_ops_stack",
        ":check_ops",
        ":cond",
        ":confusion_matrix",
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "sets_impl",
    srcs = ["sets_impl.py"],
    deps = [
        ":set_ops_gen",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "weak_tensor_ops",
    srcs = ["weak_tensor_ops.py"],
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":bitwise_ops_gen",
        ":clip_ops",
        ":image_ops",
        ":math_ops",
        ":math_ops_gen",
        ":nn_impl",
        ":nn_ops",
        ":resource_variable_ops",
        ":special_math_ops",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_math_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_strict_test(
    name = "weak_tensor_ops_test",
    srcs = ["weak_tensor_ops_test.py"],
    tags = ["noasan"],  # TODO(b/293304945): Reenable.
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":bitwise_ops_gen",
        ":clip_ops",
        ":image_ops",
        ":math_ops",
        ":math_ops_gen",
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:np_math_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:dispatch",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "weak_tensor_math_ops_test",
    srcs = ["weak_tensor_math_ops_test.py"],
    deps = [
        ":array_ops",
        ":math_ops",
        ":math_ops_gen",
        ":resource_variable_ops",
        ":tensor_array_ops",
        ":variables",
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "weak_tensor_array_ops_test",
    srcs = ["weak_tensor_array_ops_test.py"],
    deps = [
        ":array_ops",
        ":array_ops_gen",
        ":math_ops",
        ":random_ops",
        ":weak_tensor_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "weak_tensor_nn_test",
    size = "medium",
    srcs = ["weak_tensor_nn_test.py"],
    main = "weak_tensor_nn_test.py",
    python_version = "PY3",
    deps = [
        ":gradient_checker_v2",
        ":math_ops",
        ":nn_grad",
        ":nn_impl",
        ":nn_ops",
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "weak_tensor_special_math_ops_test",
    size = "medium",
    srcs = ["weak_tensor_special_math_ops_test.py"],
    main = "weak_tensor_special_math_ops_test.py",
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":gradient_checker_v2",
        ":special_math_ops",
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "weak_tensor_np_math_ops_test",
    srcs = ["weak_tensor_np_math_ops_test.py"],
    deps = [
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_arrays",
        "//tensorflow/python/ops/numpy_ops:np_math_ops",
        "//tensorflow/python/ops/numpy_ops:np_random",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "weak_tensor_np_array_ops_test",
    srcs = ["weak_tensor_np_array_ops_test.py"],
    deps = [
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_arrays",
        "//tensorflow/python/ops/numpy_ops:np_math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "weak_tensor_image_ops_test",
    srcs = ["weak_tensor_image_ops_test.py"],
    deps = [
        ":image_ops",
        ":image_ops_impl",
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "weak_tensor_constant_op_test",
    srcs = ["weak_tensor_constant_op_test.py"],
    deps = [
        ":gradients_impl",
        ":weak_tensor_ops",
        ":weak_tensor_test_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:flexible_dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:weak_tensor",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

py_strict_library(
    name = "weak_tensor_test_util",
    srcs = ["weak_tensor_test_util.py"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:weak_tensor",
        "//third_party/py/numpy",
    ],
)
