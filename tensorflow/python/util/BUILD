# Tensorflow util package

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "tf_py_strict_test", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//tensorflow:internal",
    "//tensorflow/compiler/tf2xla:__pkg__",
    "//tensorflow/lite/toco/python:__pkg__",
    "//tensorflow_models:__subpackages__",
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/py/launchpad:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/tensorflow_examples:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/tensorflow_docs:__subpackages__",
    "//third_party/py/keras:__subpackages__",
]

util_subpackage_visibility = visibility + [
    # copybara:uncomment "//learning/brain:composite_tensor_allowlist",
    "//tensorflow:__pkg__",
    "//third_party/py/tensorflow_core:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",
    "//third_party/py/tfx:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

py_strict_library(
    name = "core",
    deps = [
        ":tf_decorator",
        ":tf_export",
        ":tf_stack",
    ],
)

# TODO(mdan): Move this utility outside of TF.
cc_library(
    name = "kernel_registry",
    srcs = ["kernel_registry.cc"],
    hdrs = ["kernel_registry.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_tfprof",
    srcs = ["tfprof_wrapper.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/profiler/internal:print_model_analysis_hdr",
        "//third_party/eigen3",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_utils",
    srcs = ["util_wrapper.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_nest",
    srcs = ["nest_wrapper.cc"],
    hdrs = ["nest.h"],
    deps = [
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "cpp_nest",
    srcs = ["nest.cc"],
    hdrs = ["nest.h"],
    deps = [
        ":cpp_python_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_kernel_registry",
    srcs = ["kernel_registry_wrapper.cc"],
    hdrs = ["kernel_registry.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_stat_summarizer",
    srcs = ["stat_summarizer_wrapper.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util:stat_summarizer",
        "//tensorflow/tsl/util:stats_calculator_portable",
        "//third_party/eigen3",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tensor_float_32_execution",
    srcs = ["tensor_float_32.cc"],
    hdrs = ["//tensorflow/core/platform:tensor_float_32_hdr"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_determinism",
    srcs = ["determinism.cc"],
    hdrs = ["//tensorflow/core/util:determinism_hdr"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_util_port",
    srcs = ["port_wrapper.cc"],
    hdrs = ["//tensorflow/core/util:port_hdrs"],
    deps = [
        "//tensorflow/core/util:port",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_transform_graph",
    srcs = ["transform_graph_wrapper.cc"],
    hdrs = ["//tensorflow/tools/graph_transforms:transform_graph_hdrs"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_checkpoint_reader",
    srcs = ["py_checkpoint_reader_wrapper.cc"],
    hdrs = [
        "//tensorflow/c:checkpoint_reader_hdrs",
        "//tensorflow/c:headers",
        "//tensorflow/c:safe_ptr_hdr",
        "//tensorflow/c/eager:headers",
        "//tensorflow/python/lib/core:ndarray_tensor_hdr",
        "//tensorflow/python/lib/core:py_exception_registry_hdr",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    deps = [
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/tensor_bundle:tensor_bundle_headers_lib",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

cc_library(
    name = "cpp_python_util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/c:safe_ptr",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
    ],
)

tf_py_strict_test(
    name = "decorator_utils_test",
    srcs = ["decorator_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":decorator_utils",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "deprecation_test",
    srcs = [
        "deprecated_module.py",
        "deprecated_module_new.py",
        "deprecation_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":deprecation",
        ":tf_inspect",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.py"],
    python_version = "PY3",
    deps = [
        ":deprecation",
        ":dispatch",
        ":nest",
        ":tf_export",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:proto_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg:linear_operator_diag",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
    ],
)

tf_py_strict_test(
    name = "keyword_args_test",
    srcs = ["keyword_args_test.py"],
    python_version = "PY3",
    deps = [
        ":keyword_args",
        "//tensorflow/python:client_testlib",
    ],
)

py_strict_library(
    name = "tf_export",
    srcs = ["tf_export.py"],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:__subpackages__",
        "//tensorflow_estimator:__subpackages__",
    ],
    deps = [
        ":tf_decorator",
    ],
)

tf_py_strict_test(
    name = "tf_export_test",
    srcs = ["tf_export_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_decorator",
        ":tf_export",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "vlog_test",
    srcs = ["vlog_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
    ],
)

# Leaf library: may not depend on anything else inside TensorFlow.
# TODO(mdan): Move this utility outside of TF.
py_strict_library(
    name = "tf_decorator",
    srcs = [
        "tf_contextlib.py",
        "tf_decorator.py",
        "tf_inspect.py",
    ],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__subpackages__",
        # TODO(mdan): Remove these dependencies.
        "//third_party/py/tf_slim:__subpackages__",
        "//learning/deepmind/research/language/translation/lm:__subpackages__",
        "//learning/brain/analytics:__subpackages__",
        "//tensorflow:__pkg__",
    ],
    deps = [
        "@six_archive//:six",
    ],
)

# Note: this is a heavyweight library specialized for TensorFlow graphs. Do not use for
# other purposes.
py_strict_library(
    name = "tf_stack",
    srcs = ["tf_stack.py"],
    srcs_version = "PY3",
    # TODO(mdan): Remove public visibility.
    visibility = ["//visibility:public"],
    deps = [
        ":_tf_stack",
        "@six_archive//:six",
    ],
)

tf_python_pybind_extension(
    name = "_tf_stack",
    srcs = ["tf_stack.cc"],
    hdrs = [
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:headers",
        # Using header directly is required to avoid ODR violations.
        "stack_trace.h",
    ],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/platform:path",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_static([
        ":stack_trace",
    ]),
)

tf_py_strict_test(
    name = "tf_stack_test",
    srcs = ["tf_stack_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_stack",
        "//tensorflow/python:client_testlib",
    ],
)

cc_library(
    name = "stack_trace",
    srcs = ["stack_trace.cc"],
    hdrs = ["stack_trace.h"],
    deps = [
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/util:managed_stack_trace",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "function_parameter_canonicalizer",
    srcs = ["function_parameter_canonicalizer.cc"],
    hdrs = ["function_parameter_canonicalizer.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/python/lib/core:py_util",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "function_parameter_canonicalizer_hdrs",
    textual_hdrs = ["function_parameter_canonicalizer.h"],
)

tf_python_pybind_extension(
    name = "_function_parameter_canonicalizer_binding_for_test",
    testonly = True,
    srcs = ["function_parameter_canonicalizer_binding_for_test.cc"],
    hdrs = [
        "function_parameter_canonicalizer.h",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "function_parameter_canonicalizer_test",
    srcs = ["function_parameter_canonicalizer_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # b/168621686
        "no_windows",  # b/169275019
    ],
    deps = [
        ":_function_parameter_canonicalizer_binding_for_test",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_strict_test(
    name = "traceback_utils_test",
    size = "small",
    srcs = ["traceback_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":traceback_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "object_identity_test",
    size = "small",
    srcs = ["object_identity_test.py"],
    python_version = "PY3",
    deps = [
        ":nest",
        ":object_identity",
        "//tensorflow/python/platform:client_testlib",
    ],
)

# Placeholder for intenal nest_test comments.
tf_py_strict_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    main = "nest_test.py",
    python_version = "PY3",
    deps = [":nest_test_main_lib"],
)

py_strict_library(
    name = "nest_test_main_lib",
    testonly = True,
    srcs = ["nest_test.py"],
    srcs_version = "PY3",
    deps = [
        ":nest",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    main = "serialization_test.py",
    python_version = "PY3",
    deps = [
        ":serialization",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

tf_py_strict_test(
    name = "function_utils_test",
    srcs = ["function_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":function_utils",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tf_contextlib_test",
    size = "small",
    srcs = ["tf_contextlib_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_decorator",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tf_decorator_test",
    size = "small",
    srcs = ["tf_decorator_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_decorator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "tf_should_use",
    srcs = ["tf_should_use.py"],
    srcs_version = "PY3",
    deps = [
        ":tf_decorator",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "tf_should_use_test",
    size = "small",
    srcs = ["tf_should_use_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_should_use",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "tf_inspect_test",
    size = "small",
    srcs = ["tf_inspect_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_decorator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "example_parser_configuration",
    srcs = ["example_parser_configuration.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "deprecation",
    srcs = ["deprecation.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":decorator_utils",
        ":is_in_graph_mode",
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_strict_library(
    name = "type_annotations",
    srcs = ["type_annotations.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "nest_util",
    srcs = ["nest_util.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":compat",
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        "@six_archive//:six",
        "@wrapt",
        "//tensorflow/python:pywrap_tensorflow",
        ":_pywrap_utils",
    ],
)

py_strict_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "deprecated_module",
    srcs = ["deprecated_module.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":deprecated_module_new",
        ":deprecation",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "lazy_loader",
    srcs = ["lazy_loader.py"],
    visibility = util_subpackage_visibility,
    deps = [
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "decorator_utils",
    srcs = ["decorator_utils.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "variable_utils",
    srcs = ["variable_utils.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":nest",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":_pywrap_utils",
    ],
)

py_strict_library(
    name = "compat",
    srcs = ["compat.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "object_identity",
    srcs = ["object_identity.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":compat",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "keyword_args",
    srcs = ["keyword_args.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":decorator_utils",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "serialization",
    srcs = ["serialization.py"],
    visibility = util_subpackage_visibility + [
        "//tensorflow_kfac/python/kernel_tests:__subpackages__",
    ],
    deps = [
        ":compat",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        "//third_party/py/numpy",
        "@wrapt",
    ],
)

py_strict_library(
    name = "traceback_utils",
    srcs = ["traceback_utils.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        ":tf_export",
    ],
)

py_strict_library(
    name = "deprecation_wrapper",
    srcs = ["deprecation_wrapper.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":module_wrapper",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "module_wrapper",
    srcs = ["module_wrapper.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        "//tensorflow/tools/compatibility:all_renames_v2",
        ":fast_module_type",
    ],
)

py_strict_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    visibility = util_subpackage_visibility,
    deps = [
        "//tensorflow/core:protos_all_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "lock_util",
    srcs = ["lock_util.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "is_in_graph_mode",
    srcs = ["is_in_graph_mode.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "dispatch",
    srcs = ["dispatch.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":traceback_utils",
        ":type_annotations",
        "//tensorflow/python/framework:_pywrap_python_api_dispatcher",
        "//tensorflow/python/framework:ops",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        ":tf_export",
    ],
)

py_strict_library(
    name = "keras_deps",
    srcs = ["keras_deps.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
    ],
)

py_strict_library(
    name = "nest",
    srcs = ["nest.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":compat",
        ":nest_util",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
        "@wrapt",
        ":_pywrap_utils",
        ":_pywrap_nest",
    ],
)

py_strict_library(
    name = "deprecated_module_new",
    srcs = ["deprecated_module_new.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "all_util",
    srcs = ["all_util.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
    ],
)

tf_py_strict_test(
    name = "lock_util_test",
    size = "small",
    srcs = ["lock_util_test.py"],
    main = "lock_util_test.py",
    python_version = "PY3",
    deps = [
        ":lock_util",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "fast_module_type",
    srcs = ["fast_module_type.cc"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "fast_module_type_test",
    srcs = ["fast_module_type_test.py"],
    python_version = "PY3",
    deps = [
        ":fast_module_type",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "module_wrapper_test",
    size = "small",
    srcs = ["module_wrapper_test.py"],
    python_version = "PY3",
    deps = [
        ":module_wrapper",
        ":tf_inspect",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/tools/compatibility:all_renames_v2",
    ],
)

tf_py_strict_test(
    name = "example_parser_configuration_test",
    size = "small",
    srcs = ["example_parser_configuration_test.py"],
    main = "example_parser_configuration_test.py",
    python_version = "PY3",
    deps = [
        ":example_parser_configuration",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/client:session",
    ],
)

tf_py_strict_test(
    name = "type_annotations_test",
    size = "small",
    srcs = ["type_annotations_test.py"],
    python_version = "PY3",
    deps = [
        ":type_annotations",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

filegroup(
    name = "util_hdr",
    srcs = ["util.h"],
)

tf_py_strict_test(
    name = "compat_test",
    srcs = ["compat_test.py"],
    python_version = "PY3",
    deps = [
        ":compat",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_strict_test(
    name = "variable_utils_test",
    srcs = ["variable_utils_test.py"],
    deps = [
        ":nest",
        ":variable_utils",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "pywrap_xla_ops_test",
    srcs = ["pywrap_xla_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_gpu",
        "no_pip",
    ],
    deps = [
        ":pywrap_xla_ops",
        "//tensorflow/python/platform:test",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_xla_ops",
    srcs = ["tf2xla_opset_wrapper.cc"],
    hdrs = [
        "//tensorflow/compiler/tf2xla:tf2xla_opset_hdrs",
    ],
    deps = [
        "//tensorflow/python:pybind11_lib",
        "@com_google_absl//absl/status:statusor",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
