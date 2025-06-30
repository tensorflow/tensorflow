# Tensorflow util package

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "tf_py_strict_test", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap", "if_static")

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
    "//third_party/py/tf_keras:__subpackages__",
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
        ":tf_contextlib",
        ":tf_decorator_py",
        ":tf_export",
        ":tf_inspect",
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
        "@com_google_absl//absl/log",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_tfprof",
    srcs = ["tfprof_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tfprof.pyi",
    ],
    deps = if_pywrap(
        if_false = [
            "//tensorflow/core/profiler/internal:print_model_analysis_hdr",
            "@pybind11",
        ],
        if_true = [
            "//tensorflow/core:framework",
            "//tensorflow/core/framework:reader_base",
            "//tensorflow/core:lib_headers_for_pybind",
            "//tensorflow/core/profiler/internal:print_model_analysis",
            "@local_xla//third_party/python_runtime:headers",
            "@com_google_absl//absl/strings",
            "@eigen_archive//:eigen3",
            "@pybind11",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_pywrap_utils",
    srcs = ["util_wrapper.cc"],
    hdrs = ["util.h"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_utils.pyi",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_nest",
    srcs = ["nest_wrapper.cc"],
    hdrs = ["nest.h"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_nest.pyi",
    ],
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap([":cpp_nest"]),
)

cc_library(
    name = "cpp_nest",
    srcs = ["nest.cc"],
    hdrs = ["nest.h"],
    deps = [
        ":cpp_python_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "@local_xla//third_party/python_runtime:headers",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_kernel_registry",
    srcs = ["kernel_registry_wrapper.cc"],
    hdrs = ["kernel_registry.h"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_kernel_registry.pyi",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap([":kernel_registry"]),
)

tf_python_pybind_extension(
    name = "_pywrap_stat_summarizer",
    srcs = ["stat_summarizer_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_stat_summarizer.pyi",
    ],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util:stat_summarizer",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tensor_float_32_execution",
    srcs = ["tensor_float_32.cc"],
    hdrs = ["//tensorflow/core/platform:tensor_float_32_hdr"],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tensor_float_32_execution.pyi",
    ],
    deps = [
        "//tensorflow/core/platform:tensor_float_32_hdr_lib",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_determinism",
    srcs = ["determinism.cc"],
    hdrs = ["//tensorflow/core/util:determinism_hdr"],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_determinism.pyi",
    ],
    deps = [
        "//tensorflow/core/util:determinism_hdr_lib",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_util_port",
    srcs = ["port_wrapper.cc"],
    hdrs = ["//tensorflow/core/util:port_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_util_port.pyi",
    ],
    deps = [
        "//tensorflow/core/util:port",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_transform_graph",
    srcs = ["transform_graph_wrapper.cc"],
    hdrs = ["//tensorflow/tools/graph_transforms:transform_graph_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_transform_graph.pyi",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@pybind11",
    ] + if_pywrap(["//tensorflow/tools/graph_transforms:transform_graph_lib"]),
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_checkpoint_reader.pyi",
    ],
    deps = [
        "//tensorflow/compiler/tf2tensorrt:common_utils",
        "//tensorflow/compiler/tf2tensorrt:trt_parameters",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap(["@com_google_absl//absl/strings"]),
)

cc_library(
    name = "cpp_python_util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/memory",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

tf_py_strict_test(
    name = "decorator_utils_test",
    srcs = ["decorator_utils_test.py"],
    deps = [
        ":decorator_utils",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "deprecation_test",
    srcs = ["deprecation_test.py"],
    deps = [
        ":deprecated_module",
        ":deprecation",
        ":tf_inspect",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:strict_mode",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.py"],
    deps = [
        ":deprecation",
        ":dispatch",
        ":nest",
        ":tf_export",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bitwise_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:proto_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/linalg:linear_operator_diag",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:core",
    ],
)

tf_py_strict_test(
    name = "keyword_args_test",
    srcs = ["keyword_args_test.py"],
    deps = [
        ":keyword_args",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tf_export",
    srcs = ["tf_export.py"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//third_party/py/tensorflow_core:__subpackages__",
    # "//tensorflow:__pkg__",
    # "//tensorflow:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
    ],
)

pytype_strict_library(
    name = "tf_contextlib",
    srcs = ["tf_contextlib.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//learning/deepmind/research/language/translation/lm:__subpackages__",
        "//tensorflow:__pkg__",
        "//tensorflow:__subpackages__",
        "//third_party/py/tensorflow_core:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [":tf_decorator_py"],
)

py_strict_library(
    name = "tf_decorator_py",
    srcs = ["tf_decorator.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//learning/deepmind/research/language/translation/lm:__subpackages__",
        "//tensorflow:__pkg__",
        "//tensorflow:__subpackages__",
        "//third_party/py/tensorflow_core:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
)

tf_py_strict_test(
    name = "tf_export_test",
    srcs = ["tf_export_test.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_export",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "vlog_test",
    srcs = ["vlog_test.py"],
    deps = [
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "tf_stack",
    srcs = ["tf_stack.py"],
    # TODO(mdan): Remove public visibility.
    visibility = ["//visibility:public"],
    deps = [
        ":_tf_stack",
        "//tensorflow/core:protos_all_py",
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_tf_stack.pyi",
    ],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/compiler/tf2tensorrt:common_utils",
        "//tensorflow/compiler/tf2tensorrt:trt_parameters",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:stack_frame",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:mutex",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_static([
        ":stack_trace",
    ]),
)

tf_py_strict_test(
    name = "tf_stack_test",
    srcs = ["tf_stack_test.py"],
    deps = [
        ":tf_stack",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cc_library(
    name = "stack_trace",
    srcs = ["stack_trace.cc"],
    hdrs = ["stack_trace.h"],
    deps = [
        "//tensorflow/core/platform:stack_frame",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "function_parameter_canonicalizer_hdrs",
    textual_hdrs = ["function_parameter_canonicalizer.h"],
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

tf_python_pybind_extension(
    name = "_function_parameter_canonicalizer_binding_for_test",
    # testonly = True,
    srcs = ["function_parameter_canonicalizer_binding_for_test.cc"],
    hdrs = [
        "function_parameter_canonicalizer.h",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_function_parameter_canonicalizer_binding_for_test.pyi",
    ],
    starlark_only = True,
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//tensorflow/compiler/tf2xla:tf2xla_opset",
            "//tensorflow/python/lib/core:pybind11_lib",
        ],
    ),
)

tf_py_strict_test(
    name = "function_parameter_canonicalizer_test",
    srcs = ["function_parameter_canonicalizer_test.py"],
    tags = [
        "no_pip",  # b/168621686
        "no_windows",  # b/169275019
    ],
    deps = [
        ":_function_parameter_canonicalizer_binding_for_test",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "traceback_utils_test",
    size = "small",
    srcs = ["traceback_utils_test.py"],
    deps = [
        ":traceback_utils",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "object_identity_test",
    size = "small",
    srcs = ["object_identity_test.py"],
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
    deps = [
        ":nest_test_main_lib",
    ],
)

py_strict_library(
    name = "nest_test_main_lib",
    testonly = True,
    srcs = ["nest_test.py"],
    deps = [
        ":nest",
        ":nest_util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    main = "serialization_test.py",
    deps = [
        ":serialization",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "function_utils_test",
    srcs = ["function_utils_test.py"],
    deps = [
        ":function_utils",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tf_contextlib_test",
    size = "small",
    srcs = ["tf_contextlib_test.py"],
    deps = [
        ":tf_contextlib",
        ":tf_decorator_py",
        ":tf_inspect",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tf_decorator_test",
    size = "small",
    srcs = ["tf_decorator_test.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "tf_should_use",
    srcs = ["tf_should_use.py"],
    deps = [
        ":tf_decorator_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "tf_should_use_test",
    size = "small",
    srcs = ["tf_should_use_test.py"],
    deps = [
        ":tf_should_use",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "tf_inspect_test",
    size = "small",
    srcs = ["tf_inspect_test.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "example_parser_configuration",
    srcs = ["example_parser_configuration.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "deprecation",
    srcs = ["deprecation.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":decorator_utils",
        ":is_in_graph_mode",
        ":tf_contextlib",
        ":tf_decorator_py",
        ":tf_inspect",
        "//tensorflow/python/framework:strict_mode",
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
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
        ":custom_nest_protocol",
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        "@pypi_wrapt//:pkg",
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
        ":tf_decorator_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
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
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

tf_py_strict_test(
    name = "lazy_loader_test",
    srcs = ["lazy_loader_test.py"],
    deps = [
        ":lazy_loader",
        ":tf_inspect",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
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
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "numpy_compat",
    srcs = ["numpy_compat.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py test because not all tensorflow tests use tensorflow.bzl's py test.
        "//tensorflow/python:global_test_configuration",
        "//third_party/py/numpy",
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
        "@pypi_wrapt//:pkg",
    ],
)

py_strict_library(
    name = "traceback_utils",
    srcs = ["traceback_utils.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":tf_decorator_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
    ],
)

py_strict_library(
    name = "module_wrapper",
    srcs = ["module_wrapper.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":tf_decorator_py",
        ":tf_inspect",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        "//tensorflow/tools/compatibility:all_renames_v2",
        ":fast_module_type",
    ],
)

py_strict_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        "//tensorflow/core:protos_all_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
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
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        ":traceback_utils",
        ":type_annotations",
        "//tensorflow/python/framework:_pywrap_python_api_dispatcher",
        "//tensorflow/python/framework:ops",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
    ],
)

py_strict_library(
    name = "keras_deps",
    srcs = ["keras_deps.py"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
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
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":__init__",
        ":nest_util",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_export",
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
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tf_inspect",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
    ],
)

tf_py_strict_test(
    name = "lock_util_test",
    size = "small",
    srcs = ["lock_util_test.py"],
    main = "lock_util_test.py",
    deps = [
        ":lock_util",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "fast_module_type",
    srcs = ["fast_module_type.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "fast_module_type.pyi",
    ],
    deps = [
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "fast_module_type_test",
    srcs = ["fast_module_type_test.py"],
    deps = [
        ":fast_module_type",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "module_wrapper_test",
    size = "small",
    srcs = ["module_wrapper_test.py"],
    deps = [
        ":module_wrapper",
        ":tf_inspect",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/tools/compatibility:all_renames_v2",
    ],
)

tf_py_strict_test(
    name = "example_parser_configuration_test",
    size = "small",
    srcs = ["example_parser_configuration_test.py"],
    main = "example_parser_configuration_test.py",
    deps = [
        ":example_parser_configuration",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "type_annotations_test",
    size = "small",
    srcs = ["type_annotations_test.py"],
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
    deps = [
        ":compat",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "numpy_compat_test",
    srcs = ["numpy_compat_test.py"],
    deps = [
        ":numpy_compat",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "variable_utils_test",
    srcs = ["variable_utils_test.py"],
    deps = [
        ":nest",
        ":variable_utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "pywrap_xla_ops_test",
    srcs = ["pywrap_xla_ops_test.py"],
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
    hdrs = if_pywrap(
        if_false = [
            "//tensorflow/compiler/tf2xla:tf2xla_opset_hdrs",
        ],
        if_true = [],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "pywrap_xla_ops.pyi",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_pywrap(["//tensorflow/compiler/tf2xla:tf2xla_opset"]),
)

py_strict_library(
    name = "tf_decorator_export",
    srcs = ["tf_decorator_export.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_export",
    ],
)

py_strict_library(
    name = "custom_nest_protocol",
    srcs = ["custom_nest_protocol.py"],
)
