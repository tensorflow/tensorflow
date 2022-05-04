# Tensorflow util package

load("//tensorflow:tensorflow.bzl", "py_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//tensorflow:internal",
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

package(
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
    module_name = "_pywrap_tfprof",
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
    module_name = "_pywrap_utils",
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
    module_name = "_pywrap_nest",
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
    module_name = "_pywrap_kernel_registry",
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
    module_name = "_pywrap_stat_summarizer",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
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
    module_name = "_pywrap_tensor_float_32_execution",
    deps = [
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_determinism",
    srcs = ["determinism.cc"],
    hdrs = ["//tensorflow/core/util:determinism_hdr"],
    compatible_with = get_compatible_with_portable(),
    module_name = "_pywrap_determinism",
    deps = [
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_util_port",
    srcs = ["port_wrapper.cc"],
    hdrs = ["//tensorflow/core/util:port_hdrs"],
    module_name = "_pywrap_util_port",
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
    module_name = "_pywrap_transform_graph",
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
        "//tensorflow/c/eager:headers",
        "//tensorflow/python/lib/core:ndarray_tensor_hdr",
        "//tensorflow/python/lib/core:py_exception_registry_hdr",
        "//tensorflow/python/lib/core:safe_ptr_hdr",
    ],
    module_name = "_pywrap_checkpoint_reader",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/python/lib/core:safe_ptr",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/memory",
    ],
)

tf_py_test(
    name = "decorator_utils_test",
    srcs = ["decorator_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "deprecation_test",
    srcs = [
        "deprecated_module.py",
        "deprecated_module_new.py",
        "deprecation_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "keyword_args_test",
    srcs = ["keyword_args_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
    ],
)

py_strict_library(
    name = "tf_export",
    srcs = ["tf_export.py"],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":tf_decorator",
    ],
)

tf_py_test(
    name = "tf_export_test",
    srcs = ["tf_export_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
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
    # TODO(b/138203821): change to "util._tf_stack" once the bug is fixed.
    module_name = "_tf_stack",
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@pybind11",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/framework:pywrap_required_hdrs",
        "//tensorflow/core/platform:path",
    ] + if_static([
        ":stack_trace",
    ]),
)

tf_py_test(
    name = "tf_stack_test",
    srcs = ["tf_stack_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_export",
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
    ],
    module_name = "_function_parameter_canonicalizer_binding_for_test",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

tf_py_test(
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

py_library(
    name = "util",
    srcs = glob(
        ["*.py"],
        exclude = [
            "example_parser*",
            "tf_contextlib.py",
            "tf_should_use.py",
            "tf_export.py",
            "tf_stack.py",
            "tf_decorator.py",
            "*_test.py",
        ],
    ),
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:__pkg__",
        "//third_party/py/tensorflow_core:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tfx:__subpackages__",
    ],
    deps = [
        ":_pywrap_tensor_float_32_execution",
        ":_pywrap_determinism",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via tensorflow.bzl's
        # py_test because not all tensorflow tests use tensorflow.bzl's py_test.
        "//tensorflow/python:global_test_configuration",
        ":tf_decorator",
        ":tf_export",
        "@com_google_protobuf//:protobuf_python",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "@wrapt",
        "//tensorflow/python/util/protobuf",
        "//tensorflow/tools/docs:doc_controls",
        "//tensorflow/tools/compatibility:all_renames_v2",
    ],
)

py_strict_library(
    name = "traceback_utils",
    srcs = ["traceback_utils.py"],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":tf_decorator",
    ],
)

tf_py_test(
    name = "traceback_utils_test",
    size = "small",
    srcs = ["traceback_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:def_function",
    ],
)

tf_py_test(
    name = "object_identity_test",
    size = "small",
    srcs = ["object_identity_test.py"],
    python_version = "PY3",
)

# Placeholder for intenal nest_test comments.
tf_py_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    main = "nest_test.py",
    python_version = "PY3",
    deps = [":nest_test_main_lib"],
)

py_library(
    name = "nest_test_main_lib",
    testonly = True,
    srcs = ["nest_test.py"],
    srcs_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    main = "serialization_test.py",
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "function_utils_test",
    srcs = ["function_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tf_contextlib_test",
    size = "small",
    srcs = ["tf_contextlib_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tf_decorator_test",
    size = "small",
    srcs = ["tf_decorator_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "tf_should_use",
    srcs = ["tf_should_use.py"],
    srcs_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "tf_should_use_test",
    size = "small",
    srcs = ["tf_should_use_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_should_use",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tf_inspect_test",
    size = "small",
    srcs = ["tf_inspect_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
    ],
)

py_library(
    name = "example_parser_configuration",
    srcs = ["example_parser_configuration.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_py_test(
    name = "lock_util_test",
    size = "small",
    srcs = ["lock_util_test.py"],
    main = "lock_util_test.py",
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "fast_module_type",
    srcs = ["fast_module_type.cc"],
    module_name = "fast_module_type",
    deps = [
        "//tensorflow/core/platform:logging",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@pybind11",
    ],
)

tf_py_test(
    name = "fast_module_type_test",
    srcs = ["fast_module_type_test.py"],
    python_version = "PY3",
    deps = [
        ":fast_module_type",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "module_wrapper_test",
    size = "small",
    srcs = ["module_wrapper_test.py"],
    python_version = "PY3",
    deps = [
        ":fast_module_type",
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/tools/compatibility:all_renames_v2",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "example_parser_configuration_test",
    size = "small",
    srcs = ["example_parser_configuration_test.py"],
    main = "example_parser_configuration_test.py",
    python_version = "PY3",
    deps = [
        ":example_parser_configuration",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:parsing_ops",
    ],
)

tf_py_test(
    name = "type_annotations_test",
    size = "small",
    srcs = ["type_annotations_test.py"],
    python_version = "PY3",
    deps = [
    ],
)

filegroup(
    name = "util_hdr",
    srcs = ["util.h"],
)

tf_py_test(
    name = "compat_test",
    srcs = ["compat_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
    ],
)
