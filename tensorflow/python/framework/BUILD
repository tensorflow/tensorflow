# python/framework package

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "if_oss",
    "if_xla_available",
    "py_test",
    "tf_cc_binary",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "tf_py_test", "tf_python_pybind_extension")
load("//tensorflow:pytype.default.bzl", "pytype_library", "pytype_strict_library")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")
load("//tensorflow/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_static", "tf_additional_xla_deps_py")

visibility = [
    "//tensorflow:__subpackages__",
    "//tensorflow/dtensor:dtensor-internal",
    "//learning/brain/tfrt:__subpackages__",
    "//third_party/py/tensorflow_numerics:__subpackages__",
    "//tensorflow_models/google:__subpackages__",
    "//learning/brain/google/data:__subpackages__",
    "//learning/brain/experimental/tfq:__subpackages__",
    "//learning/brain/google/python/ops:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_cc_shared_object(
    name = "test_file_system.so",
    srcs = ["test_file_system.cc"],
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
    srcs = ["file_system_test.py"],
    data = [":test_file_system.so"],
    main = "file_system_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",  # Path issues due to test environment
        "no_windows",
        "notap",
    ],
    deps = [
        ":for_generated_wrappers",
        ":framework",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/util:compat",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = [
        "python_op_gen.cc",
    ],
    hdrs = [
        "python_op_gen.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        ":python_op_gen_annotator",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

cc_library(
    name = "python_op_gen_annotator",
    srcs = ["python_op_gen_annotator.cc"],
    hdrs = ["python_op_gen_annotator.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":kythe_metadata_proto_cc",
        ":op_reg_offset_proto_cc",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "python_op_gen_annotator_test",
    srcs = ["python_op_gen_annotator_test.cc"],
    deps = [
        ":kythe_metadata_proto_cc",
        ":python_op_gen_annotator",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:protobuf",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_python_op_gen",
    srcs = ["python_op_gen_wrapper.cc"],
    deps = [
        "//tensorflow/python:python_op_gen_headers_lib",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "python_op_gen_main",
    srcs = ["python_op_gen_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        ":python_op_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "op_reg_offset_proto",
    srcs = ["op_reg_offset.proto"],
    cc_api_version = 3,
    visibility = ["//tensorflow:internal"],
)

tf_proto_library(
    name = "kythe_metadata_proto",
    srcs = ["kythe_metadata.proto"],
    cc_api_version = 3,
    visibility = ["//tensorflow:internal"],
)

tf_cc_binary(
    name = "offset_counter",
    srcs = ["offset_counter.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":offset_counter_helper",
        ":op_reg_offset_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/util:command_line_flags",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "offset_counter_helper",
    srcs = ["offset_counter_helper.cc"],
    hdrs = ["offset_counter_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "offset_counter_helper_test",
    srcs = ["offset_counter_helper_test.cc"],
    deps = [
        ":offset_counter_helper",
        ":op_reg_offset_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings:str_format",
    ],
)

py_library(
    name = "for_generated_wrappers",
    deprecation = "Depending on this target can cause build dependency cycles. Depend on the fine-grained sub-targets instead.",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":byte_swap_tensor",
        ":device",
        ":device_spec",
        ":dtypes",
        ":function",
        ":op_def_library",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":versions",
    ],
)

# What is needed for tf_gen_op_wrapper_py. This is the same as
# "for_generated_wrappers" minus the "function" dep. This is to avoid
# circular dependencies, as "function" uses generated op wrappers.
py_library(
    name = "for_generated_wrappers_v2",
    deprecation = "Depending on this target can cause build dependency cycles. Depend on the fine-grained sub-targets instead.",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":byte_swap_tensor",
        ":constant_op",
        ":device",
        ":device_spec",
        ":dtypes",
        ":op_def_library",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":tensor_shape",
        ":versions",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:execute",
        "//tensorflow/security/fuzzing/py:annotation_types",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "subscribe",
    srcs = ["subscribe.py"],
    srcs_version = "PY3",
    deps = [
        ":ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_library(
    name = "framework",
    srcs = [
        "framework_lib.py",
        "graph_io.py",
        "importer.py",
        "load_library.py",
        "meta_graph.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_errors_test_helper",
        ":_pywrap_python_api_dispatcher",
        ":_pywrap_python_api_info",
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_op_gen",
        ":composite_tensor",
        ":config",
        ":convert_to_constants",
        ":cpp_shape_inference_proto_py",
        ":errors",
        ":fast_tensor_util",
        ":for_generated_wrappers",
        ":function",
        ":graph_util",
        ":random_seed",
        ":sparse_tensor",
        ":tensor",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        "//tensorflow/python:_pywrap_py_exception_registry",
        "//tensorflow/python:_pywrap_quantize_training",
        "//tensorflow/python:_pywrap_utils",
        "//tensorflow/python:lib",
        "//tensorflow/python:pywrap_mlir",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/client:_pywrap_debug_events_writer",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/lib/core:_pywrap_py_func",  # TODO(b/142001480): remove once the bug is fixed.
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ] + if_xla_available([
        "//tensorflow/python:_pywrap_tfcompile",
    ]),
)

py_library(
    name = "byte_swap_tensor",
    srcs = ["byte_swap_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
    ],
)

py_library(
    name = "c_api_util",
    srcs = ["c_api_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_decorator",
    ],
)

py_library(
    name = "common_shapes",
    srcs = ["common_shapes.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_shape",
    ],
)

py_library(
    name = "constant_op",
    srcs = ["constant_op.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "device_spec",
    srcs = ["device_spec.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "device",
    srcs = ["device.py"],
    srcs_version = "PY3",
)

tf_python_pybind_extension(
    name = "_dtypes",
    srcs = ["dtypes.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
        "@pybind11",
    ],
)

py_library(
    name = "dtypes",
    srcs = ["dtypes.py"],
    srcs_version = "PY3",
    deps = [
        ":_dtypes",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/lib/core:_pywrap_bfloat16",
        "//tensorflow/python/lib/core:_pywrap_custom_casts",
        "//tensorflow/python/lib/core:_pywrap_float8",
        "//tensorflow/python/types:doc_typealias",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "errors",
    srcs = [
        "errors.py",
        "errors_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":c_api_util",
        "//tensorflow/python:_pywrap_py_exception_registry",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "error_interpolation",
    srcs = [
        "error_interpolation.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "function",
    srcs = ["function.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":graph_to_function_def",
        ":op_def_registry",
        ":ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_decorator",
    ],
)

py_library(
    name = "graph_to_function_def",
    srcs = ["graph_to_function_def.py"],
    srcs_version = "PY3",
    deps = [
        ":op_def_registry",
        "//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "function_def_to_graph",
    srcs = ["function_def_to_graph.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":cpp_shape_inference_proto_py",
        ":framework",
        ":function",
        ":ops",
        ":tensor_shape",
        ":versions",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "function_def_to_graph_test",
    size = "small",
    srcs = ["function_def_to_graph_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":function",
        ":function_def_to_graph",
        ":graph_to_function_def",
        ":op_def_library",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":test_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
    ],
)

py_library(
    name = "graph_util",
    srcs = [
        "graph_util.py",
        "graph_util_impl.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_proto_comparators",
        ":dtypes",
        ":ops",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "convert_to_constants",
    srcs = [
        "convert_to_constants.py",
    ],
    srcs_version = "PY3",
    visibility = visibility + [
        "//learning/deepmind/deepfunc:__subpackages__",
        "//platforms/darwinn:__subpackages__",
    ],
    deps = [
        ":dtypes",
        ":ops",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "kernels",
    srcs = [
        "kernels.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python/util:compat",
    ],
)

py_library(
    name = "op_def_library",
    srcs = ["op_def_library.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":op_callbacks",
        ":op_def_library_pybind",
        ":op_def_registry",
        ":ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_decorator",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_library_pybind",
    srcs = ["op_def_library_pybind.cc"],
    deps = [
        ":op_def_util_headers",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "op_def_library_pybind",
    srcs = ["op_def_library_pybind.py"],
    srcs_version = "PY3",
    deps = [
        ":_op_def_library_pybind",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_registry",
    srcs = ["op_def_registry.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_library(
    name = "op_def_registry",
    srcs = ["op_def_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":_op_def_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

cc_library(
    name = "py_context_manager",
    srcs = ["py_context_manager.cc"],
    hdrs = ["py_context_manager.h"],
    deps = [
        "//tensorflow/core:lib",  # for core/platform/logging.h
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
    ],
)

# Pybind extension used by py_context_manager_test.
tf_python_pybind_extension(
    name = "_py_context_manager",
    srcs = ["py_context_manager_pybind.cc"],
    deps = [
        ":py_context_manager",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_test(
    name = "py_context_manager_test",
    srcs = ["py_context_manager_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_py_context_manager",
        "//tensorflow/python/platform:test",
    ],
)

cc_library(
    name = "op_def_util_headers",
    textual_hdrs = ["op_def_util.h"],
    visibility = ["//tensorflow:internal"],
)

cc_library(
    name = "op_def_util_cc",
    srcs = ["op_def_util.cc"],
    hdrs = ["op_def_util.h"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/strings",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_util",
    srcs = ["op_def_util_pybind.cc"],
    deps = [
        ":op_def_util_headers",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_test(
    name = "op_def_util_test",
    srcs = ["op_def_util_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
    deps = ["//tensorflow/python/platform:test"],
)

cc_library(
    name = "python_api_parameter_converter",
    srcs = ["python_api_parameter_converter.cc"],
    hdrs = ["python_api_parameter_converter.h"],
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
        "@com_google_absl//absl/types:span",
    ],
)

# Note: this target is only used by python_api_parameter_converter_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_parameter_converter",
    srcs = ["python_api_parameter_converter_wrapper.cc"],
    hdrs = [
        "op_def_util.h",
        "python_api_info.h",
        "python_api_parameter_converter.h",
        "python_tensor_converter.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "//tensorflow/tsl/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/tsl/python/lib/core:numpy_hdr",
    ],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/config:flags_headers",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
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

tf_py_test(
    name = "python_api_parameter_converter_test",
    srcs = ["python_api_parameter_converter_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_tensor_converter",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
    ],
)

cc_library(
    name = "python_api_info",
    srcs = ["python_api_info.cc"],
    hdrs = ["python_api_info.h"],
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
        "@com_google_absl//absl/types:span",
    ],
)

# Note: this target is only used by python_api_info_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_info",
    srcs = ["python_api_info_wrapper.cc"],
    hdrs = [
        "op_def_util.h",
        "python_api_info.h",
        "python_tensor_converter.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "//tensorflow/tsl/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/tsl/python/lib/core:numpy_hdr",
    ],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/config:flags_headers",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_static(
        extra_deps = [
            "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/tsl/protobuf:coordination_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_test(
    name = "python_api_info_test",
    srcs = ["python_api_info_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_info",
        ":_pywrap_python_tensor_converter",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

cc_library(
    name = "python_api_dispatcher",
    srcs = ["python_api_dispatcher.cc"],
    hdrs = ["python_api_dispatcher.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/platform:logging",
        "//tensorflow/python/lib/core:py_util",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//tensorflow/python/util:cpp_python_util",
        "//tensorflow/python/util:function_parameter_canonicalizer",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_python_api_dispatcher",
    srcs = ["python_api_dispatcher_wrapper.cc"],
    hdrs = [
        "python_api_dispatcher.h",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    deps = [
        "//tensorflow/python/util:function_parameter_canonicalizer_hdrs",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@pybind11",
    ],
)

tf_py_test(
    name = "python_api_dispatcher_test",
    srcs = ["python_api_dispatcher_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
    deps = [
        ":_pywrap_python_api_dispatcher",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

cc_library(
    name = "python_tensor_converter",
    srcs = ["python_tensor_converter.cc"],
    hdrs = ["python_tensor_converter.h"],
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
    srcs = ["python_tensor_converter_wrapper.cc"],
    hdrs = [
        "python_tensor_converter.h",
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "//tensorflow/tsl/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/tsl/python/lib/core:numpy_hdr",
    ],
    deps = [
        "//tensorflow/c:pywrap_required_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_headers_lib",
        "//tensorflow/core/config:flags_headers",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_static(
        extra_deps = [
            "//tensorflow/tsl/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/tsl/protobuf:coordination_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_test(
    name = "python_tensor_converter_test",
    srcs = ["python_tensor_converter_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_tensor_converter",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

py_library(
    name = "ops",  # "ops" is already the name of a deprecated target
    srcs = ["ops.py"],
    srcs_version = "PY3",
    deps = [
        ":_op_def_util",
        ":c_api_util",
        ":device",
        ":dtypes",
        ":error_interpolation",
        ":op_callbacks",
        ":op_def_registry",
        ":registry",
        ":stack",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":traceable_stack",
        ":type_spec",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:decorator_utils",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:lazy_loader",
        "//tensorflow/python/util:lock_util",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_stack",
        "//tensorflow/python/util:traceback_utils",
    ],
)

py_library(
    name = "op_callbacks",
    srcs = ["op_callbacks.py"],
    srcs_version = "PY3",
)

cuda_py_test(
    name = "op_callbacks_test",
    srcs = ["op_callbacks_test.py"],
    python_version = "PY3",
    deps = [
        ":sparse_tensor",
        ":test_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "indexed_slices",
    srcs = ["indexed_slices.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":composite_tensor_gradient",
        ":dtypes",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_spec",
        ":type_spec",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "indexed_slices_test",
    srcs = ["indexed_slices_test.py"],
    deps = [
        ":composite_tensor_gradient",
        "//tensorflow/python:ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ],
)

py_library(
    name = "tensor_conversion",
    srcs = ["tensor_conversion.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_conversion_registry",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "tensor_conversion_registry",
    srcs = ["tensor_conversion_registry.py"],
    srcs_version = "PY3",
    # TODO(b/266747022): remove extra visibility
    visibility = visibility + ["//learning/brain/experimental:__subpackages__"],
    deps = [
        ":dtypes",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "func_graph",
    srcs = ["func_graph.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps",
        ":composite_tensor",
        ":dtypes",
        ":errors",
        ":indexed_slices",
        ":ops",
        ":tensor_spec",
        ":type_spec",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/autograph",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "auto_control_deps",
    srcs = ["auto_control_deps.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps_utils",
        ":ops",
        ":sparse_tensor",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator",
    ],
)

py_library(
    name = "auto_control_deps_utils",
    srcs = ["auto_control_deps_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        "//tensorflow/python/util:object_identity",
    ],
)

tf_py_test(
    name = "auto_control_deps_test",
    size = "small",
    srcs = ["auto_control_deps_test.py"],
    python_version = "PY3",
    deps = [
        ":auto_control_deps",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_switch_case",
        "//tensorflow/python:sendrecv_ops_gen",
        "//tensorflow/python:while_loop",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":ops",
        "//tensorflow/python/util:_pywrap_determinism",
        "//tensorflow/python/util:_pywrap_tensor_float_32_execution",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    python_version = "PY3",
    tags = [
        "multi_gpu",
        "no_pip",
    ],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":framework",
        ":ops",
        ":test_lib",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

cuda_py_test(
    name = "config_vgpu_test",
    size = "small",
    srcs = ["config_vgpu_test.py"],
    python_version = "PY3",
    shard_count = 10,  # use a large number to ensure no 2 tests run in the same process.
    tags = [
        "multi_gpu",
        "no_pip",
    ],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":errors",
        ":ops",
        ":test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_test(
    name = "config_test_tpu",
    size = "small",
    srcs = ["config_test.py"],
    disable_tfrt = True,  # TODO(tfrt-devs): Update this test to call initialize_tpu_system
    main = "config_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":test_ops",
        "//tensorflow/python:client_testlib",
    ] + tf_additional_xla_deps_py(),
)

py_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    srcs_version = "PY3",
    deps = [
        ":config",
        ":ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "registry",
    srcs = ["registry.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
    ],
)

py_library(
    name = "smart_cond",
    srcs = ["smart_cond.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_util",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_case",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "smart_cond_test",
    size = "small",
    srcs = ["smart_cond_test.py"],
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":ops",
        ":smart_cond",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python/platform:test",
    ],
)

py_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":ops",
        ":tensor_util",
        ":type_spec",
        ":type_spec_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":dtypes",
        ":tensor_util",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_test(
    name = "composite_tensor_test",
    srcs = ["composite_tensor_test.py"],
    main = "composite_tensor_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "composite_tensor_gradient",
    srcs = ["composite_tensor_gradient.py"],
    deps = [
        ":composite_tensor",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "extension_type",
    srcs = ["extension_type.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":dtypes",
        ":extension_type_field",
        ":immutable_dict",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":type_spec",
        ":type_spec_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:composite_tensor_ops",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "extension_type_field",
    srcs = ["extension_type_field.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":immutable_dict",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":type_spec",
        "//tensorflow/python/util:type_annotations",
    ],
)

py_library(
    name = "immutable_dict",
    srcs = ["immutable_dict.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "tensor_shape",
    srcs = ["tensor_shape.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_library(
    name = "type_spec",
    srcs = ["type_spec.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":tensor_conversion_registry",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/types:trace",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_library(
    name = "type_spec_registry",
    srcs = ["type_spec_registry.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//third_party/py/tensorflow_gnn:__subpackages__"],
    deps = [
        "//tensorflow/python/types:internal",
    ],
)

pytype_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    srcs_version = "PY3",
    visibility = [
        # Limit the exposure of fulltype (which is a graph/C++ concept) to Python.
        "//tensorflow/python/data/ops:__pkg__",
        "//tensorflow/python/ops/ragged:__pkg__",
        "//tensorflow/python/ops/structured:__pkg__",
    ],
    deps = [
        ":type_spec",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/structured:structured_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "stack",
    srcs = ["stack.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "tensor",
    srcs = ["tensor.py"],
    deps = [
        ":common_shapes",
        ":constant_op",
        ":dtypes",
        ":tensor_shape",
        ":tensor_util",
        ":type_spec",
        ":type_spec_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tensor_spec",
    srcs = ["tensor_spec.py"],
    deps = [
        ":tensor",
    ],
)

py_library(
    name = "tensor_util",
    srcs = ["tensor_util.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "traceable_stack",
    srcs = ["traceable_stack.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_library(
    name = "versions",
    srcs = ["versions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "gpu_util",
    srcs = ["gpu_util.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "test_lib",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow_estimator/python/estimator:__subpackages__",
        "//third_party/py/keras:__subpackages__",
    ],
    deps = [
        ":_test_metrics_util",
        ":errors",
        ":for_generated_wrappers",
        ":gpu_util",
        ":random_seed",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:pywrap_sanitizers",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sync_ops_gen",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/ops/ragged:ragged_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:traceback_utils",
        "//tensorflow/python/util/protobuf",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# Including this as a dependency will result in tests using
# :test_lib to use XLA.
py_library(
    name = "is_xla_test_true",
    srcs = ["is_xla_test_true.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:internal",
        "//third_party/py/keras:__subpackages__",
        "//third_party/py/keras_cv:__subpackages__",
    ],
)

# Including this as a dependency will result in tests using
# :test_lib to use MLIR.
py_library(
    name = "is_mlir_bridge_test_true",
    srcs = ["is_mlir_bridge_test_true.py"],
    srcs_version = "PY3",
    visibility = visibility,
)

# Including this as a dependency will result in tests using
# :test_lib to NOT use MLIR.
py_library(
    name = "is_mlir_bridge_test_false",
    srcs = ["is_mlir_bridge_test_false.py"],
    srcs_version = "PY3",
    visibility = visibility,
)

py_library(
    name = "tfrt_utils",
    srcs = ["tfrt_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "combinations",
    srcs = ["combinations.py"],
    srcs_version = "PY3",
    deps = [
        ":ops",
        ":test_combinations_lib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "test_combinations_lib",
    srcs = ["test_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "test_combinations_test",
    srcs = ["test_combinations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_combinations_lib",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "memory_checker",
    srcs = [
        "memory_checker.py",
        "python_memory_checker.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_python_memory_checker_helper",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/util:tf_decorator",
    ],
)

tf_python_pybind_extension(
    name = "_python_memory_checker_helper",
    srcs = ["python_memory_checker_helper.cc"],
    deps = [
        "@pybind11",
    ],
)

tf_py_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    main = "constant_op_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

tf_py_test(
    name = "registry_test",
    size = "small",
    srcs = ["registry_test.py"],
    main = "registry_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.py"],
    main = "errors_test.py",
    python_version = "PY3",
    tags = ["no_windows"],  # TODO(b/184424727): Enable this test on Windows.
    deps = [
        ":_errors_test_helper",
        ":errors",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_test(
    name = "error_interpolation_test",
    size = "small",
    srcs = ["error_interpolation_test.py"],
    main = "error_interpolation_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":error_interpolation",
        ":traceable_stack",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_assert",
    ],
)

tf_py_test(
    name = "subscribe_test",
    size = "small",
    srcs = ["subscribe_test.py"],
    main = "subscribe_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":subscribe",
        ":test_lib",
        "//tensorflow/python:cond",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:variable_v1",
    ],
)

tf_py_test(
    name = "proto_test",
    size = "small",
    srcs = ["proto_test.py"],
    main = "proto_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "function_test",
    size = "medium",
    srcs = ["function_test.py"],
    python_version = "PY3",
    shard_count = 10,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":errors",
        ":for_generated_wrappers",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_ops_gen",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:logging_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "versions_test",
    size = "small",
    srcs = ["versions_test.py"],
    main = "versions_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "importer_test",
    size = "small",
    srcs = ["importer_test.py"],
    main = "importer_test.py",
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "meta_graph_testdata",
    srcs = [
        "testdata/metrics_export_meta_graph.pb",
    ],
    visibility = if_oss(["//visibility:public"]),
)

tf_py_test(
    name = "meta_graph_test",
    size = "small",
    srcs = ["meta_graph_test.py"],
    data = [":meta_graph_testdata"],
    main = "meta_graph_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":for_generated_wrappers",
        ":framework",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/platform:gfile",
    ],
)

cuda_py_test(
    name = "node_file_writer_test",
    size = "small",
    srcs = ["node_file_writer_test.py"],
    python_version = "PY3",
    # On Windows, fails in tempfile.TemporaryDirectory.cleanup() with error: "The directory is not
    # empty", so test is disabled on Windows.
    # TODO(reedwm): Debug this.
    tags = ["no_windows"],
    deps = [
        ":test_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_test(
    name = "traceable_stack_test",
    size = "small",
    srcs = ["traceable_stack_test.py"],
    main = "traceable_stack_test.py",
    python_version = "PY3",
    deps = [
        ":test_lib",
        ":test_ops",
        ":traceable_stack",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/util:tf_decorator",
    ],
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "test_ops.py",
    deps = [":test_ops_kernels"],
)

tf_kernel_library(
    name = "test_ops_kernels",
    srcs = ["test_ops.cc"],
    hdrs = ["test_ops.h"],
    gpu_srcs = ["test_ops.cu.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/time",
    ] + if_static(["//tensorflow/core/platform:tensor_float_32_utils"]),
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "test_ops_2",
    out = "test_ops_2.py",
    deps = [":test_ops_2_kernels"],
)

cc_library(
    name = "test_ops_2_kernels",
    srcs = ["test_ops_2.cc"],
    linkstatic = 1,
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_py_test(
    name = "common_shapes_test",
    size = "small",
    srcs = ["common_shapes_test.py"],
    main = "common_shapes_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.py"],
    main = "ops_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # test_ops_2 is not available in pip.
    deps = [
        ":errors",
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        ":test_ops",
        ":test_ops_2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:cond",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python:while_v2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util:compat",
    ],
)

# Test that op_allowlist works with ops with namespaces
tf_gen_op_wrapper_py(
    name = "test_namespace_ops",
    out = "test_namespace_ops.py",
    op_allowlist = ["Namespace>TestStringOutput"],
    deps = [
        ":test_ops_kernels",
    ],
)

tf_py_test(
    name = "op_allowlist_namespace_test",
    size = "small",
    srcs = ["op_allowlist_namespace_test.py"],
    main = "op_allowlist_namespace_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # Avoid a missing test dependency error
    deps = [
        ":errors",
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        ":test_namespace_ops",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_test(
    name = "python_op_gen_annotation_test",
    size = "small",
    srcs = ["python_op_gen_annotation_test.py"],
    main = "python_op_gen_annotation_test.py",
    python_version = "PY3",
    deps = [
        "//tensorflow/python/platform:test",
        "//tensorflow/security/fuzzing/py:annotation_types",
    ],
)

tf_py_test(
    name = "ops_enable_eager_test",
    size = "small",
    srcs = ["ops_enable_eager_test.py"],
    main = "ops_enable_eager_test.py",
    python_version = "PY3",
    deps = [
        ":framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
    ],
)

tf_py_test(
    name = "extension_type_test",
    size = "small",
    srcs = ["extension_type_test.py"],
    main = "extension_type_test.py",
    python_version = "PY3",
    deps = [
        ":extension_type",
        ":test_lib",
        ":type_spec_registry",
        ":type_utils",
        "//tensorflow/python:cond",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "extension_type_field_test",
    size = "small",
    srcs = ["extension_type_field_test.py"],
    main = "extension_type_field_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/184565242) Enable pip once we add a module that depends on this.
    deps = [
        ":extension_type_field",
        ":test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "immutable_dict_test",
    size = "small",
    srcs = ["immutable_dict_test.py"],
    main = "immutable_dict_test.py",
    python_version = "PY3",
    deps = [
        ":immutable_dict",
        ":test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "tensor_shape_test",
    size = "small",
    srcs = ["tensor_shape_test.py"],
    main = "tensor_shape_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "type_spec_test",
    size = "small",
    srcs = ["type_spec_test.py"],
    main = "type_spec_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":test_lib",
        ":type_spec",
        ":type_spec_registry",
        ":type_utils",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "tensor_test",
    size = "small",
    srcs = ["tensor_test.py"],
    main = "tensor_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":tensor",
        ":test_lib",
        ":type_utils",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "sparse_tensor_test",
    size = "small",
    srcs = ["sparse_tensor_test.py"],
    main = "sparse_tensor_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        ":type_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "device_spec_test",
    size = "small",
    srcs = ["device_spec_test.py"],
    main = "device_spec_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "device_test",
    size = "small",
    srcs = ["device_test.py"],
    main = "device_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "random_seed_test",
    size = "small",
    srcs = ["random_seed_test.py"],
    main = "random_seed_test.py",
    python_version = "PY3",
    deps = [
        ":framework",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tensor_util_test",
    size = "small",
    srcs = ["tensor_util_test.py"],
    main = "tensor_util_test.py",
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:shape_util",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:variable_v1",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.py"],
    main = "test_util_test.py",
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":combinations",
        ":errors",
        ":for_generated_wrappers",
        ":test_lib",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util/protobuf:compare_test_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "memory_checker_test",
    size = "medium",
    srcs = ["memory_checker_test.py"],
    main = "memory_checker_test.py",
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
        ":test_lib",
        # TODO(kkb): Find more appropriate place to add `memory_checker` as deps
        # Adding directly to `test_lib` caused a Colab binary size
        # regression b/149433910 .
        ":memory_checker",
        ":_memory_checker_test_helper",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_python_pybind_extension(
    name = "_memory_checker_test_helper",
    srcs = ["memory_checker_test_helper.cc"],
    deps = [
        "@pybind11",
    ],
)

tf_py_test(
    name = "dtypes_test",
    size = "small",
    srcs = ["dtypes_test.py"],
    main = "dtypes_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":test_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["op_def_library_test.py"],
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":op_def_library_pybind",
        ":test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_test(
    name = "kernels_test",
    size = "small",
    srcs = ["kernels_test.py"],
    main = "kernels_test.py",
    python_version = "PY3",
    deps = [
        ":kernels",
        ":test_lib",
        ":test_ops",
        "//tensorflow/python:platform_test",
    ],
)

tf_proto_library(
    name = "cpp_shape_inference_proto",
    srcs = ["cpp_shape_inference.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
    # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    visibility = ["//tensorflow:internal"],
)

tf_python_pybind_extension(
    name = "_errors_test_helper",
    srcs = ["errors_test_helper.cc"],
    deps = [
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

tf_py_test(
    name = "c_api_util_test",
    size = "small",
    srcs = ["c_api_util_test.py"],
    python_version = "PY3",
    deps = [
        ":c_api_util",
        ":test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["graph_util_test.py"],
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":framework",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops_gen",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_test(
    name = "convert_to_constants_test",
    size = "medium",
    srcs = ["convert_to_constants_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_rocm"],
    deps = [
        ":convert_to_constants",
        ":test_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:control_flow_switch_case",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ref_variable",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
    ],
)

cuda_py_test(
    name = "graph_building_benchmark",
    size = "medium",
    srcs = ["graph_building_benchmark.py"],
    main = "graph_building_benchmark.py",
    python_version = "PY3",
    deps = [
        ":dtypes",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python/eager:context",
    ],
)

tf_python_pybind_extension(
    name = "_proto_comparators",
    srcs = ["proto_comparators.cc"],
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

pyx_library(
    name = "fast_tensor_util",
    srcs = ["fast_tensor_util.pyx"],
    cc_deps = ["//third_party/py/numpy:headers"],
    py_deps = ["//tensorflow/python/util:compat"],
)

py_test(
    name = "tf2_test",
    srcs = ["tf2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_python_pybind_extension(
    name = "_test_metrics_util",
    srcs = ["test_metrics_util.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "cpp_shape_inference_proto_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     deps = [":cpp_shape_inference_proto"],
# )
# copybara:uncomment_end

py_library(
    name = "summary_test_util",
    testonly = True,
    srcs = ["summary_test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:lib",
        "//tensorflow/python/platform:gfile",
    ],
)
