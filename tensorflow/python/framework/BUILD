# python/framework package

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_windows",
    "if_oss",
    "if_xla_available",
    "py_test",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pytype_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_static", "tf_additional_xla_deps_py")

visibility = [
    "//tensorflow:__subpackages__",
    "//learning/brain/tfrt:__subpackages__",
    "//third_party/py/tensorflow_numerics:__subpackages__",
]

package(
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
        "//tensorflow/python:platform",
        "//tensorflow/python/util",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = [
        "python_op_gen.cc",
        "python_op_gen_internal.cc",
    ],
    hdrs = [
        "python_op_gen.h",
        "python_op_gen_internal.h",
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

tf_python_pybind_extension(
    name = "_pywrap_python_op_gen",
    srcs = ["python_op_gen_wrapper.cc"],
    module_name = "_pywrap_python_op_gen",
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
    srcs = ["python_op_gen_test.cc"],
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
    name = "for_generated_wrappers",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":device",
        ":device_spec",
        ":dtypes",
        ":function",
        ":op_def_library",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":tensor_shape",
        ":versions",
    ],
)

# What is needed for tf_gen_op_wrapper_py. This is the same as
# "for_generated_wrappers" minus the "function" dep. This is to avoid
# circular dependencies, as "function" uses generated op wrappers.
py_library(
    name = "for_generated_wrappers_v2",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
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
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
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
        ":_pywrap_python_api_dispatcher",
        ":_pywrap_python_api_info",
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_op_gen",
        ":_errors_test_helper",
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
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        "//tensorflow/python/client:_pywrap_debug_events_writer",
        "//tensorflow/python/client:_pywrap_events_writer",
        "//tensorflow/python/util:_pywrap_nest",
        "//tensorflow/python/util:_pywrap_kernel_registry",
        "//tensorflow/python:_pywrap_py_exception_registry",
        "//tensorflow/python/lib/core:_pywrap_py_func",  # TODO(b/142001480): remove once the bug is fixed.
        "//tensorflow/python:_pywrap_quantize_training",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/util:_pywrap_checkpoint_reader",
        "//tensorflow/python/util:_pywrap_stat_summarizer",
        "//tensorflow/python/util:_pywrap_tfprof",
        "//tensorflow/python/util:_pywrap_transform_graph",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python:_pywrap_utils",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:pywrap_mlir",
        "//tensorflow/python/util:util",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python/eager:context",
    ] + if_xla_available([
        "//tensorflow/python:_pywrap_tfcompile",
    ]),
)

py_library(
    name = "c_api_util",
    srcs = ["c_api_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tf_session",
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
        "//tensorflow/python/profiler:trace",
    ],
)

py_library(
    name = "device_spec",
    srcs = ["device_spec.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util",
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
    srcs = ["dtypes.py"],
    srcs_version = "PY3",
    deps = [
        ":_dtypes",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/lib/core:_pywrap_bfloat16",
        "//tensorflow/python/types:doc_typealias",
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
        "//tensorflow/python/util",
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
        "//tensorflow/python/util",
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
        ":test_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
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
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "convert_to_constants",
    srcs = [
        "convert_to_constants.py",
    ],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":dtypes",
        ":ops",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python/grappler:tf_optimizer",
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
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "op_def_library",
    srcs = ["op_def_library.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":op_callbacks",
        ":op_def_registry",
        ":ops",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python/util",
        "@six_archive//:six",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_registry",
    srcs = ["op_def_registry.cc"],
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
    module_name = "_py_context_manager",
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
    ],
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
        "op_def_util.cc",
        "op_def_util_pybind.cc",
    ],
    hdrs = [
        "op_def_util.h",
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
    srcs = ["op_def_util_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
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
            "//tensorflow/core/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:coordination_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
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
            "//tensorflow/core/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:coordination_service_proto_cc_headers_only",
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
    hdrs = ["python_api_dispatcher.h"],
    module_name = "_pywrap_python_api_dispatcher",
    deps = [
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
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
            "//tensorflow/core/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/protobuf:coordination_service_proto_cc_headers_only",
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
        ":indexed_slices",
        ":op_def_registry",
        ":registry",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":traceable_stack",
        ":type_spec",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:platform",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/util",
        "@six_archive//:six",
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
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "indexed_slices",
    srcs = ["indexed_slices.py"],
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
    srcs = ["tensor_conversion_registry.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "func_graph",
    srcs = ["func_graph.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps",
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":ops",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:handle_data_util",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/autograph",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:graph_only_ops",
        "//tensorflow/python/eager:tape",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/util",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
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
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "auto_control_deps_utils",
    srcs = ["auto_control_deps_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
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
        "//tensorflow/python:sendrecv_ops_gen",
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    deps = [
        ":errors",
        ":ops",
        "//tensorflow/python/util",
    ],
)

cuda_py_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":test_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python/util",
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
        "//tensorflow/python:platform",
        "//tensorflow/python/util",
    ] + tf_additional_xla_deps_py(),
)

py_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    srcs_version = "PY3",
    deps = [
        ":config",
        ":ops",
    ],
)

py_library(
    name = "registry",
    srcs = ["registry.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python/util",
        # TODO(mdan): Remove this once the transitive dependency is fixed.
        "//tensorflow/python/util:tf_stack",
    ],
)

py_library(
    name = "smart_cond",
    srcs = ["smart_cond.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_util",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:pywrap_tf_session",
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
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
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
        "//tensorflow/python/types",
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
        "//tensorflow/python:composite_tensor_ops",
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
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/util",
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
        ":tensor_shape",
        "//tensorflow/python:platform",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tensor_spec",
    srcs = ["tensor_spec.py"],
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
    srcs = ["tensor_util.py"],
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
    srcs = ["traceable_stack.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "versions",
    srcs = ["versions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:pywrap_tf_session",
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
    ],
    deps = [
        ":_test_metrics_util",
        ":errors",
        ":for_generated_wrappers",
        ":gpu_util",
        ":random_seed",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:pywrap_sanitizers",
        "//tensorflow/python:pywrap_tf_session",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
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
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
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

# Including this as a dependency will result in tests to use TFRT.
# TODO(b/170139514): Add a curated list of TFRT native ops.
# TODO(kkb): Deprecated by `tfrt_utils`, remove.
py_library(
    name = "is_tfrt_test_true",
    srcs = ["is_tfrt_test_true.py"],
    srcs_version = "PY3",
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
        "//tensorflow/python/util",
    ],
)

py_library(
    name = "test_combinations_lib",
    srcs = ["test_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util",
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
        "//tensorflow/python/profiler:trace",
    ],
)

tf_python_pybind_extension(
    name = "_python_memory_checker_helper",
    srcs = ["python_memory_checker_helper.cc"],
    module_name = "_python_memory_checker_helper",
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
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:client_testlib",
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
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:script_ops",
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
        "noasan",
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
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:logging_ops_gen",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
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
    tags = [
        "no_oss",
        "no_rocm",
    ],
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
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
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
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
        "//tensorflow/python/util",
    ],
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "test_ops.py",
    deps = [":test_ops_kernels"],
)

cc_library(
    name = "test_ops_kernels",
    srcs = ["test_ops.cc"],
    linkstatic = 1,
    deps = [
        "@com_google_absl//absl/time",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_v2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util",
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
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "tensor_spec_test",
    size = "small",
    srcs = ["tensor_spec_test.py"],
    main = "tensor_spec_test.py",
    python_version = "PY3",
    deps = [
        ":for_generated_wrappers",
        ":tensor_spec",
        ":test_lib",
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
    tags = ["no_windows"],
    deps = [
        ":for_generated_wrappers",
        ":framework",
        ":test_lib",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops_gen",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.py"],
    main = "test_util_test.py",
    python_version = "PY3",
    tags = ["no_windows"],
    deps = [
        ":combinations",
        ":errors",
        ":for_generated_wrappers",
        ":test_lib",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
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
    ],
)

tf_python_pybind_extension(
    name = "_memory_checker_test_helper",
    srcs = ["memory_checker_test_helper.cc"],
    module_name = "_memory_checker_test_helper",
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
        ":test_lib",
        "//tensorflow/python:platform_test",
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
    module_name = "_errors_test_helper",
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
        "//tensorflow/python:variables",
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
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:math_ops",
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

pyx_library(
    name = "fast_tensor_util",
    srcs = ["fast_tensor_util.pyx"],
    cc_deps = ["//third_party/py/numpy:headers"],
    py_deps = ["//tensorflow/python:util"],
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
    ],
)

tf_python_pybind_extension(
    name = "_test_metrics_util",
    srcs = ["test_metrics_util.cc"],
    module_name = "_test_metrics_util",
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
