# python/framework package

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_tools",
    "if_not_windows",
    "if_oss",
    "if_xla_available",
    "tf_cc_binary",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "stripped_cc_info", "tf_py_strict_test", "tf_python_framework_friends", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap", "if_static", "tf_additional_xla_deps_py")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")
load(
    "//tensorflow/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

visibility = tf_python_framework_friends()  # buildifier: disable=package-on-top

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

cc_library(
    name = "test_file_system",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:null_file_system",
    ],
)

stripped_cc_info(
    name = "test_file_system_stripped",
    deps = [":test_file_system"],
)

tf_cc_shared_object(
    name = "test_file_system.so",
    srcs = if_pywrap(if_false = ["test_file_system.cc"]),
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = if_pywrap(
        if_false = [
            "@eigen_archive//:eigen3",
            "//tensorflow/core:framework_headers_lib",
        ],
        if_true = [
            ":test_file_system_stripped",
            "//tensorflow/python:_pywrap_tensorflow_common",
        ],
    ) + ["@com_google_protobuf//:protobuf_headers"],
)

tf_py_strict_test(
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
        ":dtypes",
        ":load_library",
        ":test_lib",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:io_ops",
        "//tensorflow/python/platform:client_testlib",
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
        "@local_tsl//tsl/platform:protobuf",
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_op_gen.pyi",
    ],
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

tf_cc_test(
    name = "python_op_gen_test",
    srcs = ["python_op_gen_test.cc"],
    deps = [
        ":kythe_metadata_proto_cc",
        ":op_reg_offset_proto_cc",
        ":python_op_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "op_reg_offset_proto",
    srcs = ["op_reg_offset.proto"],
    visibility = ["//tensorflow:internal"],
)

tf_proto_library(
    name = "kythe_metadata_proto",
    srcs = ["kythe_metadata.proto"],
    visibility = ["//tensorflow:internal"],
)

tf_cc_binary(
    name = "offset_counter",
    srcs = ["offset_counter.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":offset_counter_helper",
        ":op_reg_offset_proto_cc",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:types",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "offset_counter_helper",
    srcs = ["offset_counter_helper.cc"],
    hdrs = ["offset_counter_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:types",
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

# Do not depend on this rule! Depend on the fine-grained sub-targets instead.
py_strict_library(
    name = "for_generated_wrappers",
    srcs_version = "PY3",
    tags = ["avoid_dep"],
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

# This rule should only be depended on by tf_gen_op_wrapper_py.
# Do not depend on this rule! Depend on the fine-grained sub-targets instead.
# This is the same as "for_generated_wrappers" minus the "function" dep.
# This is to avoid circular dependencies, as "function" uses generated op wrappers.
py_strict_library(
    name = "for_generated_wrappers_v2",
    srcs_version = "PY3",
    tags = ["avoid_dep"],
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
        "@pypi_typing_extensions//:pkg",
    ],
)

py_strict_library(
    name = "subscribe",
    srcs = ["subscribe.py"],
    srcs_version = "PY3",
    deps = [
        ":ops",
        ":tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "byte_swap_tensor",
    srcs = ["byte_swap_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        "//tensorflow/core:protos_all_py",
    ],
)

py_strict_library(
    name = "c_api_util",
    srcs = ["c_api_util.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_contextlib",
    ],
)

py_strict_library(
    name = "common_shapes",
    srcs = ["common_shapes.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_shape",
    ],
)

py_strict_library(
    name = "constant_op",
    srcs = [
        "constant_op.py",
        "constant_tensor_conversion.py",
    ],
    srcs_version = "PY3",
    visibility = visibility + [
        "//smartass:__subpackages__",
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":dtypes",
        ":ops",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "device_spec",
    srcs = ["device_spec.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//third_party/py/tf_slim:__subpackages__"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "device",
    srcs = ["device.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":device_spec",
        "//tensorflow/python:tf2",
    ],
)

tf_python_pybind_extension(
    name = "_dtypes",
    srcs = ["dtypes.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_dtypes.pyi",
    ],
    visibility = visibility + ["//third_party/py/tensorfn:__subpackages__"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "@eigen_archive//:eigen3",
        "@pybind11",
    ],
)

py_strict_library(
    name = "dtypes",
    srcs = ["dtypes.py"],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//smartass:__subpackages__",
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/mlperf:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//tensorflow:internal",
    # "//tensorflow_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_dtypes",
        # copybara:uncomment_begin(google-only)
        # ":cpp_shape_inference_proto_py",
        # "//third_party/py/numpy",
        # copybara:uncomment_end
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/types:trace",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/types:doc_typealias",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "@ml_dtypes",
    ],
)

py_strict_library(
    name = "errors",
    srcs = [
        "errors.py",
        "errors_impl.py",
    ],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//smartass/brain/ops:__subpackages__",
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//tensorflow:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":c_api_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:_pywrap_py_exception_registry",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "error_interpolation",
    srcs = [
        "error_interpolation.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:tf_stack",
    ],
)

tf_cc_binary(
    name = "_native_proto_caster",
    linkshared = True,
    deps = ["@pybind11_protobuf//pybind11_protobuf:native_proto_caster"],
)

py_strict_library(
    name = "function",
    srcs = ["function.py"],
    data = [":_native_proto_caster"],  # copybara:comment
    srcs_version = "PY3",
    visibility = visibility + [
        "//smartass/brain:__subpackages__",
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/mlperf:__subpackages__",
    ],
    deps = [
        ":c_api_util",
        ":dtypes",
        ":graph_to_function_def",
        ":ops",
        ":tensor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_inspect",
        # copybara:uncomment "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_strict_library(
    name = "graph_to_function_def",
    srcs = ["graph_to_function_def.py"],
    srcs_version = "PY3",
    deps = [
        ":op_def_registry",
        "//tensorflow/core:protos_all_py",
    ],
)

py_strict_library(
    name = "function_def_to_graph",
    srcs = ["function_def_to_graph.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":cpp_shape_inference_proto_py",
        ":func_graph",
        ":importer",
        ":ops",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:resource_variable_ops",
    ],
)

pytype_strict_library(
    name = "strict_mode",
    srcs = ["strict_mode.py"],
    deps = [
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "function_def_to_graph_test",
    size = "small",
    srcs = ["function_def_to_graph_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":function_def_to_graph",
        ":graph_to_function_def",
        ":op_def_library",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":test_lib",
        ":test_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "graph_util",
    srcs = [
        "graph_util.py",
        "graph_util_impl.py",
    ],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + ["//tensorflow:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_proto_comparators",
        ":dtypes",
        ":ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "convert_to_constants",
    srcs = [
        "convert_to_constants.py",
    ],
    srcs_version = "PY3",
    visibility = visibility + [
        "//learning/deepmind/deepfunc:__subpackages__",
        "//platforms/darwinn:__subpackages__",
        "//tensorflow:internal",
    ],
    deps = [
        ":dtypes",
        ":errors",
        ":graph_util",
        ":ops",
        ":tensor_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "kernels",
    srcs = [
        "kernels.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "op_def_library",
    srcs = ["op_def_library.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":op_callbacks",
        ":op_def_library_pybind",
        ":op_def_registry",
        ":ops",
        ":tensor",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:_pywrap_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_contextlib",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_library_pybind",
    srcs = ["op_def_library_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_op_def_library_pybind.pyi",
    ],
    deps = if_pywrap(
        if_false = [":op_def_util_headers"],
        if_true = [":op_def_util_cc"],
    ) + [
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_op_def_registry.pyi",
    ],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_strict_library(
    name = "op_def_registry",
    srcs = ["op_def_registry.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":_op_def_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_strict_library(
    name = "override_binary_operator",
    srcs = ["override_binary_operator.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        ":ops",
        ":tensor",
        ":tensor_shape",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops/numpy_ops:np_dtypes",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:traceback_utils",
        "//third_party/py/numpy",
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_py_context_manager.pyi",
    ],
    deps = [
        ":py_context_manager",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "py_context_manager_test",
    srcs = ["py_context_manager_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_py_context_manager",
        ":test_lib",
        "//tensorflow/python/platform:test",
    ],
)

cc_library(
    name = "op_def_util_headers",
    textual_hdrs = ["op_def_util.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
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

tf_python_pybind_extension(
    name = "_op_def_util",
    srcs = ["op_def_util_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_op_def_util.pyi",
    ],
    deps = if_pywrap(
        if_false = [":op_def_util_headers"],
        if_true = [":op_def_util_cc"],
    ) + [
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:safe_pyobject_ptr",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "op_def_util_test",
    srcs = ["op_def_util_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
    deps = [
        ":_op_def_util",
        ":dtypes",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
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
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ] + if_pywrap(
        if_false = [
            "op_def_util.h",
            "python_api_info.h",
            "python_api_parameter_converter.h",
            "python_tensor_converter.h",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_api_parameter_converter.pyi",
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
    ] + if_pywrap(
        if_true = [
            ":python_api_parameter_converter",
            ":python_api_info",
            ":op_def_util_cc",
            ":python_tensor_converter",
        ],
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

tf_py_strict_test(
    name = "python_api_parameter_converter_test",
    srcs = ["python_api_parameter_converter_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_info",
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_tensor_converter",
        ":constant_op",
        ":dtypes",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
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
        "//tensorflow/c:headers",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/c/experimental/ops:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/python/eager:pywrap_required_hdrs",
        "//tensorflow/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ] + if_pywrap(
        if_false = [
            "op_def_util.h",
            "python_api_info.h",
            "python_tensor_converter.h",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_api_info.pyi",
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
    ] + if_pywrap(
        if_true = [
            ":python_api_info",
            ":op_def_util_cc",
            ":python_tensor_converter",
        ],
    ) + if_static(
        extra_deps = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_strict_test(
    name = "python_api_info_test",
    srcs = ["python_api_info_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_info",
        ":constant_op",
        ":test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
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
    # This data is needed to add hermetic CUDA tools in python runfiles.
    data = if_cuda_tools(
        [
            "@cuda_nvcc//:ptxas",
            "@cuda_nvcc//:nvvm",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_api_dispatcher.pyi",
    ],
    deps = [
        "//tensorflow/python/util:function_parameter_canonicalizer_hdrs",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ] + if_pywrap([":python_api_dispatcher"]),
)

tf_py_strict_test(
    name = "python_api_dispatcher_test",
    srcs = ["python_api_dispatcher_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
    deps = [
        ":_pywrap_python_api_dispatcher",
        ":constant_op",
        ":tensor",
        ":test_lib",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
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
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_tensor_converter.pyi",
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
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_strict_test(
    name = "python_tensor_converter_test",
    srcs = ["python_tensor_converter_test.py"],
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_tensor_converter",
        ":constant_op",
        ":dtypes",
        ":indexed_slices",
        ":tensor",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "ops",  # "ops" is already the name of a deprecated target
    srcs = ["ops.py"],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/mlperf:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//tensorflow:internal",
    # "//tensorflow_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":c_api_util",
        ":composite_tensor",
        ":device",
        ":dtypes",
        ":errors",
        ":op_callbacks",
        ":registry",
        ":stack",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":traceable_stack",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:tf2",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:decorator_utils",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:function_utils",
        "//tensorflow/python/util:lock_util",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_stack",
        "//tensorflow/python/util:traceback_utils",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "op_callbacks",
    srcs = ["op_callbacks.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//third_party/cloud_tpu/convergence_tools:__subpackages__"],
    deps = [
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
    ],
)

cuda_py_strict_test(
    name = "op_callbacks_test",
    srcs = ["op_callbacks_test.py"],
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":op_callbacks",
        ":ops",
        ":sparse_tensor",
        ":test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "indexed_slices",
    srcs = ["indexed_slices.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":composite_tensor",
        ":composite_tensor_gradient",
        ":dtypes",
        ":ops",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "indexed_slices_test",
    srcs = ["indexed_slices_test.py"],
    deps = [
        ":composite_tensor_gradient",
        ":constant_op",
        ":indexed_slices",
        "//tensorflow/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tensor_conversion",
    srcs = ["tensor_conversion.py"],
    srcs_version = "PY3",
    deps = [
        ":tensor_conversion_registry",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
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

py_strict_library(
    name = "func_graph",
    srcs = ["func_graph.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps",
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":indexed_slices",
        ":ops",
        ":tensor",
        ":type_spec",
        "//tensorflow/core/function/capture:capture_container",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager/polymorphic_function:composite_tensor_utils",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/saved_model:save_context",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "auto_control_deps",
    srcs = ["auto_control_deps.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_control_deps_utils",
        ":dtypes",
        ":indexed_slices",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":sparse_tensor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_strict_library(
    name = "auto_control_deps_utils",
    srcs = ["auto_control_deps_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":dtypes",
        "//tensorflow/python/util:object_identity",
    ],
)

tf_py_strict_test(
    name = "auto_control_deps_test",
    size = "small",
    srcs = ["auto_control_deps_test.py"],
    python_version = "PY3",
    deps = [
        ":auto_control_deps",
        ":constant_op",
        ":dtypes",
        ":ops",
        ":test_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_switch_case",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:resource_variable_ops_gen",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:sendrecv_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/training:adam",
        "//tensorflow/python/training:momentum",
    ],
)

py_strict_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":errors",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:_pywrap_determinism",
        "//tensorflow/python/util:_pywrap_tensor_float_32_execution",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
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
        ":ops",
        ":test_lib",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

cuda_py_strict_test(
    name = "config_vgpu_test",
    size = "small",
    srcs = ["config_vgpu_test.py"],
    python_version = "PY3",
    shard_count = 5,
    tags = [
        "gpu",
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

tpu_py_strict_test(
    name = "config_test_tpu",
    size = "small",
    srcs = ["config_test.py"],
    # copybara:uncomment_begin
    # args = ["--tpu_use_tfrt=false"],
    # copybara:uncomment_end
    disable_tfrt = True,  # TODO(b/291666911): Update this test to call initialize_tpu_system
    main = "config_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":ops",
        ":test_lib",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

py_strict_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":config",
        ":ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "registry",
    srcs = ["registry.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
    ],
)

py_strict_library(
    name = "smart_cond",
    srcs = ["smart_cond.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow_model_optimization:__subpackages__"],
    deps = [
        ":tensor",
        ":tensor_util",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_case",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "smart_cond_test",
    size = "small",
    srcs = ["smart_cond_test.py"],
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":ops",
        ":smart_cond",
        ":test_lib",
        "//tensorflow/python/client:session",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/mlperf:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":ops",
        ":override_binary_operator",
        ":tensor",
        ":tensor_shape",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        ":type_spec_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tf2",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:sparse_ops_gen",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "weak_tensor",
    srcs = ["weak_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_tensor_gradient",
        ":dtypes",
        ":errors",
        ":extension_type",
        ":ops",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_spec",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/types:core",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "weak_tensor_test",
    srcs = ["weak_tensor_test.py"],
    main = "weak_tensor_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":errors",
        ":extension_type",
        ":ops",
        ":tensor",
        ":test_lib",
        ":weak_tensor",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/types:core",
    ],
)

py_strict_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "none_tensor",
    srcs = ["none_tensor.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//learning/deepmind:__subpackages__",
        "//third_party/py/tf_keras:__subpackages__",
    ],
    deps = [
        ":composite_tensor",
        ":type_spec",
        ":type_spec_registry",
    ],
)

py_strict_test(
    name = "composite_tensor_test",
    srcs = ["composite_tensor_test.py"],
    main = "composite_tensor_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":indexed_slices",
        ":sparse_tensor",
        ":test_lib",
        ":type_spec",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "composite_tensor_gradient",
    srcs = ["composite_tensor_gradient.py"],
    deps = [
        ":composite_tensor",
        "//tensorflow/python/util:nest",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_strict_library(
    name = "extension_type",
    srcs = ["extension_type.py"],
    srcs_version = "PY3",
    visibility = visibility,
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":extension_type_field",
        ":immutable_dict",
        ":tensor",
        ":tensor_shape",
        ":type_spec",
        ":type_spec_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:composite_tensor_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_strict_library(
    name = "extension_type_field",
    srcs = ["extension_type_field.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":immutable_dict",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":type_spec",
        "//tensorflow/python/util:type_annotations",
    ],
)

py_strict_library(
    name = "immutable_dict",
    srcs = ["immutable_dict.py"],
    srcs_version = "PY3",
    deps = [],
)

py_strict_library(
    name = "tensor_shape",
    srcs = ["tensor_shape.py"],
    srcs_version = "PY3",
    visibility = [
        "//smartass/brain/configure/python:__pkg__",
        "//tensorflow:internal",
        "//third_party/py/reverb:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:trace",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

pytype_strict_library(
    name = "type_spec",
    srcs = ["type_spec.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//third_party/py/tensorflow_gnn:__subpackages__"],
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
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "type_spec_registry",
    srcs = ["type_spec_registry.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//third_party/py/tensorflow_gnn:__subpackages__"],
    deps = [
        "//tensorflow/python/types:internal",
    ],
)

pytype_strict_library(
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/structured:structured_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
    ],
)

pytype_strict_library(
    name = "flexible_dtypes",
    srcs = ["flexible_dtypes.py"],
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":indexed_slices",
        ":ops",
        ":tensor_shape",
        ":weak_tensor",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "meta_graph",
    srcs = ["meta_graph.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":byte_swap_tensor",
        ":error_interpolation",
        ":graph_io",
        ":importer",
        ":op_def_registry",
        ":ops",
        ":tensor",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "@pypi_packaging//:pkg",
    ],
)

pytype_strict_library(
    name = "framework_lib",
    srcs = ["framework_lib.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":device",
        ":dtypes",
        ":importer",
        ":indexed_slices",
        ":load_library",
        ":ops",
        ":random_seed",
        ":sparse_tensor",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
    ],
)

pytype_strict_library(
    name = "graph_io",
    srcs = ["graph_io.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":byte_swap_tensor",
        ":ops",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "importer",
    srcs = ["importer.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":c_api_util",
        ":device",
        ":errors",
        ":function",
        ":op_def_registry",
        ":ops",
        ":tensor",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ] + if_xla_available([
        "//tensorflow/python:_pywrap_tfcompile",
    ]),
)

pytype_strict_library(
    name = "load_library",
    srcs = ["load_library.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":_pywrap_python_op_gen",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ] + if_xla_available([
        "//tensorflow/python:_pywrap_tfcompile",
    ]),
)

pytype_strict_library(
    name = "stack",
    srcs = ["stack.py"],
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/util:tf_contextlib",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tensor",
    srcs = ["tensor.py"],
    visibility = visibility + [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/mlperf:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":common_shapes",
        ":dtypes",
        ":errors",
        ":op_callbacks",
        ":stack",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":type_spec",
        ":type_spec_registry",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/eager:record",
        "//tensorflow/python/ops:handle_data_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:object_identity",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tensor_spec",
    srcs = ["tensor_spec.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/py/reverb:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//tensorflow:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tensor",
    ],
)

py_strict_library(
    name = "tensor_util",
    srcs = ["tensor_util.py"],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/py/tf_slim:__subpackages__",
    # "//tensorflow:internal",
    # "//tensorflow_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":dtypes",
        ":errors",
        ":tensor_shape",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:numpy_compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "traceable_stack",
    srcs = ["traceable_stack.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_strict_library(
    name = "versions",
    srcs = ["versions.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "gpu_util",
    srcs = ["gpu_util.py"],
    srcs_version = "PY3",
    deps = [],
)

pytype_strict_library(
    name = "test_lib",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/py/neural_structured_learning:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//third_party/py/tf_keras:__subpackages__",
    # "//tensorflow:internal",
    # "//tensorflow_model_optimization:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_test_metrics_util",
        ":config",
        ":device",
        ":dtypes",
        ":errors",
        ":gpu_util",
        ":importer",
        ":indexed_slices",
        ":ops",
        ":random_seed",
        ":sparse_tensor",
        ":tensor",
        ":tensor_shape",
        ":tensor_util",
        ":tfrt_utils",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python:pywrap_sanitizers",
        "//tensorflow/python:tf2",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/client:session",
        "//tensorflow/python/compat",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_util",
        "//tensorflow/python/ops:control_flow_util_v2",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:sparse_ops_gen",
        "//tensorflow/python/ops:summary_ops_v2",
        "//tensorflow/python/ops:sync_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//tensorflow/python/platform:_pywrap_stacktrace_handler",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:server_lib",
        "//tensorflow/python/util:_pywrap_util_port",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:numpy_compat",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/python/util:traceback_utils",
        "//tensorflow/python/util/protobuf",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@pypi_portpicker//:pkg",
    ],
)

# Including this as a dependency will result in tests using
# :test_lib to use XLA.
py_strict_library(
    name = "is_xla_test_true",
    srcs = ["is_xla_test_true.py"],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/py/keras_cv:__subpackages__",
        "//third_party/py/tf_keras:__subpackages__",
    ],
)

# Including this as a dependency will result in tests using
# :test_lib to use MLIR.
py_strict_library(
    name = "is_mlir_bridge_test_true",
    srcs = ["is_mlir_bridge_test_true.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
)

# Including this as a dependency will result in tests using
# :test_lib to NOT use MLIR.
py_strict_library(
    name = "is_mlir_bridge_test_false",
    srcs = ["is_mlir_bridge_test_false.py"],
    srcs_version = "PY3",
    visibility = visibility,
)

py_strict_library(
    name = "tfrt_utils",
    srcs = ["tfrt_utils.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "combinations",
    srcs = ["combinations.py"],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":ops",
        ":test_combinations_lib",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "test_combinations_lib",
    srcs = ["test_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "test_combinations_test",
    srcs = ["test_combinations_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":test_combinations_lib",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "memory_checker",
    srcs = [
        "memory_checker.py",
        "python_memory_checker.py",
    ],
    srcs_version = "PY3",
    visibility = visibility + ["//tensorflow:internal"],
    deps = [
        ":_python_memory_checker_helper",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/util:tf_inspect",
    ],
)

tf_python_pybind_extension(
    name = "_python_memory_checker_helper",
    srcs = ["python_memory_checker_helper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_python_memory_checker_helper.pyi",
    ],
    deps = [
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    main = "constant_op_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":importer",
        ":ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

tf_py_strict_test(
    name = "registry_test",
    size = "small",
    srcs = ["registry_test.py"],
    main = "registry_test.py",
    python_version = "PY3",
    deps = [
        ":registry",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.py"],
    main = "errors_test.py",
    python_version = "PY3",
    tags = ["no_windows"],  # TODO(b/184424727): Enable this test on Windows.
    deps = [
        ":_errors_test_helper",
        ":c_api_util",
        ":errors",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/lib/io:_pywrap_file_io",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "error_interpolation_test",
    size = "small",
    srcs = ["error_interpolation_test.py"],
    main = "error_interpolation_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":error_interpolation",
        ":errors",
        ":ops",
        ":test_lib",
        ":traceable_stack",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "subscribe_test",
    size = "small",
    srcs = ["subscribe_test.py"],
    main = "subscribe_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":ops",
        ":sparse_tensor",
        ":subscribe",
        ":test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "proto_test",
    size = "small",
    srcs = ["proto_test.py"],
    main = "proto_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
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
        ":constant_op",
        ":dtypes",
        ":errors",
        ":function",
        ":graph_to_function_def",
        ":ops",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_ops_gen",
        "//tensorflow/python/ops:functional_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:logging_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:template",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "versions_test",
    size = "small",
    srcs = ["versions_test.py"],
    main = "versions_test.py",
    python_version = "PY3",
    deps = [
        ":versions",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "importer_test",
    size = "small",
    srcs = ["importer_test.py"],
    main = "importer_test.py",
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        ":constant_op",
        ":device",
        ":dtypes",
        ":function",
        ":importer",
        ":ops",
        ":tensor_shape",
        ":test_lib",
        ":test_ops",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_grad",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
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

tf_py_strict_test(
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
        ":constant_op",
        ":dtypes",
        ":error_interpolation",
        ":function",
        ":meta_graph",
        ":ops",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:metrics",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/training:queue_runner_impl",
    ],
)

cuda_py_strict_test(
    name = "node_file_writer_test",
    size = "small",
    srcs = ["node_file_writer_test.py"],
    python_version = "PY3",
    # On Windows, fails in tempfile.TemporaryDirectory.cleanup() with error: "The directory is not
    # empty", so test is disabled on Windows.
    # TODO(reedwm): Debug this.
    tags = ["no_windows"],
    deps = [
        ":config",
        ":constant_op",
        ":dtypes",
        ":tensor_util",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "traceable_stack_test",
    size = "small",
    srcs = ["traceable_stack_test.py"],
    main = "traceable_stack_test.py",
    python_version = "PY3",
    deps = [
        ":test_lib",
        ":traceable_stack",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:tf_inspect",
    ],
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "test_ops.py",
    api_def_srcs = [
        "//tensorflow/core/api_def:base_api_def",
        "//tensorflow/core/api_def:python_api_def",
    ],
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
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

tf_py_strict_test(
    name = "common_shapes_test",
    size = "small",
    srcs = ["common_shapes_test.py"],
    main = "common_shapes_test.py",
    python_version = "PY3",
    deps = [
        ":common_shapes",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.py"],
    main = "ops_test.py",
    python_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":composite_tensor",
        ":config",
        ":constant_op",
        ":device",
        ":dtypes",
        ":errors",
        ":function",
        ":indexed_slices",
        ":ops",
        ":sparse_tensor",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":test_lib",
        ":test_ops",
        ":type_spec",
        ":versions",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/autograph/core:ag_ctx",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops_gen",
        "//tensorflow/python/ops:gradients",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:resources",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

# Test that op_allowlist works with ops with namespaces
tf_gen_op_wrapper_py(
    name = "test_namespace_ops",
    out = "test_namespace_ops.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
    op_allowlist = ["Namespace>TestStringOutput"],
    py_lib_rule = py_strict_library,
    deps = [
        ":test_ops_kernels",
    ],
)

tf_py_strict_test(
    name = "op_allowlist_namespace_test",
    size = "small",
    srcs = ["op_allowlist_namespace_test.py"],
    main = "op_allowlist_namespace_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # Avoid a missing test dependency error
    deps = [
        ":test_namespace_ops",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "python_op_gen_annotation_test",
    size = "small",
    srcs = ["python_op_gen_annotation_test.py"],
    main = "python_op_gen_annotation_test.py",
    python_version = "PY3",
    deps = [
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops_gen",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "ops_enable_eager_test",
    size = "small",
    srcs = ["ops_enable_eager_test.py"],
    main = "ops_enable_eager_test.py",
    python_version = "PY3",
    deps = [
        ":ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "extension_type_test",
    size = "small",
    srcs = ["extension_type_test.py"],
    main = "extension_type_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":extension_type",
        ":extension_type_field",
        ":immutable_dict",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        ":type_spec",
        ":type_spec_registry",
        ":type_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_inspect",
        "@absl_py//absl/testing:parameterized",
        "@pypi_typing_extensions//:pkg",
    ],
)

tf_py_strict_test(
    name = "extension_type_field_test",
    size = "small",
    srcs = ["extension_type_field_test.py"],
    main = "extension_type_field_test.py",
    python_version = "PY3",
    tags = ["no_pip"],  # TODO(b/184565242) Enable pip once we add a module that depends on this.
    deps = [
        ":constant_op",
        ":dtypes",
        ":extension_type_field",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
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

tf_py_strict_test(
    name = "tensor_shape_test",
    size = "small",
    srcs = ["tensor_shape_test.py"],
    main = "tensor_shape_test.py",
    python_version = "PY3",
    deps = [
        ":dtypes",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "type_spec_test",
    size = "small",
    srcs = ["type_spec_test.py"],
    main = "type_spec_test.py",
    python_version = "PY3",
    deps = [
        ":dtypes",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":test_lib",
        ":type_spec",
        ":type_spec_registry",
        ":type_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "tensor_test",
    size = "small",
    srcs = ["tensor_test.py"],
    main = "tensor_test.py",
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        ":type_spec",
        ":type_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sparse_tensor_test",
    size = "small",
    srcs = ["sparse_tensor_test.py"],
    main = "sparse_tensor_test.py",
    python_version = "PY3",
    deps = [
        ":dtypes",
        ":errors",
        ":ops",
        ":sparse_tensor",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        ":type_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "device_spec_test",
    size = "small",
    srcs = ["device_spec_test.py"],
    main = "device_spec_test.py",
    python_version = "PY3",
    deps = [
        ":device_spec",
        ":test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "device_test",
    size = "small",
    srcs = ["device_test.py"],
    main = "device_test.py",
    python_version = "PY3",
    deps = [
        ":device",
        ":device_spec",
        ":ops",
        ":test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "random_seed_test",
    size = "small",
    srcs = ["random_seed_test.py"],
    main = "random_seed_test.py",
    python_version = "PY3",
    deps = [
        ":random_seed",
        ":test_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tensor_util_test",
    size = "small",
    srcs = ["tensor_util_test.py"],
    main = "tensor_util_test.py",
    python_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":constant_op",
        ":dtypes",
        ":func_graph",
        ":indexed_slices",
        ":ops",
        ":sparse_tensor",
        ":tensor_shape",
        ":tensor_util",
        ":test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:shape_util",
        "//tensorflow/python/ops:state_ops_gen",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
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
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":indexed_slices",
        ":ops",
        ":random_seed",
        ":tensor",
        ":test_lib",
        ":test_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_sanitizers",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util/protobuf:compare_test_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
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
        "not_run:arm",
        "notsan",  # TODO(b/149948895): Re-enable.
        "nozapfhahn",
    ],
    deps = [
        ":constant_op",
        ":ops",
        ":test_lib",
        # TODO(kkb): Find more appropriate place to add `memory_checker` as deps
        # Adding directly to `test_lib` caused a Colab binary size
        # regression b/149433910 .
        ":memory_checker",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "dtypes_test",
    size = "small",
    srcs = ["dtypes_test.py"],
    main = "dtypes_test.py",
    python_version = "PY3",
    deps = [
        ":_dtypes",
        ":dtypes",
        ":test_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["op_def_library_test.py"],
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":function",
        ":op_def_library",
        ":op_def_library_pybind",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "kernels_test",
    size = "small",
    srcs = ["kernels_test.py"],
    main = "kernels_test.py",
    python_version = "PY3",
    deps = [
        ":kernels",
        ":test_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_proto_library(
    name = "cpp_shape_inference_proto",
    srcs = ["cpp_shape_inference.proto"],
    protodeps = tf_additional_all_protos(),
    # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    visibility = ["//tensorflow:internal"],
)

tf_python_pybind_extension(
    name = "_errors_test_helper",
    srcs = ["errors_test_helper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_errors_test_helper.pyi",
    ],
    deps = [
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core/platform:status",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "c_api_util_test",
    size = "small",
    srcs = ["c_api_util_test.py"],
    python_version = "PY3",
    deps = [
        ":c_api_util",
        ":test_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["graph_util_test.py"],
    python_version = "PY3",
    deps = [
        ":constant_op",
        ":dtypes",
        ":function",
        ":graph_util",
        ":ops",
        ":tensor_util",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:state_ops_gen",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "convert_to_constants_test",
    size = "medium",
    srcs = ["convert_to_constants_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_rocm"],
    deps = [
        ":constant_op",
        ":convert_to_constants",
        ":dtypes",
        ":function",
        ":importer",
        ":ops",
        ":tensor_spec",
        ":test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/grappler:tf_optimizer",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:cond_v2",
        "//tensorflow/python/ops:control_flow_case",
        "//tensorflow/python/ops:control_flow_switch_case",
        "//tensorflow/python/ops:control_flow_v2_toggles",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:ref_variable",
        "//tensorflow/python/ops:rnn",
        "//tensorflow/python/ops:rnn_cell_impl",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops:while_v2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/saved_model:simple_save",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

cuda_py_benchmark_test(
    name = "graph_building_benchmark",
    size = "medium",
    srcs = ["graph_building_benchmark.py"],
    main = "graph_building_benchmark.py",
    python_version = "PY3",
    deps = [
        ":dtypes",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_python_pybind_extension(
    name = "_proto_comparators",
    srcs = ["proto_comparators.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_proto_comparators.pyi",
    ],
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

py_strict_test(
    name = "tf2_test",
    srcs = ["tf2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":combinations",
        "//tensorflow/python:tf2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/platform:_pywrap_tf2",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "flexible_dtypes_test",
    srcs = ["flexible_dtypes_test.py"],
    tags = [
        "no_windows",  # TODO(b/286939592): Enable this test on Windows.
    ],
    deps = [
        ":constant_op",
        ":dtypes",
        ":extension_type",
        ":flexible_dtypes",
        ":ops",
        ":tensor",
        ":weak_tensor",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:weak_tensor_test_util",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "_test_metrics_util",
    srcs = ["test_metrics_util.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_test_metrics_util.pyi",
    ],
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
#     deps = [":cpp_shape_inference_proto"],
# )
# copybara:uncomment_end

py_strict_library(
    name = "summary_test_util",
    testonly = True,
    srcs = ["summary_test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/lib/io:tf_record",
        "//tensorflow/python/platform:gfile",
    ],
)

cc_library(
    name = "tensor_shape_casters",
    hdrs = ["tensor_shape_casters.h"],
    features = ["-parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "@pybind11",
    ],
)
