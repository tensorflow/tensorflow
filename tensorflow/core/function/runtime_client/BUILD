load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/function:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "runtime_client_cc",
    srcs = [
        "runtime_client.cc",
    ],
    hdrs = [
        "runtime_client.h",
    ],
    defines = select({
        "//tensorflow/compiler/mlir/python:disable_mlir_config": ["DISABLE_MLIR"],
        "//conditions:default": [],
    }),
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:function_def_utils",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/importexport:graphdef_export",
        "//tensorflow/core/ir/importexport:graphdef_import",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
    ] + select({
        "//tensorflow/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//tensorflow/compiler/mlir/python:mlir",
        ],
    }),
    # TODO(mdan): Get rid of alwayslink, it's nonstandard.
    alwayslink = 1,
)

# TODO(mdan): Pull these transitive header deps in a more decent fashion.
# TODO(mdan): Get rid of headers-only lib, it's nonstandard. Use cc_shared_library?
cc_library(
    name = "runtime_client_headers",
    textual_hdrs = [
        "runtime_client.h",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/config:flags_headers",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
    ],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:conversion_macros",
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c:tf_attrtype",
        "//tensorflow/c:tf_datatype_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:core_stringpiece",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/common_runtime:composite_device",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/config:flag_defs",
        "//tensorflow/core/config:flags",
        "//tensorflow/core/example:example_protos_cc",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:cost_graph_proto_cc",
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/framework:full_type_proto_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/framework:shape_inference",
        "//tensorflow/core/framework:step_stats_proto_cc",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:notification",
        "//tensorflow/core/lib/core:refcount",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:threadpool",
        "//tensorflow/core/platform:casts",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform:intrusive_ptr",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/protobuf:eager_service_proto_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/protobuf:master_proto_cc",
        "//tensorflow/core/protobuf:worker_proto_cc",
        "//tensorflow/core/public:version",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:refcount",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
        "@local_xla//xla/tsl/protobuf:coordination_config_proto_cc",
        "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
    ],
)

tf_cc_test(
    name = "runtime_client_cc_test",
    srcs = ["runtime_client_test.cc"],
    deps = [
        ":runtime_client_cc",
        "//tensorflow/c:c_api_experimental",  # buildcleaner: keep (registers CPU ops?)
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/function/testing:test_pass_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//third_party/protobuf",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

# TODO(b/221223841): Get rid of if_static, it's nonstandard.
tf_python_pybind_extension(
    name = "runtime_client_pybind",
    srcs = ["runtime_client_pybind.cc"],
    data = [
        "runtime_client_pybind.pyi",
    ],
    enable_stub_generation = True,
    deps = [
        ":runtime_client_headers",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@pybind11",
    ] + if_static(
        extra_deps = [
            "//tensorflow/core/framework:function_proto_cc",
            "//tensorflow/core/protobuf:eager_service_proto_cc",
            "//tensorflow/core/protobuf:master_proto_cc",
            "//tensorflow/core/protobuf:worker_proto_cc",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
        ],
        otherwise = [
            "//tensorflow/core/framework:function_proto_cc_headers_only",
            "//tensorflow/core/protobuf:eager_service_proto_cc_headers_only",
            "//tensorflow/core/protobuf:master_proto_cc_headers_only",
            "//tensorflow/core/protobuf:worker_proto_cc_headers_only",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
        ],
    ),
)

# TODO(mdan): Drop function_proto_py_pb2 once pybind11_protobuf is available.
pytype_strict_library(
    name = "runtime_client_py",
    srcs = [
        "runtime_client.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//learning/brain/experimental/tfq:__subpackages__",
        "//tensorflow/core/function/transform:__subpackages__",
        "//tensorflow/python/eager:__subpackages__",
    ],
    deps = [
        ":runtime_client_pybind",
        "//tensorflow/core/framework:function_proto_py",
        "//tensorflow/python:pywrap_tensorflow",  # buildcleaner: keep (required for TF pybind)
    ],
)

py_strict_test(
    name = "runtime_client_py_test",
    srcs = ["runtime_client_test.py"],
    main = "runtime_client_test.py",
    python_version = "PY3",
    tags = ["no_oss"],  # TODO(b/219089812)
    deps = [
        ":runtime_client_py",
        #internal proto upb dep
        "//tensorflow/core/framework:function_proto_py",
        "//tensorflow/core/function/testing:test_pass_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)
