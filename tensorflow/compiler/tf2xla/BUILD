load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

# copybara:uncomment_begin(google-only)
# load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
# copybara:uncomment_end

load("@local_xla//xla:xla.default.bzl", "xla_py_proto_library")
load("@local_xla//xla/service/cpu:build_defs.bzl", "runtime_copts")
load("@local_xla//xla/tsl/mkl:build_defs.bzl", "mkl_deps")
load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_static")
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_google", "if_libtpu", "tf_cc_binary", "tf_cc_test", "tf_copts", "tf_cuda_cc_test", "tf_gen_op_wrapper_py", "tf_openmp_copts")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "filegroup", "get_compatible_with_portable")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_tensor_coding_deps",
    "tf_proto_library",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/aot/...",
        "//tensorflow/compiler/jit/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tests/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/core/tpu/...",
        "//tensorflow/python/compiler/...",
        "//tensorflow/python/util/...",
    ],
)

package_group(
    name = "friends",
    includes = [":internal"],
    packages = [
        "//platforms/performance/automl/...",
        "//tensorflow/...",
        "//tensorflow_federated/cc/core/impl/executors/...",
        "//tensorflow_models/...",
        "//third_party/deepmind/deepmind_research/density_functional_approximation_dm21/...",
        "//third_party/mlir_edge/model_curriculum/iree/...",
        "//third_party/mlperf/submissions/training/v0_7/models/...",
        "//third_party/py/ai_edge_torch/...",
        "//third_party/py/keras_cv/...",
        "//third_party/py/tf_keras/...",
        "//third_party/sparse_conv/ops/...",
        "//waymo/ml/deploy/benchmark/...",
    ],
)

cc_library(
    name = "tf2xla_supported_ops_lib",
    srcs = ["tf2xla_supported_ops.cc"],
    hdrs = ["tf2xla_supported_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_compiler",
        ":xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_binary(
    name = "tf2xla_supported_ops",
    srcs = ["tf2xla_supported_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [":tf2xla_supported_ops_lib"],
)

tf_proto_library(
    name = "tf2xla_proto",
    srcs = ["tf2xla.proto"],
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

xla_py_proto_library(
    name = "tf2xla_py",
    has_services = False,
    visibility = ["//visibility:public"],
    deps = [":tf2xla_proto"],
)

tf_proto_library(
    name = "host_compute_metadata_proto",
    srcs = ["host_compute_metadata.proto"],
    protodeps = [
        "//tensorflow/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_compiler_util",
    srcs = ["graph_compiler_util.cc"],
    hdrs = ["graph_compiler_util.h"],
    visibility = [":friends"],
    deps = [
        ":functionalize_control_flow",
        ":sharding_util",
        ":tf2xla_proto_cc",
        ":tf2xla_util",
        ":xla_compiler",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "tf2xla",
    srcs = ["tf2xla.cc"],
    hdrs = ["tf2xla.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":graph_compiler_util",
        ":tf2xla_proto_cc",
        ":tf2xla_util",
        ":xla_compiler",
        ":xla_op_registry",
        "//tensorflow/compiler/aot:aot_only_var_handle_op",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/client",
        "@local_xla//xla/hlo/builder:xla_computation",
    ],
)

cc_library(
    name = "mlir_tf2xla",
    srcs = ["mlir_tf2xla.cc"],
    hdrs = ["tf2xla.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf2xla_proto_cc",
        ":tf2xla_util",
        ":xla_compiler",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/transforms:bridge",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow/translate/tools:parsers",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/client",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

# The filegroups below are explicitly used by
# tensorflow/tools/pip_package:build_pip_package to ensure we include the proper
# sources for the XLA AOT CPU runtime; as these are necessary outside of bazel
# when linking tfcompile objects using saved_model_cli (e.g. using the
# tensorflow pip package). The associated .cc files are included in tensorflow
# pip package's xla_aot_runtime_srcs/ subdirectory. All necessary headers are
# also included in the pip package's include/tensorflow/ and include/external/
# subdirectories. Note however that sometimes additional object files may need
# to be linked when linking aot xla objects, e.g. abseil libraries. See the deps
# attribute of the "xla_compiled_cpu_runtime_standalone" target below for an
# exhaustive list.
filegroup(
    name = "xla_compiled_cpu_runtime_hdrs",
    srcs = [
        "xla_compiled_cpu_function.h",
        "//tensorflow/core/kernels:xla_cpu_runtime_hdrs",
        "//tensorflow/core/platform:xla_cpu_runtime_srcs",
        "@local_tsl//tsl/platform:xla_cpu_runtime_srcs",
        "@local_xla//xla:cpu_runtime_hdrs",
        "@local_xla//xla/backends/cpu:xla_cpu_runtime_hdrs",
        "@local_xla//xla/service:custom_call_status_hdrs",
        "@local_xla//xla/service/cpu:runtime_hdrs",
        "@local_xla//xla/tsl/concurrency:xla_cpu_runtime_hdrs",
        "@local_xla//xla/tsl/framework:xla_cpu_runtime_hdrs",
        "@local_xla//xla/tsl/framework/fixedpoint:xla_cpu_runtime_hdrs",
        "@local_xla//xla/tsl/lib/math:xla_cpu_runtime_hdrs",
        "@local_xla//xla/tsl/util:xla_cpu_runtime_hdrs",
    ],
    visibility = [
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
)

filegroup(
    name = "xla_compiled_cpu_runtime_srcs",
    srcs = [
        "xla_compiled_cpu_function.cc",
        "//tensorflow/core/kernels:xla_cpu_runtime_srcs",
        "//tensorflow/core/platform:xla_cpu_runtime_srcs",
        "@local_tsl//tsl/platform:xla_cpu_runtime_srcs",
        "@local_xla//xla:cpu_runtime_srcs",
        "@local_xla//xla/service:custom_call_status_srcs",
        "@local_xla//xla/service/cpu:runtime_srcs",
        "@local_xla//xla/tsl/concurrency:xla_cpu_runtime_srcs",
    ],
    visibility = [
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
)

# This stand-alone target is used to ensure that we can build tf_library type
# targets against the subset of sources declared in
# xla_compiled_cpu_runtime_{srcs,hdrs}.
#
# The macros in tensorflow/python/tools/tools.bzl produce AOT compiled binaries
# that rely on this target, as do unit tests in tensorflow/python/tools.
#
# See above for the significance of the source filegroups.
cc_library(
    name = "xla_compiled_cpu_runtime_standalone",
    srcs = [
        ":xla_compiled_cpu_runtime_srcs",
    ],
    hdrs = [
        ":xla_compiled_cpu_runtime_hdrs",
    ],
    copts = runtime_copts() + tf_openmp_copts(),
    defines = [
        "EIGEN_NEON_GEBP_NR=4",
    ],
    features = [
        "fully_static_link",
        "-parse_headers",
    ],
    linkstatic = 1,
    visibility = [":friends"],
    # Note, we specifically removed MKL and multithreaded dependencies so the
    # standalone does not require the MKL binary blob or threading libraries.
    #
    # TODO(ebrevdo): Remove tf_additoinal_tensor_coding_deps in favor of
    # absl/strings:cord when we update absl to a newer version.
    #
    # NOTE!!!!! (b/150412548 for reference)
    # DO NOT add dependencies here!
    #
    # 1) Read first the purpose of this target and the purpose of the
    #   :xla_compiled_cpu_runtime_{srcs|hrds} targets.
    #
    # Key point is: this target represents the sources available under the xla_aot_runtime_srcs
    # dir in the tensorflow pip package, which enables linking AOT-ed saved modules in customer
    # projects, including those not using bazel. So adding a dependency here to fix the tests
    # using this target (xla_compiled_cpu_runtime_standalone) defeats the purpose.
    #
    # 2) Most likely you need to split your dependency into 2 file groups and add them
    #    to the :xla_compiled_cpu_runtime_{srcs|hrds} targets accordingly.
    #
    # !!!DO NOT EDIT dependencies - please read the above!!!
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@ducc//:fft_wrapper",
        "@eigen_archive//:eigen3",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/tsl:empty",
        "//tensorflow/core/framework:numeric_types",
        "//tensorflow/core/platform:bfloat16",
        "//tensorflow/core/platform:float8",
        "//tensorflow/core/platform:stringpiece",
        # Extra dependencies required for multithreaded runtime objects.
        "//tensorflow/core/platform:blocking_counter",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "@local_tsl//tsl/platform:blocking_counter",
        "@local_tsl//tsl/platform:context",
        "@local_tsl//tsl/platform:cord",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:tstring",
        "@local_tsl//tsl/platform:types",
    ] + tf_additional_tensor_coding_deps() + mkl_deps(),
    alwayslink = 1,
)

# copybara:uncomment_begin(google-only)
# check_dependencies(
#     # Please read the whole comment above the `deps` of xla_compiled_cpu_runtime_standalone before
#     # making any change here.
#     name = "only_pip_sources_in_xla_lib",
#     of = [":xla_compiled_cpu_runtime_standalone"],
#     allowed_match_regexp = "|".join([
#         ":xla_compiled_cpu_runtime_standalone",
#         "//tensorflow/core/framework:numeric_types",
#         "@local_xla//xla/tsl/framework:numeric_types",
#         "@local_xla//xla/tsl/framework:device_type",
#         "@local_xla//xla/tsl/framework:fixedpoint_types",
#         "@local_tsl//tsl/platform:bfloat16",
#         "@local_tsl//tsl/platform:blocking_counter",
#         "@local_xla//xla/tsl/platform:byte_order",
#         "@local_tsl//tsl/platform:context",
#         "@local_tsl//tsl/platform:cord",
#         "@local_tsl//tsl/platform:env_time",
#         "@local_tsl//tsl/platform:ml_dtypes",
#         "@local_tsl//tsl/platform:logging",
#         "@local_tsl//tsl/platform:macros",
#         "@local_tsl//tsl/platform:mutex",
#         "@local_tsl//tsl/platform:platform",
#         "@local_tsl//tsl/platform:stringpiece",
#         "@local_tsl//tsl/platform:thread_annotations",
#         "@local_tsl//tsl/platform:tstring",
#         "@local_tsl//tsl/platform:types",
#         "@local_xla//xla/tsl/platform:env_time",
#         "@local_xla//xla/tsl/platform:logging",
#         "@local_xla//xla/tsl/platform:types",
#         "@local_xla//xla/tsl/platform:macros",
#         "@local_xla//xla/tsl/platform/default:context",
#         "@local_xla//xla/tsl/platform/default:cord",
#         "@local_xla//xla/tsl/platform/default:env_time",
#         "@local_xla//xla/tsl/platform/default:logging",
#         "@local_xla//xla/tsl/platform/default:types",
#         "@local_xla//xla/tsl/platform/google:context",
#         "@local_xla//xla/tsl/platform/google:cord",
#         "@local_xla//xla/tsl/platform/google:env_time",
#         "@local_xla//xla/tsl/platform/google:logging",
#         "@local_xla//xla/tsl/platform/google:types",
#         "@local_xla//xla/tsl/platform/windows:env_time",
#         "//tensorflow/core/platform:bfloat16",
#         "//tensorflow/core/platform:blocking_counter",
#         "//tensorflow/core/platform:byte_order",
#         "//tensorflow/core/platform:cord",
#         "//tensorflow/core/platform:env_time",
#         "//tensorflow/core/platform:float8",
#         "//tensorflow/core/platform:logging",
#         "//tensorflow/core/platform:macros",
#         "//tensorflow/core/platform:mutex",
#         "//tensorflow/core/platform:platform",
#         "//tensorflow/core/platform:stringpiece",
#         "//tensorflow/core/platform:thread_annotations",
#         "//tensorflow/core/platform:tstring",
#         "//tensorflow/core/platform:types",
#         "@local_xla//xla/tsl:empty",
#     ]),
#     dont_match_regexp = "|".join(["//tensorflow"]),
#     kind = "cc_library",
# )
# copybara:uncomment_end

cc_library(
    name = "xla_compiled_cpu_function",
    srcs = ["xla_compiled_cpu_function.cc"],
    hdrs = ["xla_compiled_cpu_function.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@local_xla//xla/service:custom_call_status_internal",
        # Keep dependencies to a minimum here; this library is used in every AOT
        # binary produced by tfcompile.
        "@local_xla//xla:cpu_function_runtime",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla/service/cpu:buffer_desc",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "xla_compiled_cpu_function_thunks",
    srcs = ["xla_compiled_cpu_function_thunks.cc"],
    hdrs = ["xla_compiled_cpu_function_thunks.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":xla_compiled_cpu_function",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla/backends/cpu/codegen:aot_compiled_function_library",
        "@local_xla//xla/backends/cpu/nanort:nanort_executable",
        "@local_xla//xla/backends/cpu/runtime:function_library",
        "@local_xla//xla/service:executable",  # buildcleaner: keep (b/404179184)
        "@local_xla//xla/service/cpu:cpu_aot_compilation_result",
        "@local_xla//xla/service/cpu:executable_proto_cc",
        "@local_xla//xla/tsl/concurrency:async_value",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "xla_compiled_cpu_function_factory",
    srcs = ["xla_compiled_cpu_function_factory.cc"],
    hdrs = ["xla_compiled_cpu_function_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "xla_compiled_cpu_function_thunk_factory_registerer",
    srcs = ["xla_compiled_cpu_function_thunk_factory_registerer.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_compiled_cpu_function_factory",
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "//tensorflow/compiler/tf2xla:xla_compiled_cpu_function_thunks",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "cpu_function_runtime_test",
    srcs = ["cpu_function_runtime_test.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@local_xla//xla:cpu_function_runtime",
    ],
)

cc_library(
    name = "xla_jit_compiled_cpu_function",
    srcs = ["xla_jit_compiled_cpu_function.cc"],
    hdrs = ["xla_jit_compiled_cpu_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf2xla",
        ":tf2xla_proto_cc",
        ":xla_compiled_cpu_function",
        ":xla_compiled_cpu_function_thunks",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_xla//xla:cpu_function_runtime",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/backends/cpu/codegen:compiled_function_library",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/service/cpu:cpu_aot_compilation_result",
        "@local_xla//xla/service/cpu:executable_proto_cc",
        "@local_xla//xla/stream_executor:platform",
    ] + if_libtpu(
        if_false = [
            "@local_xla//xla/service:cpu_plugin",
            "@local_xla//xla/service/cpu:buffer_info_util",
            "@local_xla//xla/service/cpu:cpu_executable",
        ],
        if_true = [],
    ),
)

tf_cc_test(
    name = "graph_compiler_test",
    srcs = ["graph_compiler_test.cc"],
    deps = [
        ":graph_compiler_util",
        ":tf2xla_proto_cc",
        ":xla_compilation_device",
        ":xla_compiler",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/platform:refcount",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xla_compiler",
    srcs = [
        "const_analysis.cc",
        "graph_compiler.cc",
        "xla_compiler.cc",
        "xla_cpu_backend.cc",
        "xla_op_kernel.cc",
    ] + if_cuda_is_configured([
        "xla_gpu_backend.cc",
    ]) + if_rocm_is_configured([
        "xla_gpu_backend.cc",
    ]),
    hdrs = [
        "const_analysis.h",
        "graph_compiler.h",
        "xla_compiler.h",
        "xla_helpers.h",
        "xla_op_kernel.h",
        "xla_op_registry.h",
    ],
    copts = tf_copts(),
    visibility = [":friends"],
    deps = [
        ":common",
        ":host_compute_metadata_proto_cc",
        ":layout_util",
        ":rearrange_function_argument",
        ":sharding_util",
        ":side_effect_util",
        ":tf2xla_util",
        ":xla_argument",
        ":xla_compilation_device",
        ":xla_context",
        ":xla_expression",
        ":xla_helpers",
        ":xla_op_registry",
        ":xla_resource",
        "//tensorflow/compiler/jit:common",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/jit:xla_compile_util",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tf2xla:mlir_bridge_rollout_policy",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
        "//tensorflow/compiler/mlir/utils:array_container_utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/util:overflow",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla:literal",
        "@local_xla//xla:protobuf_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/core/collectives:clique_id",
        "@local_xla//xla/core/collectives:clique_key",
        "@local_xla//xla/hlo/builder:value_inference",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:layout_util",
        "@local_xla//xla/service:computation_placer_hdr",
        "@local_xla//xla/service:hlo_proto_cc",
    ] + if_libtpu([
        ":xla_tpu_backend_registration",
    ]) + if_static(["@local_tsl//tsl/platform:tensor_float_32_utils"]),
    alwayslink = 1,
)

cc_library(
    name = "xla_compilation_device",
    srcs = [
        "xla_compilation_device.cc",
    ],
    hdrs = [
        "xla_compilation_device.h",
    ],
    deps = [
        ":common",
        ":frontend_attributes_util",
        ":sharding_util",
        ":xla_context",
        ":xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "@local_xla//xla/hlo/builder:xla_builder",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_tpu_backend_registration",
    srcs = ["xla_tpu_backend.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_op_registry",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_node_device_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_context",
    srcs = [
        "xla_context.cc",
    ],
    hdrs = [
        "xla_context.h",
    ],
    deps = [
        ":common",
        ":xla_expression",
        ":xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_op_registry",
    srcs = [
        "xla_op_registry.cc",
    ],
    hdrs = [
        "xla_op_registry.h",
    ],
    visibility = [":friends"],
    deps = [
        ":common",
        ":xla_context",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime/next_pluggable_device:next_pluggable_device_factory_hdrs",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_xla//xla:util",
        "@local_xla//xla/client:client_library",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_expression",
    srcs = [
        "xla_expression.cc",
    ],
    hdrs = [
        "xla_expression.h",
    ],
    deps = [
        ":common",
        ":xla_resource",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/client",
        "@local_xla//xla/hlo/builder:value_inference",
        "@local_xla//xla/hlo/builder:xla_builder",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_resource",
    srcs = [
        "xla_resource.cc",
    ],
    hdrs = [
        "xla_resource.h",
    ],
    visibility = [
        ":internal",
        "//learning/deepmind/tensorflow/tpufunc:__pkg__",
    ],
    deps = [
        ":common",
        ":sharding_util",
        ":xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_helpers",
    srcs = [
        "xla_helpers.cc",
    ],
    hdrs = [
        "xla_helpers.h",
    ],
    visibility = [":friends"],
    deps = [
        ":common",
        ":host_compute_metadata_proto_cc",
        "//tensorflow/compiler/tf2xla/lib:util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla:types",
        "@local_xla//xla/backends/gpu/collectives:gpu_clique_key",
        "@local_xla//xla/core/collectives:clique_id",
        "@local_xla//xla/core/collectives:clique_key",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/hlo/builder/lib:arithmetic",
        "@local_xla//xla/hlo/builder/lib:constants",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:layout_util",
        "@local_xla//xla/service:computation_placer_hdr",
        "@local_xla//xla/service/gpu:gpu_executable_run_options",
        "@local_xla//xla/stream_executor:stream",
        "@local_xla//xla/stream_executor:stream_executor_h",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_argument",
    srcs = [
        "xla_argument.cc",
    ],
    hdrs = [
        "xla_argument.h",
    ],
    deps = [
        ":host_compute_metadata_proto_cc",
        ":xla_resource",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/ir:hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "layout_util",
    srcs = [
        "layout_util.cc",
    ],
    hdrs = [
        "layout_util.h",
    ],
    visibility = [":friends"],
    deps = [
        ":common",
        ":xla_argument",
        ":xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/hlo/ir:hlo",
    ],
    alwayslink = 1,
)

cc_library(
    name = "common",
    srcs = [
        "literal_util.cc",
        "shape_util.cc",
        "type_util.cc",
    ],
    hdrs = [
        "literal_util.h",
        "shape_util.h",
        "type_util.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

tf_cc_test(
    name = "type_util_test",
    srcs = ["type_util_test.cc"],
    deps = [
        ":common",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "frontend_attributes_util",
    srcs = ["frontend_attributes_util.cc"],
    hdrs = ["frontend_attributes_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf2xla_defs",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "sharding_util",
    srcs = ["sharding_util.cc"],
    hdrs = ["sharding_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:xla_sharding_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/hlo/builder:sharding_builder",
    ],
)

tf_cc_test(
    name = "sharding_util_test",
    srcs = ["sharding_util_test.cc"],
    deps = [
        ":sharding_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# Internal targets below this point.

cc_library(
    name = "tf2xla_defs",
    hdrs = ["tf2xla_defs.h"],
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tf2xla_util",
    srcs = ["tf2xla_util.cc"],
    hdrs = ["tf2xla_util.h"],
    visibility = [":friends"],
    deps = [
        ":sharding_util",
        ":tf2xla_defs",
        ":tf2xla_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

tf_cc_test(
    name = "tf2xla_util_test",
    srcs = ["tf2xla_util_test.cc"],
    deps = [
        ":sharding_util",
        ":tf2xla_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tf2xla_test",
    srcs = ["tf2xla_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":tf2xla",
        ":tf2xla_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:tensor_float_32_hdr_lib",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/builder:xla_computation",
        "@local_xla//xla/service:cpu_plugin",
    ] + if_static(["@local_tsl//tsl/platform:tensor_float_32_utils"]),
)

tf_cc_test(
    name = "xla_jit_compiled_cpu_function_test",
    srcs = ["xla_jit_compiled_cpu_function_test.cc"],
    deps = [
        ":tf2xla_proto_cc",
        ":xla_compiled_cpu_function_thunks",
        ":xla_jit_compiled_cpu_function",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:casts",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/testlib:test",
        "@local_xla//xla/service:compiler",
        "@local_xla//xla/service:platform_util",
        "@local_xla//xla/service/cpu:cpu_executable",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

tf_cc_test(
    name = "xla_compiler_test",
    srcs = [
        "xla_compiler_test.cc",
        "xla_expression_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":common",
        ":layout_util",
        ":side_effect_util",
        ":xla_argument",
        ":xla_compiler",
        ":xla_expression",
        ":xla_resource",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/builder:xla_builder",
        "@local_xla//xla/service:cpu_plugin",
        "@local_xla//xla/service:hlo_module_util",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/service:hlo_proto_util",
        "@local_xla//xla/tests:literal_test_util",
    ],
)

tf_cc_test(
    name = "literal_util_test",
    srcs = [
        "literal_util_test.cc",
    ],
    deps = [
        ":common",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "const_analysis_test",
    size = "small",
    srcs = ["const_analysis_test.cc"],
    deps = [
        ":xla_compiler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/jit:xla_cluster_util",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "light_outside_compilation_kernels_for_test",
    testonly = True,
    srcs = ["light_outside_compilation_kernels_for_test.cc"],
    linkstatic = 1,
    deps = [
        ":xla_compiler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/compiler/tf2xla/kernels:light_outside_compilation",
        "//tensorflow/core:framework",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:stream",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "test_ops_for_light_outside_compilation",
    testonly = True,
    out = "test_ops_for_light_outside_compilation.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [
        ":light_outside_compilation_kernels_for_test",
    ],
)

cuda_py_strict_test(
    name = "light_outside_compilation_test",
    srcs = ["light_outside_compilation_test.py"],
    tags = [
        "no_oss",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":test_ops_for_light_outside_compilation",
        "//tensorflow/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
    ] + if_google([":light_outside_compilation_kernels_for_test"]),
)

cc_library(
    name = "functionalize_control_flow_util",
    srcs = [
        "functionalize_control_flow_util.cc",
    ],
    hdrs = [
        "functionalize_control_flow_util.h",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "functionalize_cond",
    srcs = [
        "functionalize_cond.cc",
    ],
    hdrs = [
        "functionalize_cond.h",
    ],
    deps = [
        ":frontend_attributes_util",
        ":functionalize_control_flow_util",
        ":tf2xla_util",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:union_find",
    ],
)

cc_library(
    name = "functionalize_control_flow",
    srcs = [
        "functionalize_control_flow.cc",
    ],
    hdrs = [
        "functionalize_control_flow.h",
    ],
    visibility = [":friends"],
    deps = [
        ":functionalize_cond",
        ":functionalize_control_flow_util",
        ":functionalize_while",
        ":tf2xla_util",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:union_find",
    ],
)

cc_library(
    name = "mlir_bridge_pass",
    srcs = ["mlir_bridge_pass.cc"],
    hdrs = ["mlir_bridge_pass.h"],
    visibility = [":internal"],
    deps = [
        ":tf2xla_defs",
        ":xla_op_registry",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:attribute_utils",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms/host_runtime:lower_cluster_to_runtime_ops",
        "//tensorflow/compiler/mlir/tf2xla:mlir_bridge_rollout_policy",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:cluster_tf",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:tf_dialect_to_executor",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:cluster_tf",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_dialect_to_executor",
        "//tensorflow/compiler/mlir/tf2xla/internal:mlir_bridge_pass_util",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_bridge_pass_registration",
    srcs = [
        "mlir_bridge_pass_registration.cc",
    ],
    visibility = [
        # We define a new TPU device in TFRT to enable TFRT TPU Runtime.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
        ":internal",
    ],
    deps = [
        ":mlir_bridge_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rearrange_function_argument",
    srcs = [
        "rearrange_function_argument.cc",
    ],
    hdrs = [
        "rearrange_function_argument.h",
    ],
    deps = [
        ":tf2xla_util",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "functionalize_control_flow_pass_registration",
    srcs = [
        "functionalize_control_flow_pass_registration.cc",
    ],
    visibility = [":friends"],
    deps = [
        ":functionalize_control_flow",
    ],
    alwayslink = 1,
)

cc_library(
    name = "functionalize_while",
    srcs = [
        "functionalize_while.cc",
    ],
    hdrs = [
        "functionalize_while.h",
    ],
    deps = [
        ":frontend_attributes_util",
        ":functionalize_cond",
        ":functionalize_control_flow_util",
        ":tf2xla_util",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:union_find",
    ],
)

tf_cc_test(
    name = "functionalize_control_flow_test",
    srcs = ["functionalize_control_flow_test.cc"],
    deps = [
        ":functionalize_control_flow",
        ":test_util",
        ":tf2xla_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:resource_variable_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@local_xla//xla:status_macros",
    ],
)

tf_cc_test(
    name = "functionalize_cond_test",
    srcs = ["functionalize_cond_test.cc"],
    deps = [
        ":functionalize_cond",
        ":functionalize_control_flow",
        ":test_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:resource_variable_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/platform:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@local_xla//xla:status_macros",
    ],
)

tf_cc_test(
    name = "xla_op_registry_test",
    srcs = ["xla_op_registry_test.cc"],
    deps = [
        ":xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "resource_operation_table",
    srcs = ["resource_operation_table.cc"],
    hdrs = ["resource_operation_table.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "resource_operation_table_test",
    srcs = ["resource_operation_table_test.cc"],
    deps = [
        ":resource_operation_table",
        ":xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "side_effect_util",
    srcs = ["side_effect_util.cc"],
    hdrs = ["side_effect_util.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:core_cpu",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "fused_batchnorm_reserve_space_test",
    size = "medium",
    srcs = ["fused_batchnorm_reserve_space_test.cc"],
    tags = [
        "no_cuda_asan",  # TODO(b/193450885)
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "mlir_xla_op_kernel",
    srcs = ["mlir_xla_op_kernel.cc"],
    hdrs = ["mlir_xla_op_kernel.h"],
    deps = [
        ":xla_compiler",
        ":xla_expression",
        "//tensorflow/compiler/jit:xla_compile_util",
        "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
        "//tensorflow/compiler/mlir/utils:array_container_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:resource_base",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/hlo/builder:xla_builder",
    ],
)

cc_library(
    name = "resource_util",
    srcs = ["resource_util.cc"],
    hdrs = ["resource_util.h"],
    visibility = [":friends"],
    deps = [
        ":resource_operation_table",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
    ],
)

tf_cuda_cc_test(
    name = "resource_util_test",
    size = "small",
    srcs = ["resource_util_test.cc"],
    deps = [
        ":resource_util",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
)

filegroup(
    name = "tf2xla_opset_hdrs",
    srcs = [
        "tf2xla_opset.h",
    ],
    visibility = ["//tensorflow/python/util:__pkg__"],
)

cc_library(
    name = "tf2xla_opset",
    srcs = [
        "tf2xla_opset.cc",
    ],
    hdrs = ["tf2xla_opset.h"],
    visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/util:__pkg__",
    ],
    deps = [
        ":tf2xla_util",
        ":xla_op_registry",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tf2xla_opset_test",
    srcs = [
        "tf2xla_opset_test.cc",
    ],
    deps = [
        ":tf2xla_opset",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
