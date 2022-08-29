load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config.bzl", "pyx_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_py_test_deps")
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_cc_test",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pytype_library")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow:internal",
        "//tensorflow/compiler/xla:friends",
    ],
)

pytype_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_extension",
        "@absl_py//absl/logging",
    ],
)

exports_files(["xla_client.pyi"])

pyx_library(
    name = "custom_call_for_test",
    testonly = True,
    srcs = ["custom_call_for_test.pyx"],
)

py_test(
    name = "xla_client_backend_independent_test",
    srcs = ["xla_client_backend_independent_test.py"],
    python_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

py_library(
    name = "xla_client_test",
    testonly = 1,
    srcs = ["xla_client_test.py"],
    srcs_version = "PY3",
    visibility = [":friends"],
    deps = [
        ":xla_client",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":custom_call_for_test",
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_test(
    name = "xla_client_test_gpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=gpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "config-cuda-only",
        "no_oss",
        "requires-gpu-nvidia",
    ],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:framework_test_lib",
    ] + xla_py_test_deps(),
)

cc_library(
    name = "status_casters",
    hdrs = ["status_casters.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":exceptions",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "@pybind11",
    ],
)

cc_library(
    name = "exceptions",
    hdrs = ["exceptions.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow/compiler/xla:status",
    ],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        ":exceptions",
        ":status_casters",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:lib",
        "//tensorflow/python:bfloat16_lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

cc_library(
    name = "python_ref_manager",
    srcs = ["python_ref_manager.cc"],
    hdrs = ["python_ref_manager.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cc_library(
    name = "python_utils",
    hdrs = ["python_utils.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

cc_library(
    name = "traceback",
    srcs = ["traceback.cc"],
    hdrs = ["traceback.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        ":exceptions",
        ":python_ref_manager",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
    ],
)

cc_library(
    name = "pprof_profile_builder",
    srcs = ["pprof_profile_builder.cc"],
    hdrs = ["pprof_profile_builder.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/profiler:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@pybind11",
    ],
)

cc_library(
    name = "py_client",
    srcs = [
        "callback.cc",
        "py_buffer.cc",
        "py_client.cc",
        "py_executable.cc",
        "py_values.cc",
        "sharded_device_array.cc",
    ] + if_cuda_or_rocm([
        "py_client_gpu.cc",
    ]),
    hdrs = [
        "callback.h",
        "py_buffer.h",
        "py_client.h",
        "py_executable.h",
        "py_values.h",
        "sharded_device_array.h",
    ] + if_cuda_or_rocm([
        "py_client_gpu.h",
    ]),
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = if_cuda(["GOOGLE_CUDA=1"]),
    features = ["-use_header_modules"],
    deps = [
        ":exceptions",
        ":pprof_profile_builder",
        ":python_ref_manager",
        ":python_utils",
        ":traceback",
        ":transfer_guard_lib",
        ":types",
        ":util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@pybind11",
        "//tensorflow/compiler/xla/pjrt:host_callback",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_stream_executor_client",
        "//tensorflow/compiler/xla/pjrt:transpose",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/core/platform:fingerprint",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/python/lib/core:numpy_lib",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cc_library(
    name = "dlpack",
    srcs = ["dlpack.cc"],
    hdrs = ["dlpack.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":py_client",
        ":python_ref_manager",
        ":traceback",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@pybind11",
    ],
)

cc_library(
    name = "jax_jit",
    srcs = ["jax_jit.cc"],
    hdrs = ["jax_jit.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":exceptions",
        ":py_client",
        ":python_ref_manager",
        ":python_utils",
        ":pytree",
        ":types",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/pjrt:lru_cache",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "//third_party/python_runtime:headers",  # build_cleaner: keep
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cc_library(
    name = "ops",
    srcs = ["ops.cc"],
    hdrs = ["ops.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:approx_topk",
        "//tensorflow/compiler/xla/client/lib:approx_topk_shape",
        "//tensorflow/compiler/xla/client/lib:comparators",
        "//tensorflow/compiler/xla/client/lib:lu_decomposition",
        "//tensorflow/compiler/xla/client/lib:math",
        "//tensorflow/compiler/xla/client/lib:qr",
        "//tensorflow/compiler/xla/client/lib:self_adjoint_eig",
        "//tensorflow/compiler/xla/client/lib:sorting",
        "//tensorflow/compiler/xla/client/lib:svd",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

cc_library(
    name = "outfeed_receiver",
    srcs = ["outfeed_receiver.cc"],
    hdrs = ["outfeed_receiver.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "pmap_lib",
    srcs = ["pmap_lib.cc"],
    hdrs = ["pmap_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":exceptions",
        ":jax_jit",
        ":py_client",
        ":python_utils",
        ":types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

tf_cc_test(
    name = "outfeed_receiver_test_cpu",
    size = "small",
    srcs = ["outfeed_receiver_test.cc"],
    deps = [
        ":outfeed_receiver",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/pjrt:cpu_device",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_stream_executor_client",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "outfeed_receiver_py",
    srcs = ["outfeed_receiver_py.cc"],
    hdrs = ["outfeed_receiver_py.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":outfeed_receiver",
        ":py_client",
        ":types",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
    ],
)

cc_library(
    name = "pytree",
    srcs = ["pytree.cc"],
    hdrs = ["pytree.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        ":exceptions",
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":types",
        "//tensorflow/compiler/mlir/xla:hlo_to_mlir_hlo",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo/stablehlo:chlo_ops",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@pybind11",
    ],
)

cc_library(
    name = "profiler",
    srcs = ["profiler.cc"],
    hdrs = ["profiler.h"],
    # TODO(b/172353882): figure out why compatible_with is needed to avoid some internal errors.
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":types",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/core/profiler/backends/cpu:python_tracer",
        "//tensorflow/core/profiler/lib:profiler_backends",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/rpc:profiler_server_impl",
        "//tensorflow/core/profiler/rpc/client:capture_profile",
        "//tensorflow/core/profiler/rpc/client:profiler_client",
        "//tensorflow/python/profiler/internal:traceme_wrapper",
        "@pybind11",
    ],
)

cc_library(
    name = "transfer_guard_lib",
    srcs = ["transfer_guard_lib.cc"],
    hdrs = ["transfer_guard_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:util",
        "@com_google_absl//absl/base:core_headers",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@pybind11",
    ],
)

cc_library(
    name = "weakref_lru_cache",
    srcs = ["weakref_lru_cache.cc"],
    hdrs = ["weakref_lru_cache.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/compiler/xla/pjrt:lru_cache",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/synchronization",
        "@pybind11",
    ],
)

cc_library(
    name = "xla_compiler",
    srcs = ["xla_compiler.cc"],
    hdrs = ["xla_compiler.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":py_client",
        ":types",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:name_uniquer",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ],
)

# TODO(phawkins): the configuration settings here are overly confusing. The right fix is to split
# xla_extension.so so that each backend is a separate plugin, however that must wait for a clean
# ABI separation between devices.
config_setting(
    name = "link_gpu_plugin",
    define_values = {"xla_python_enable_gpu": "true"},
)

bool_flag(
    name = "enable_gpu",
    build_setting_default = True,
)

config_setting(
    name = "gpu_enabled",
    flag_values = {
        ":enable_gpu": "True",
    },
)

bool_flag(
    name = "enable_tpu",
    build_setting_default = True,
)

config_setting(
    name = "tpu_enabled",
    flag_values = {
        ":enable_tpu": "True",
    },
)

pybind_extension(
    name = "xla_extension",
    srcs = [
        "xla.cc",
    ],
    defines = select({
        ":gpu_enabled": ["XLA_PYTHON_ENABLE_GPU=1"],
        "//conditions:default": [],
    }) + select({
        ":tpu_enabled": ["XLA_PYTHON_ENABLE_TPU=1"],
        "//conditions:default": [],
    }),
    pytype_deps = [
        "//third_party/py/numpy",
    ],
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
               ":dlpack",
               ":jax_jit",
               ":mlir",
               ":ops",
               ":pmap_lib",
               ":weakref_lru_cache",
               ":pprof_profile_builder",
               ":profiler",
               ":transfer_guard_lib",
               ":py_client",
               ":pytree",
               ":python_ref_manager",
               ":traceback",
               ":outfeed_receiver_py",
               ":types",
               ":xla_compiler",
               "@com_google_absl//absl/strings",
               "@com_google_absl//absl/strings:str_format",
               "@com_google_absl//absl/types:span",
               "@pybind11",
               "//third_party/python_runtime:headers",  # buildcleaner: keep
               "//tensorflow/compiler/xla:literal",
               "//tensorflow/compiler/xla:shape_util",
               "//tensorflow/compiler/xla:status",
               "//tensorflow/compiler/xla:statusor",
               "//tensorflow/compiler/xla:types",
               "//tensorflow/compiler/xla:util",
               "//tensorflow/compiler/xla/pjrt:cpu_device",
               "//tensorflow/compiler/xla/pjrt:interpreter_device",
               "//tensorflow/compiler/xla/pjrt:pjrt_client",
               "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
               "//tensorflow/compiler/xla/pjrt/distributed",
               "//tensorflow/compiler/xla/pjrt/distributed:client",
               "//tensorflow/compiler/xla/pjrt/distributed:service",
               "//tensorflow/core:lib",
               # Do NOT remove this dependency. The XLA Python extension must not
               # depend on any part of TensorFlow at runtime, **including**
               # libtensorflow_framework.so. The XLA module is deployed self-contained
               # without any TF dependencies as "jaxlib" on Pypi, and "jaxlib" does
               # not require Tensorflow.
               "//tensorflow/core:lib_internal_impl",  # buildcleaner: keep
               "//tensorflow/core/distributed_runtime/preemption:preemption_sync_manager",
               "//tensorflow/python:bfloat16_lib",
           ] + select({
               ":gpu_enabled": [
                   "//tensorflow/compiler/xla/pjrt:gpu_device",
               ],
               "//conditions:default": [],
           }) + select({
               ":link_gpu_plugin": ["//tensorflow/compiler/xla/service:gpu_plugin"],
               "//conditions:default": [],
           }) +
           select({
               ":tpu_enabled": [
                   "//tensorflow/compiler/xla/pjrt:tpu_client",
                   "//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
               ],
               "//conditions:default": [],
           }),
)
