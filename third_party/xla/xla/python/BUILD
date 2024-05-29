load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_library", "py_strict_test")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
    "xla_py_test_deps",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_cuda_or_rocm",
    "if_google",
    "internal_visibility",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable", "tsl_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":friends",
    ]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
)

pytype_strict_library(
    name = "xla_client",
    srcs = ["xla_client.py"],
    pytype_srcs = ["xla_client.pyi"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_extension",  # buildcleaner: keep
        "@ml_dtypes",
    ],
)

exports_files([
    "xla_client.py",
    "xla_client.pyi",
])

tsl_pybind_extension(
    name = "custom_calls_testlib",
    srcs = ["custom_calls_testlib.cc"],
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/nanobind",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:ffi",
        "@com_google_absl//absl/status",
    ],
)

py_strict_test(
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

py_strict_library(
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

py_strict_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",
        "not_run:arm",
    ],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":custom_calls_testlib",
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_strict_test(
    name = "weakref_lru_cache_test",
    srcs = ["weakref_lru_cache_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

py_strict_test(
    name = "xla_client_test_gpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=gpu"],
    main = "xla_client_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",
        "requires-gpu-nvidia",
    ] + if_google(["config-cuda-only"]),  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + if_google(
        [":xla_gpu_extension"],
        [],
    ) + xla_py_test_deps(),
)

tsl_pybind_extension(
    name = "status_casters_ext",
    srcs = ["status_casters_ext.cc"],
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/nanobind",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

py_strict_test(
    name = "status_casters_test",
    srcs = ["status_casters_test.py"],
    main = "status_casters_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":status_casters_ext",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
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
        ":nb_helpers",
        ":nb_numpy",
        "//third_party/nanobind",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:exceptions",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
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
        "//third_party/nanobind",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
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
        ":nb_class_ptr",
        ":python_ref_manager",
        # placeholder for index annotation deps
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/pjrt:exceptions",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "pprof_profile_builder",
    srcs = ["pprof_profile_builder.cc"],
    hdrs = ["pprof_profile_builder.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/nanobind",
        "//xla:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:profile_proto_cc",
    ],
)

cc_library(
    name = "py_client",
    srcs = [
        "py_array.cc",
        "py_client.cc",
        "py_compile_only_client.cc",
        "py_device.cc",
        "py_device_list.cc",
        "py_executable.cc",
        "py_host_callback.cc",
        "py_memory_space.cc",
        "py_program.cc",
        "py_values.cc",
        "sharding.cc",
    ],
    hdrs = [
        "py_array.h",
        "py_client.h",
        "py_compile_only_client.h",
        "py_device.h",
        "py_device_list.h",
        "py_executable.h",
        "py_host_callback.h",
        "py_memory_space.h",
        "py_program.h",
        "py_values.h",
        "sharded_device_array.h",
        "sharding.h",
    ],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    features = ["-use_header_modules"],
    deps = [
        ":aggregate_profile",
        ":callback",
        ":nb_absl_span",
        ":nb_class_ptr",
        ":nb_helpers",
        ":nb_numpy",
        ":pprof_profile_builder",
        ":py_host_callback_proto_cc",
        ":python_ref_manager",
        ":traceback",
        ":transfer_guard_lib",
        ":types",
        ":util",
        ":xplane_to_profile_instructions",
        # placeholder for index annotation deps
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:arithmetic",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:lru_cache",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt:status_casters",
        "//xla/pjrt:transpose",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:plugin_program",
        "//xla/python/ifrt:plugin_program_serdes",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:xla_host_callback_proto_cc",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/python/lib/core:numpy",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
        # TODO(b/324133505): remove this dependency after JAX OSS migrates to cuda plugin.
        "//xla/stream_executor/cuda:cuda_driver",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rocm_headers",
    ]) + if_cuda_or_rocm([":py_client_gpu"]),  # TODO(b/337876408): remove after migration to plugin
)

cc_library(
    name = "callback",
    srcs = [
        "callback.cc",
    ],
    hdrs = [
        "callback.h",
    ],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":nb_numpy",
        ":python_ref_manager",
        "//third_party/nanobind",
        "//xla:comparison_util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:transpose",
        "//xla/service:custom_call_status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "py_client_gpu",
    srcs = if_google(
        ["py_client_gpu.cc"],
        if_cuda_or_rocm(["py_client_gpu.cc"]),
    ),
    hdrs = if_google(
        ["py_client_gpu.h"],
        if_cuda_or_rocm(["py_client_gpu.h"]),
    ),
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = if_cuda(["GOOGLE_CUDA=1"]) + if_rocm([
        "TENSORFLOW_USE_ROCM=1",
    ]),
    features = ["-use_header_modules"],
    deps = [
        ":callback",
        ":nb_numpy",
        "//third_party/nanobind",
        "//xla:comparison_util",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:host_callback",
        "//xla/service:custom_call_status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
    ] + if_rocm(
        ["@local_config_rocm//rocm:rocm_headers"],
        ["@local_config_cuda//cuda:cuda_headers"],
    ),
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
        ":nb_class_ptr",
        ":py_client",
        ":python_ref_manager",
        ":traceback",
        ":types",
        ":util",
        "//third_party/nanobind",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@llvm-project//llvm:Support",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
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
    visibility = [":friends"],  # For the functions to access C++ flags/thread-local variables
    deps = [
        ":nb_helpers",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # build_cleaner: keep
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:status_casters",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "inspect_sharding",
    srcs = ["inspect_sharding.cc"],
    hdrs = ["inspect_sharding.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/status",
    ],
    # Always register 'InspectSharding' custom partitioning handler.
    alwayslink = 1,
)

cc_library(
    name = "custom_partition_callback",
    srcs = ["custom_partition_callback.cc"],
    hdrs = ["custom_partition_callback.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/service:call_inliner",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_call_sharding",
    srcs = ["custom_call_sharding.cc"],
    hdrs = ["custom_call_sharding.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":custom_partition_callback",
        ":inspect_sharding",
        # placeholder for index annotation deps
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//third_party/nanobind",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
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
        ":nb_absl_span",
        ":nb_helpers",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:approx_topk",
        "//xla/client/lib:approx_topk_shape",
        "//xla/client/lib:comparators",
        "//xla/client/lib:lu_decomposition",
        "//xla/client/lib:math",
        "//xla/client/lib:qr",
        "//xla/client/lib:self_adjoint_eig",
        "//xla/client/lib:sorting",
        "//xla/client/lib:svd",
        "//xla/pjrt:status_casters",
    ],
)

cc_library(
    name = "outfeed_receiver",
    srcs = ["outfeed_receiver.cc"],
    hdrs = ["outfeed_receiver.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/client:executable_build_options",
        "//xla/client:sharding_builder",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/pjrt_ifrt",
        "//xla/service:computation_placer_hdr",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "pjit",
    srcs = ["pjit.cc"],
    hdrs = ["pjit.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":jax_jit",
        ":nb_helpers",
        ":nb_numpy",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":traceback",
        ":transfer_guard_lib",
        # placeholder for index annotation deps
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:util",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:lru_cache",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:status_casters",
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
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
        ":jax_jit",
        ":nb_class_ptr",
        ":nb_helpers",
        ":nb_numpy",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":traceback",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:status_casters",
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/python/lib/core:numpy",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "outfeed_receiver_test_cpu",
    size = "small",
    srcs = ["outfeed_receiver_test.cc"],
    deps = [
        ":outfeed_receiver",
        "//xla:test",
        "//xla/client:client_library",
        "//xla/client:executable_build_options",
        "//xla/client:xla_builder",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/service:platform_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:test_main",
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
        ":nb_class_ptr",
        ":outfeed_receiver",
        ":py_client",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "//third_party/nanobind",
        "//xla:literal",
        "//xla/client:executable_build_options",
        "//xla/client:xla_builder",
        "//xla/pjrt:status_casters",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "@local_tsl//tsl/platform:logging",
        "@llvm-project//llvm:Support",
    ],
)

py_strict_test(
    name = "pytree_test",
    srcs = ["pytree_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

tf_proto_library(
    name = "pytree_proto",
    srcs = ["pytree.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
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
        ":nb_class_ptr",
        ":pytree_proto_cc",
        # placeholder for index annotation deps
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/pjrt:exceptions",
        "@local_tsl//tsl/platform:logging",
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
        ":refine_polymorphic_shapes",
        # placeholder for index annotation deps
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//third_party/nanobind",
        "//xla/client:xla_computation",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:all_passes",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:status_casters",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_serialization",
    ],
)

cc_library(
    name = "refine_polymorphic_shapes",
    srcs = ["refine_polymorphic_shapes.cc"],
    hdrs = ["refine_polymorphic_shapes.h"],
    deps = [
        "//xla/mlir/utils:error_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@stablehlo//:base",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//stablehlo/experimental:experimental_stablehlo_passes",
    ],
)

cc_library(
    name = "profiler_utils",
    srcs = ["profiler_utils.cc"],
    hdrs = ["profiler_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/plugin:plugin_tracer",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
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
        ":aggregate_profile",
        ":profiler_utils",
        ":py_client",
        ":types",
        ":xplane_to_profile_instructions",
        # placeholder for index annotation deps
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/cpu:python_tracer",
        "//xla/backends/profiler/plugin:plugin_tracer",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/rpc:profiler_server_impl",
        "@local_tsl//tsl/profiler/rpc/client:capture_profile",
        "@local_tsl//tsl/profiler/rpc/client:profiler_client_impl",
    ] + select({
        ":gpu_enabled": [
            "//xla/backends/profiler/gpu:device_tracer",
        ],
        "//conditions:default": [],
    }),
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
        # placeholder for index annotation deps
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "//third_party/nanobind",
        "//xla:util",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
        "//xla/python/ifrt",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
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
        ":nb_helpers",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "//third_party/nanobind",
        "//xla/pjrt:lru_cache",
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
        ":nb_absl_span",
        ":nb_helpers",
        ":nb_numpy",
        ":py_client",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "//xla:array",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:status_casters",
        "//xla/service:call_inliner",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_target_registry",
        "//xla/service:flatten_call_graph",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_proto_cc",
        "//xla/service:name_uniquer",
        "//xla/service:tuple_simplifier",
        "@local_tsl//tsl/lib/strings:proto_serialization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "py_host_callback_proto",
    srcs = ["py_host_callback.proto"],
    cc_api_version = 2,
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

# If this flag is enabled, it sets RPATH on the xla_extension to values that are suitable for
# finding NVIDIA's CUDA libraries when they are installed as pip packages.
bool_flag(
    name = "jax_cuda_pip_rpaths",
    build_setting_default = False,
)

config_setting(
    name = "use_jax_cuda_pip_rpaths",
    flag_values = {
        ":jax_cuda_pip_rpaths": "True",
    },
)

# We cannot nest select and if_cuda_is_configured so we introduce
# a standalone cc_library target.
cc_library(
    name = "gpu_plugin_deps",
    deps = [
        "//xla/service:gpu_plugin",
    ] + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]),
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "@com_google_absl//absl/log:initialize",
    ],
)

cc_library(
    name = "gpu_support",
    srcs = ["gpu_support.cc"],
    hdrs = ["gpu_support.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":nb_class_ptr",
        ":py_client",
        "//third_party/nanobind",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/gpu:gpu_helpers",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/python/pjrt_ifrt",
        "@local_config_python//:python_headers",  # buildcleaner: keep
    ],
)

tsl_pybind_extension(
    name = "xla_gpu_extension",
    srcs = ["xla_gpu_support.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    pytype_deps = [],
    pytype_srcs = [],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_support",
        ":logging",
        ":py_client",
        # placeholder for index annotation deps
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
    ],
)

tsl_pybind_extension(
    name = "xla_extension",
    srcs = ["xla.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    defines = if_google(
        [],
        select({
            ":gpu_enabled": ["XLA_PYTHON_ENABLE_GPU=1"],
            "//conditions:default": [],
        }),
    ),
    features = ["-use_header_modules"],
    linkopts = select({
        ":use_jax_cuda_pip_rpaths": [
            "-Wl,-rpath,$$ORIGIN/../nvidia/cuda_cupti/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cuda_runtime/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cublas/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cufft/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cudnn/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/cusolver/lib",
            "-Wl,-rpath,$$ORIGIN/../nvidia/nccl/lib",
        ],
        "//conditions:default": [],
    }),
    pytype_deps = [
        "//third_party/py/numpy",
    ],
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        ":custom_call_sharding",
        ":dlpack",
        ":jax_jit",
        ":logging",
        ":mlir",
        ":nb_absl_flat_hash_map",
        ":nb_absl_span",
        ":nb_class_ptr",
        ":ops",
        ":outfeed_receiver_py",
        ":pjit",
        ":pmap_lib",
        ":pprof_profile_builder",
        ":profiler",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":refine_polymorphic_shapes",
        ":traceback",
        ":transfer_guard_lib",
        ":types",
        ":util",
        ":weakref_lru_cache",
        ":xla_compiler",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//third_party/nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla/ffi:ffi_api",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/pjrt/distributed:service",
        "//xla/python/ifrt",
        "//xla/python/ifrt:plugin_program",
        "//xla/python/ifrt:plugin_program_serdes",
        "//xla/python/ifrt_proxy/client:py_module",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service/cpu:collectives_interface",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/distributed_runtime/preemption:preemption_sync_manager",
        "//xla/tsl/python/lib/core:numpy",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform/cloud:gcs_file_system",
        "@llvm-project//mlir:IR",
    ] + select({
        # gloo transport only builds on linux
        "//xla/tsl:macos": [],
        "//xla/tsl:windows": [],
        "//conditions:default": [
            "//third_party/gloo:transport_tcp",
            "//xla/pjrt/cpu:gloo_collectives",
            "//xla/pjrt/cpu:gloo_kv_store",
        ],
    }) + select({
        # mpitrampoline does not build on windows
        "//xla/tsl:windows": [],
        "//conditions:default": [
            "//xla/pjrt/cpu:mpi_collectives",
        ],
    }) + if_google(
        [],
        select({
            ":gpu_enabled": [
                ":gpu_support",
            ],
            "//conditions:default": [],
        }) + select({
            ":link_gpu_plugin": [
                ":gpu_plugin_deps",
            ],
            "//conditions:default": [],
        }),
    ),
)

cc_library(
    name = "xplane_to_profile_instructions",
    srcs = ["xplane_to_profile_instructions.cc"],
    hdrs = ["xplane_to_profile_instructions.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/convert:xla_op_utils",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/utils:file_system_utils",
        "@local_tsl//tsl/profiler/utils:tf_xplane_visitor",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
        "@local_tsl//tsl/profiler/utils:xplane_utils",
        "@local_tsl//tsl/profiler/utils:xplane_visitor",
    ],
)

xla_cc_test(
    name = "xplane_to_profile_instructions_test",
    srcs = ["xplane_to_profile_instructions_test.cc"],
    deps = [
        ":xplane_to_profile_instructions",
        "//xla/service:hlo_proto_cc",
        "//xla/tests:verified_hlo_module",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/profiler/convert:xla_op_utils",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_tsl//tsl/profiler/rpc/client:save_profile",
        "@local_tsl//tsl/profiler/utils:file_system_utils",
        "@local_tsl//tsl/profiler/utils:xplane_builder",
        "@local_tsl//tsl/profiler/utils:xplane_schema",
    ],
)

cc_library(
    name = "nb_class_ptr",
    hdrs = ["nb_class_ptr.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = ["//third_party/nanobind"],
)

cc_library(
    name = "nb_helpers",
    srcs = ["nb_helpers.cc"],
    hdrs = ["nb_helpers.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/nanobind",
        "@com_google_absl//absl/strings:str_format",
        "@local_config_python//:python_headers",
    ],
)

cc_library(
    name = "nb_numpy",
    srcs = ["nb_numpy.cc"],
    hdrs = ["nb_numpy.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/nanobind",
        "//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",
    ],
)

cc_library(
    name = "nb_absl_span",
    hdrs = ["nb_absl_span.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/nanobind",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "nb_absl_flat_hash_map",
    hdrs = ["nb_absl_flat_hash_map.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/nanobind",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "nb_absl_flat_hash_set",
    hdrs = ["nb_absl_flat_hash_set.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//third_party/nanobind",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "aggregate_profile",
    srcs = ["aggregate_profile.cc"],
    hdrs = ["aggregate_profile.h"],
    deps = [
        ":xplane_to_profile_instructions",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ],
)

xla_cc_test(
    name = "aggregate_profile_test",
    srcs = ["aggregate_profile_test.cc"],
    deps = [
        ":aggregate_profile",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
    ],
)
