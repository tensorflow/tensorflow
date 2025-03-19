load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_library", "py_strict_test")
load(
    "//xla:xla.bzl",
    "xla_cc_test",
    "xla_py_test_deps",
)
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "if_oss",
    "internal_visibility",
)
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable", "tsl_pybind_extension")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//visibility:private"]),
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
    visibility = ["//visibility:public"],
    deps = [
        ":xla_extension",  # buildcleaner: keep
    ] + if_google([
        "@ml_dtypes_py//ml_dtypes",
        "//third_party/py/numpy",
    ]),
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
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:ffi",
        "@com_google_absl//absl/status",
        "@nanobind",
    ],
)

py_strict_test(
    name = "xla_client_backend_independent_test",
    srcs = ["xla_client_backend_independent_test.py"],
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/testing:absltest",
    ] + if_google([
        "//third_party/py/numpy",
        "//third_party/py/portpicker",
    ]) + xla_py_test_deps(),
)

py_strict_library(
    name = "xla_client_test",
    testonly = 1,
    srcs = ["xla_client_test.py"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":xla_client",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@ml_dtypes_py//ml_dtypes",
    ] + if_google(["//third_party/py/numpy"]),
)

py_strict_test(
    name = "xla_client_test_cpu",
    srcs = ["xla_client_test.py"],
    args = ["--backend=cpu"],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    main = "xla_client_test.py",
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
        "@ml_dtypes_py//ml_dtypes",
    ] + if_google(["//third_party/py/numpy"]) + xla_py_test_deps(),
)

py_strict_test(
    name = "weakref_lru_cache_test",
    srcs = ["weakref_lru_cache_test.py"],
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
)

tsl_pybind_extension(
    name = "status_casters_ext",
    srcs = ["status_casters_ext.cc"],
    visibility = ["//visibility:private"],
    deps = [
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@nanobind",
    ],
)

py_strict_test(
    name = "status_casters_test",
    srcs = ["status_casters_test.py"],
    main = "status_casters_test.py",
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
    visibility = internal_visibility([":friends"]),
    deps = [
        ":nb_numpy",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:exceptions",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt:pjrt_dtype",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/python/lib/core:numpy",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@nanobind",
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
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@nanobind",
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
    visibility = internal_visibility([":friends"]),
    deps = [
        ":nb_class_ptr",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/pjrt:exceptions",
        "//xla/tsl/platform:logging",
        "@local_tsl//tsl/platform",
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
        "//xla:util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/protobuf:profile_proto_cc",
        "@nanobind",
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
        "py_memory_space.cc",
        "py_program.cc",
        "py_values.cc",
        "sharding.cc",
        "to_ifrt_sharding.cc",
    ],
    hdrs = [
        "py_array.h",
        "py_client.h",
        "py_compile_only_client.h",
        "py_device.h",
        "py_device_list.h",
        "py_executable.h",
        "py_memory_space.h",
        "py_program.h",
        "py_values.h",
        "sharded_device_array.h",
        "sharding.h",
        "to_ifrt_sharding.h",
    ],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":aggregate_profile",
        ":callback",
        ":guard_lib",
        ":nb_absl_span",
        ":nb_class_ptr",
        ":nb_helpers",
        ":nb_numpy",
        ":pprof_profile_builder",
        ":py_client_cpu",
        ":py_host_callback",
        ":py_host_callback_proto_cc",
        ":python_ref_manager",
        ":traceback",
        ":types",
        ":util",
        ":xplane_to_profile_instructions",
        # placeholder for index annotation deps
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/cleanup",
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
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@nanobind",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:lru_cache",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:status_casters",
        "//xla/pjrt:transpose",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/python/compile_only_ifrt:client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt:custom_call_program",
        "//xla/python/ifrt:plugin_program",
        "//xla/python/ifrt:plugin_program_serdes",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:pjrt_attribute_map_util",
        "//xla/python/pjrt_ifrt:pjrt_dtype",
        "//xla/python/pjrt_ifrt:xla_host_callback_proto_cc",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/service:computation_placer_hdr",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/framework:allocator",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/python/lib/core:numpy",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
    ] + if_google(["@com_google_protobuf//:any_cc_proto"]),
)

cc_library(
    name = "py_host_callback",
    srcs = ["py_host_callback.cc"],
    hdrs = ["py_host_callback.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":callback",
        ":py_host_callback_proto_cc",
        ":python_ref_manager",
        ":types",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:pjrt_compiler",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:xla_host_callback_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@nanobind",
    ] + if_google([
        "@com_google_protobuf//:any_cc_proto",
    ]),
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
    visibility = internal_visibility([":friends"]),
    deps = [
        ":nb_numpy",
        ":python_ref_manager",
        "//xla:comparison_util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:transpose",
        "//xla/service:custom_call_status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@nanobind",
    ],
)

cc_library(
    name = "py_client_cpu",
    srcs = ["py_client_cpu.cc"],
    hdrs = ["py_client_cpu.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":callback",
        ":nb_numpy",
        ":py_host_callback",
        ":types",
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:transpose",
        "//xla/python/ifrt",
        "//xla/service:custom_call_status",
        "//xla/service:custom_call_target_registry",
        "//xla/service:platform_util",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:errors",
        "@nanobind",
    ],
    alwayslink = 1,
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
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/pjrt_ifrt",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@dlpack",
        "@llvm-project//llvm:Support",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@nanobind",
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
    visibility = ["//visibility:private"],  # For the functions to access C++ flags/thread-local variables
    deps = [
        ":nb_absl_inlined_vector",
        ":nb_absl_span",
        ":nb_helpers",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":types",
        # placeholder for index annotation deps
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@local_config_python//:python_headers",  # build_cleaner: keep
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:status_casters",
        "//xla/tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "inspect_sharding",
    srcs = ["inspect_sharding.cc"],
    hdrs = ["inspect_sharding.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:xla_data_proto_cc",
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
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:debug_options_flags",
        "//xla:util",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/service:call_inliner",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_call_batch_partitioner",
    srcs = ["custom_call_batch_partitioner.cc"],
    hdrs = ["custom_call_batch_partitioner.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service/spmd:spmd_partitioner",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
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
        ":custom_call_batch_partitioner",
        ":custom_partition_callback",
        ":inspect_sharding",
        # placeholder for index annotation deps
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@nanobind",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
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
        "@nanobind",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:approx_topk",
        "//xla/hlo/builder/lib:approx_topk_shape",
        "//xla/hlo/builder/lib:comparators",
        "//xla/hlo/builder/lib:lu_decomposition",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:qr",
        "//xla/hlo/builder/lib:self_adjoint_eig",
        "//xla/hlo/builder/lib:sorting",
        "//xla/hlo/builder/lib:svd",
        "//xla/pjrt:status_casters",
        "//xla/service:hlo_proto_cc",
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
        ":config",
        ":guard_lib",
        ":jax_jit",
        ":nb_class_ptr",
        ":nb_helpers",
        ":nb_numpy",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":traceback",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:shape_util",
        "//xla:util",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:lru_cache",
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
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
        ":config",
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
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/python/lib/core:numpy",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

py_strict_test(
    name = "pytree_test",
    srcs = ["pytree_test.py"],
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
    visibility = internal_visibility([":friends"]),
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
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/pjrt:exceptions",
        "//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        ":python_ref_manager",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/tsl/platform:logging",
    ],
)

py_strict_test(
    name = "config_test",
    srcs = ["config_test.py"],
    tags = ["no_oss"],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
    ] + xla_py_test_deps(),
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
        "@nanobind",
        "@stablehlo//:stablehlo_serialization",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/translate:stablehlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:status_casters",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "sdy",
    srcs = ["sdy.cc"],
    hdrs = ["sdy.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        # placeholder for index annotation deps
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@nanobind",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/import:passes",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
        "//xla/mlir_hlo:all_passes",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:status_casters",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/sdy_round_trip:import_shardy_attrs",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
    ],
)

cc_library(
    name = "refine_polymorphic_shapes",
    srcs = ["refine_polymorphic_shapes.cc"],
    hdrs = ["refine_polymorphic_shapes.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "//xla/service/spmd/shardy/round_trip_common:import_constants",
        "//xla/service/spmd/shardy/sdy_round_trip:pipelines",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
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
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:base",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "profiler_utils",
    srcs = ["profiler_utils.cc"],
    hdrs = ["profiler_utils.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([":friends"]),
    deps = [
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/plugin:plugin_tracer",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        ":xplane_to_profile_instructions",
        # placeholder for index annotation deps
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        "//xla/backends/profiler:profiler_backends",
        "//xla/backends/profiler/cpu:python_tracer",
        "//xla/backends/profiler/plugin:plugin_tracer",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_profiler_extension_hdrs",
        "//xla/python/profiler:profile_data_lib",
        "//xla/tsl/platform:macros",
        "//xla/tsl/profiler/rpc:profiler_server_impl",
        "//xla/tsl/profiler/rpc/client:capture_profile",
        "//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "guard_lib",
    srcs = ["guard_lib.cc"],
    hdrs = ["guard_lib.h"],
    compatible_with = [],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:private"],
    deps = [
        # placeholder for index annotation deps
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@nanobind",
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
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
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
        # placeholder for index annotation deps
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@nanobind",
        "@local_config_python//:python_headers",
        "//xla/pjrt:lru_cache",
        "//xla/tsl/platform:logging",
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
        ":dlpack",
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
        "@nanobind",
        "//xla:array",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/transforms/simplifiers:flatten_call_graph",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "//xla/hlo/transforms/simplifiers:tuple_simplifier",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:status_casters",
        "//xla/service:call_inliner",
        "//xla/service:computation_placer",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:name_uniquer",
        "//xla/tsl/lib/strings:proto_serialization",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "py_host_callback_proto",
    srcs = ["py_host_callback.proto"],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/log:initialize",
    ],
)

tsl_pybind_extension(
    name = "xla_extension",
    srcs = ["xla.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    pytype_deps = [
        "//third_party/py/numpy",
    ],
    pytype_srcs = glob(["xla_extension/*.pyi"]),
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":custom_call_sharding",
        ":dlpack",
        ":guard_lib",
        ":jax_jit",
        ":logging",
        ":mlir",
        ":nb_absl_flat_hash_map",
        ":nb_absl_span",
        ":nb_class_ptr",
        ":ops",
        ":pjit",
        ":pmap_lib",
        ":pprof_profile_builder",
        ":profiler",
        ":py_client",
        ":python_ref_manager",
        ":pytree",
        ":refine_polymorphic_shapes",
        ":sdy",
        ":traceback",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla/backends/cpu/collectives:cpu_collectives",
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
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/pjrt/distributed:service",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:plugin_program",
        "//xla/python/ifrt:plugin_program_serdes",
        "//xla/python/ifrt_proxy/client:py_module",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:pjrt_attribute_map_util",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/distributed_runtime/preemption:preemption_sync_manager",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform/cloud:gcs_file_system",
        "//xla/tsl/python/lib/core:numpy",
        "@local_tsl//tsl/platform",
    ] + select({
        # gloo tcp transport only builds on linux
        "//xla/tsl:macos": [
            "//xla/backends/cpu/collectives:gloo_collectives",
            "//xla/backends/cpu/collectives:gloo_kv_store",
            "@gloo//:transport_uv",
        ],
        "//xla/tsl:windows": [],
        "//conditions:default": [
            "//xla/backends/cpu/collectives:gloo_collectives",
            "//xla/backends/cpu/collectives:gloo_kv_store",
            "//xla/python/transfer:py_socket_transfer",
            "@gloo//:transport_tcp",
        ],
    }) + select({
        # mpitrampoline does not build on windows
        "//xla/tsl:windows": [],
        # we support MPI collectives only in OSS builds
        "//conditions:default": if_oss(["//xla/backends/cpu/collectives:mpi_collectives"]),
    }),
)

cc_library(
    name = "xplane_to_profile_instructions",
    srcs = ["xplane_to_profile_instructions.cc"],
    hdrs = ["xplane_to_profile_instructions.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/convert:xla_op_utils",
        "//xla/tsl/profiler/utils:file_system_utils",
        "//xla/tsl/profiler/utils:tf_xplane_visitor",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "xplane_to_profile_instructions_test",
    srcs = ["xplane_to_profile_instructions_test.cc"],
    deps = [
        ":xplane_to_profile_instructions",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/profiler/convert:xla_op_utils",
        "//xla/tsl/profiler/rpc/client:save_profile",
        "//xla/tsl/profiler/utils:file_system_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "nb_class_ptr",
    hdrs = ["nb_class_ptr.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = ["@nanobind"],
)

cc_library(
    name = "nb_helpers",
    hdrs = ["nb_helpers.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@local_config_python//:python_headers",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_numpy",
    srcs = ["nb_numpy.cc"],
    hdrs = ["nb_numpy.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@nanobind",
        # copybara:uncomment "//third_party/py/numpy:multiarray",
        "@local_config_python//:python_headers",
        "//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "nb_absl_inlined_vector",
    hdrs = ["nb_absl_inlined_vector.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_absl_span",
    hdrs = ["nb_absl_span.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/types:span",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_absl_flat_hash_map",
    hdrs = ["nb_absl_flat_hash_map.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@nanobind",
    ],
)

cc_library(
    name = "nb_absl_flat_hash_set",
    hdrs = ["nb_absl_flat_hash_set.h"],
    compatible_with = [],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = internal_visibility([":friends"]),
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@nanobind",
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
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:profiled_instructions_proto_cc_impl",
    ],
)

py_strict_test(
    name = "jax_jit_test",
    srcs = ["jax_jit_test.py"],
    main = "jax_jit_test.py",
    tags = [
        "no_oss",
        "not_run:arm",
    ],  # TODO(phawkins): This test passes, but requires --config=monolithic.
    deps = [
        ":xla_client",
        ":xla_extension",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)

py_strict_test(
    name = "xla_compiler_test",
    srcs = ["xla_compiler_test.py"],
    main = "xla_compiler_test.py",
    tags = ["no_oss"],
    deps = [
        ":xla_extension",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ] + xla_py_test_deps(),
)
