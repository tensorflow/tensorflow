load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package_group(
    name = "friends",
    includes = [
        "//xla/python:friends",
    ],
    packages = [
        "//xla/python/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "//xla/python/pjrt_ifrt/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        ":friends",
        ":internal",
    ]),
)

exports_files([
    "BUILD",
])

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
cc_library(
    name = "xla_ifrt",
    srcs = [
        "xla_compiler.cc",
        "xla_sharding.cc",
    ],
    hdrs = [
        "xla_compiler.h",
        "xla_sharding.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":xla_compiler_proto_cc",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "xla_host_callback_proto",
    srcs = ["xla_host_callback.proto"],
    cc_api_version = 2,
    protodeps = ["//xla:xla_data_proto"],
)

tf_proto_library(
    name = "xla_compiler_proto",
    srcs = ["xla_compiler.proto"],
    protodeps = ["//xla/pjrt:compile_options_proto"],
)

cc_library(
    name = "xla_program_serdes",
    srcs = ["xla_program_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":xla_ifrt",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/python/ifrt:serdes",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:status",
        "@stablehlo//:stablehlo_portable_api",
        "@stablehlo//:stablehlo_serialization",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "xla_program_serdes_test",
    srcs = ["xla_program_serdes_test.cc"],
    deps = [
        ":xla_ifrt",
        ":xla_program_serdes",
        "//xla/mlir_hlo",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/python/ifrt:serdes",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_proto_library(
    name = "xla_sharding_proto",
    srcs = ["xla_sharding.proto"],
    protodeps = [
        "//xla:xla_data_proto",
        "//xla/python/ifrt:device_proto",
    ],
)

cc_library(
    name = "xla_sharding_serdes",
    srcs = ["xla_sharding_serdes.cc"],
    deps = [
        ":xla_ifrt",
        ":xla_sharding_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "xla_sharding_serdes_test",
    srcs = ["xla_sharding_serdes_test.cc"],
    deps = [
        ":xla_ifrt",
        ":xla_sharding_serdes",
        "//xla/hlo/ir:hlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
        "//xla/python/ifrt:sharding_test_util",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
cc_library(
    name = "xla_executable_impl_test_lib",
    testonly = True,
    srcs = ["xla_executable_impl_test_lib.cc"],
    deps = [
        ":xla_ifrt",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
xla_cc_test(
    name = "xla_executable_test_no_impl",
    srcs = [],
    deps = [
        ":xla_executable_impl_test_lib",
        "//xla/python/ifrt:no_impl_test_main",
        "@com_google_googletest//:gtest",
    ],
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
xla_cc_test(
    name = "xla_sharding_test",
    size = "small",
    srcs = ["xla_sharding_test.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        ":xla_ifrt",
        "//xla:xla_data_proto_cc",
        "//xla/python/ifrt",
        "//xla/python/ifrt:sharding_test_util",
        "//xla/python/ifrt:tuple_impl_test_lib",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "pjrt_ifrt",
    srcs = [
        "pjrt_array.cc",
        "pjrt_client.cc",
        "pjrt_compiler.cc",
        "pjrt_executable.cc",
        "pjrt_host_callback.cc",
        "pjrt_tuple.cc",
    ],
    hdrs = [
        "pjrt_array.h",
        "pjrt_client.h",
        "pjrt_compiler.h",
        "pjrt_executable.h",
        "pjrt_host_callback.h",
        "pjrt_tuple.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":xla_ifrt",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:utils",
        "//xla/python/ifrt",
        "//xla/service:hlo_proto_cc",
        "//xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tfrt_cpu_client_test_lib",
    testonly = True,
    srcs = ["tfrt_cpu_client_test_lib.cc"],
    deps = [
        ":pjrt_ifrt",
        "//xla/pjrt/cpu:cpu_client",
        "//xla/python/ifrt:test_util",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "pjrt_array_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["pjrt_array_impl_test_tfrt_cpu.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:array_impl_test_lib",
        "//xla/python/ifrt:test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "pjrt_client_impl_test_tfrt_cpu",
    size = "small",
    srcs = [],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:client_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "pjrt_executable_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["pjrt_executable_impl_test_tfrt_cpu.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        ":xla_executable_impl_test_lib",
        "//xla/python/ifrt:test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "pjrt_tuple_impl_test_tfrt_cpu",
    size = "small",
    srcs = [],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:tuple_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
