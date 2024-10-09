load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google", "internal_visibility")
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
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:serdes",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "xla_host_callback_proto",
    srcs = ["xla_host_callback.proto"],
    protodeps = [
        "//xla:xla_data_proto",
    ] + if_google(["@com_google_protobuf//:any"]),
)

tf_proto_library(
    name = "xla_compiler_proto",
    srcs = ["xla_compiler.proto"],
    protodeps = ["//xla/pjrt:compile_options_proto"],
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
        "//xla/python/ifrt:device_test_util",
        "//xla/python/ifrt:serdes",
        "//xla/python/ifrt:sharding_serdes",
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
        "//xla/client:executable_build_options",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_executable",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
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
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:tile_assignment",
        "//xla/python/ifrt",
        "//xla/python/ifrt:device_test_util",
        "//xla/python/ifrt:tuple_impl_test_lib",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_ifrt",
    srcs = [
        "pjrt_array.cc",
        "pjrt_client.cc",
        "pjrt_compiler.cc",
        "pjrt_device.cc",
        "pjrt_executable.cc",
        "pjrt_host_callback.cc",
        "pjrt_memory.cc",
        "pjrt_remap.cc",
        "pjrt_topology.cc",
        "pjrt_tuple.cc",
    ],
    hdrs = [
        "pjrt_array.h",
        "pjrt_client.h",
        "pjrt_compiler.h",
        "pjrt_device.h",
        "pjrt_executable.h",
        "pjrt_host_callback.h",
        "pjrt_memory.h",
        "pjrt_remap.h",
        "pjrt_topology.h",
        "pjrt_tuple.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":basic_string_array",
        ":pjrt_attribute_map_util",
        ":pjrt_dtype",
        ":xla_ifrt",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "//xla/pjrt:host_callback",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:utils",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/pjrt/distributed:topology_util",
        "//xla/python/ifrt",
        "//xla/python/ifrt:attribute_map",
        "//xla/python/ifrt/hlo:hlo_program",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base:core_headers",
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
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
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

cc_library(
    name = "pjrt_attribute_map_util",
    srcs = ["pjrt_attribute_map_util.cc"],
    hdrs = ["pjrt_attribute_map_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/pjrt:pjrt_common",
        "//xla/python/ifrt:attribute_map",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

xla_cc_test(
    name = "pjrt_attribute_map_util_test",
    srcs = ["pjrt_attribute_map_util_test.cc"],
    deps = [
        ":pjrt_attribute_map_util",
        "//xla/pjrt:pjrt_common",
        "//xla/python/ifrt:attribute_map",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pjrt_dtype",
    srcs = ["pjrt_dtype.cc"],
    hdrs = ["pjrt_dtype.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/python/ifrt",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "basic_string_array",
    srcs = ["basic_string_array.cc"],
    hdrs = ["basic_string_array.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "basic_string_array_test",
    srcs = ["basic_string_array_test.cc"],
    deps = [
        ":basic_string_array",
        ":tfrt_cpu_client_test_lib",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt",
        "//xla/python/ifrt:test_util",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
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

xla_cc_test(
    name = "pjrt_remap_impl_test_tfrt_cpu",
    size = "small",
    srcs = [],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//xla/python/ifrt:remap_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
