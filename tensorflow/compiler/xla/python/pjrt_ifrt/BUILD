load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load("//tensorflow/tsl/platform:build_config.bzl", "tf_proto_library")

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla/python:friends",
    ],
    packages = [
        "//tensorflow/compiler/xla/python/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "//tensorflow/compiler/xla/python/pjrt_ifrt/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":friends",
        ":internal",
    ],
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
    deps = [
        ":xla_compiler_proto_cc",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/python/ifrt:serdes",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_proto_library(
    name = "xla_host_callback_proto",
    srcs = ["xla_host_callback.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/compiler/xla:xla_data_proto"],
)

tf_proto_library(
    name = "xla_compiler_proto",
    srcs = ["xla_compiler.proto"],
    protodeps = ["//tensorflow/compiler/xla/pjrt:compile_options_proto"],
)

cc_library(
    name = "xla_program_serdes",
    srcs = ["xla_program_serdes.cc"],
    deps = [
        ":xla_ifrt",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "//tensorflow/compiler/xla/python/ifrt:serdes",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Support",
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
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/python/ifrt:serdes",
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
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/python/ifrt:types_proto",
    ],
)

cc_library(
    name = "xla_sharding_serdes",
    srcs = ["xla_sharding_serdes.cc"],
    deps = [
        ":xla_ifrt",
        ":xla_sharding_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/python/ifrt:serdes",
        "//tensorflow/compiler/xla/python/ifrt:sharding_serdes",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "xla_sharding_serdes_test",
    srcs = ["xla_sharding_serdes_test.cc"],
    deps = [
        ":xla_ifrt",
        ":xla_sharding_serdes",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/python/ifrt:sharding_serdes",
        "//tensorflow/compiler/xla/python/ifrt:sharding_test_util",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
    ],
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
cc_library(
    name = "xla_executable_impl_test_lib",
    testonly = True,
    srcs = ["xla_executable_impl_test_lib.cc"],
    deps = [
        ":xla_ifrt",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/python/ifrt:test_util",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

# TODO(hyeontaek): Move this target out of pjrt_ifrt.
xla_cc_test(
    name = "xla_executable_test_no_impl",
    srcs = [],
    deps = [
        ":xla_executable_impl_test_lib",
        "//tensorflow/compiler/xla/python/ifrt:no_impl_test_main",
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
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/python/ifrt:sharding_test_util",
        "//tensorflow/compiler/xla/python/ifrt:tuple_impl_test_lib",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
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
    deps = [
        ":xla_ifrt",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/pjrt:host_callback",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/compiler/xla/pjrt:utils",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@tf_runtime//:ref_count",
    ],
)

cc_library(
    name = "tfrt_cpu_client_test_lib",
    testonly = True,
    srcs = ["tfrt_cpu_client_test_lib.cc"],
    deps = [
        ":pjrt_ifrt",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/python/ifrt:test_util",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "pjrt_array_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["pjrt_array_impl_test_tfrt_cpu.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//tensorflow/compiler/xla/python/ifrt:array_impl_test_lib",
        "//tensorflow/compiler/xla/python/ifrt:test_util",
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
        "//tensorflow/compiler/xla/python/ifrt:client_impl_test_lib",
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
        "//tensorflow/compiler/xla/python/ifrt:test_util",
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
        "//tensorflow/compiler/xla/python/ifrt:tuple_impl_test_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
