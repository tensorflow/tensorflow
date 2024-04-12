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
        "//xla/python/ifrt/...",
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

cc_library(
    name = "ifrt",
    srcs = [
        "array.cc",
        "client.cc",
        "compiler.cc",
        "device.cc",
        "dtype.cc",
        "executable.cc",
        "future.cc",
        "host_callback.cc",
        "index.cc",
        "index_domain.cc",
        "memory.cc",
        "program.cc",
        # TODO(hyeontaek): Move this SerDes out of `ifrt` target once `Compiler::DeserializeLoadedExecutable()` is split into executable deserialization and executable loading, and thus the method can be removed.
        "executable_serdes.cc",
        "shape.cc",
        "sharding.cc",
        "tuple.cc",
        "value.cc",
    ],
    hdrs = [
        "array.h",
        "client.h",
        "compiler.h",
        "device.h",
        "dtype.h",
        "executable.h",
        "future.h",
        "host_callback.h",
        "index.h",
        "index_domain.h",
        "memory.h",
        "program.h",
        # TODO(hyeontaek): Move this SerDes out of `ifrt` target once `Compiler::DeserializeLoadedExecutable()` is split into executable deserialization and executable loading, and thus the method can be removed.
        "executable_serdes.h",
        "shape.h",
        "sharding.h",
        "tuple.h",
        "value.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":device_proto_cc",
        ":dtype_proto_cc",
        ":serdes",
        ":shape_proto_cc",
        ":sharding_proto_cc",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt/ir",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":ifrt",
        ":mock",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "future_test",
    size = "small",
    srcs = ["future_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

xla_cc_test(
    name = "index_domain_test",
    size = "small",
    srcs = ["index_domain_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "memory_test",
    size = "small",
    srcs = ["memory_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "shape_test",
    size = "small",
    srcs = ["shape_test.cc"],
    deps = [
        ":ifrt",
        ":shape_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "sharding_test",
    size = "small",
    srcs = ["sharding_test.cc"],
    deps = [
        ":ifrt",
        ":sharding_test_util",
        "//xla/python/ifrt/ir",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":ifrt",
        "//xla:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "sharding_test_util",
    testonly = True,
    srcs = ["sharding_test_util.cc"],
    hdrs = ["sharding_test_util.h"],
    deps = [
        ":ifrt",
        ":mock",
        ":test_util",
        "//xla/pjrt:pjrt_common",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "no_impl_test_main",
    testonly = True,
    srcs = ["no_impl_test_main.cc"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "array_impl_test_lib",
    testonly = True,
    srcs = ["array_impl_test_lib.cc"],
    deps = [
        ":ifrt",
        ":test_util",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "array_test_no_impl",
    srcs = [],
    deps = [
        ":array_impl_test_lib",
        ":no_impl_test_main",
    ],
)

cc_library(
    name = "client_impl_test_lib",
    testonly = True,
    srcs = ["client_impl_test_lib.cc"],
    deps = [
        ":ifrt",
        ":test_util",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "client_test_no_impl",
    srcs = [],
    deps = [
        ":client_impl_test_lib",
        ":no_impl_test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tuple_impl_test_lib",
    testonly = True,
    srcs = ["tuple_impl_test_lib.cc"],
    deps = [
        ":ifrt",
        ":test_util",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "tuple_test_no_impl",
    srcs = [],
    deps = [
        ":no_impl_test_main",
        ":tuple_impl_test_lib",
    ],
)

cc_library(
    name = "mock",
    testonly = True,
    srcs = ["mock.cc"],
    hdrs = ["mock.h"],
    deps = [
        ":ifrt",
        "//xla:literal",
        "//xla:test",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_layout",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/concurrency:ref_count",
    ],
)

cc_library(
    name = "serdes",
    srcs = ["serdes.cc"],
    hdrs = ["serdes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":serdes_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "serdes_test",
    srcs = ["serdes_test.cc"],
    deps = [
        ":serdes",
        ":serdes_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "serdes_proto",
    srcs = ["serdes.proto"],
)

cc_library(
    name = "program_serdes",
    srcs = ["program_serdes.cc"],
    hdrs = ["program_serdes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":ifrt",
        ":serdes",
        "//xla:util",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "sharding_serdes",
    srcs = ["sharding_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":ifrt",
        ":serdes",
        ":sharding_serdes_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "sharding_serdes_test",
    srcs = ["sharding_serdes_test.cc"],
    deps = [
        ":ifrt",
        ":serdes",
        ":serdes_proto_cc",
        ":sharding_serdes",
        ":sharding_test_util",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "device_proto",
    srcs = ["device.proto"],
)

xla_cc_test(
    name = "device_test",
    size = "small",
    srcs = ["device_test.cc"],
    deps = [
        ":device_proto_cc",
        ":ifrt",
        ":sharding_test_util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "dtype_proto",
    srcs = ["dtype.proto"],
)

xla_cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    deps = [
        ":dtype_proto_cc",
        ":ifrt",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

tf_proto_library(
    name = "shape_proto",
    srcs = ["shape.proto"],
)

tf_proto_library(
    name = "sharding_proto",
    srcs = ["sharding.proto"],
    protodeps = [":serdes_proto"],
)

tf_proto_library(
    name = "sharding_serdes_proto",
    srcs = ["sharding_serdes.proto"],
    protodeps = [
        ":device_proto",
        ":shape_proto",
    ],
)

cc_library(
    name = "plugin_program",
    srcs = ["plugin_program.cc"],
    hdrs = ["plugin_program.h"],
    deps = [
        ":ifrt",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "plugin_program_serdes",
    srcs = ["plugin_program_serdes.cc"],
    deps = [
        ":plugin_program",
        ":serdes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "plugin_program_serdes_test",
    srcs = ["plugin_program_serdes_test.cc"],
    deps = [
        ":plugin_program",
        ":plugin_program_serdes",
        ":serdes",
        ":serdes_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_tsl//tsl/protobuf:status_proto_cc",
    ],
)
