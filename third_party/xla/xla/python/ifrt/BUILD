load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

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
        "array_spec.cc",
        "array.cc",
        "client.cc",
        "compiler.cc",
        "device.cc",
        "device_list.cc",
        "dtype.cc",
        "remap_plan.cc",
        "executable.cc",
        "host_callback.cc",
        "index.cc",
        "index_domain.cc",
        "memory.cc",
        "program.cc",
        # TODO(hyeontaek): Move this SerDes out of `ifrt` target once `Compiler::DeserializeLoadedExecutable()` is split into executable deserialization and executable loading, and thus the method can be removed.
        "executable_serdes.cc",
        "shape.cc",
        "sharding.cc",
        "topology.cc",
        "tuple.cc",
        "value.cc",
    ],
    hdrs = [
        "array.h",
        "array_spec.h",
        "client.h",
        "compiler.h",
        "remap_plan.h",
        "device.h",
        "device_list.h",
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
        "topology.h",
        "tuple.h",
        "value.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":array_spec_proto_cc",
        ":attribute_map",
        ":device_proto_cc",
        ":dtype_proto_cc",
        ":execute_options_proto_cc",
        ":remap_plan_proto_cc",
        ":serdes",
        ":shape_proto_cc",
        ":sharding_proto_cc",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/gtl:int_type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
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
        "//xla/tsl/concurrency:ref_count",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "attribute_map",
    srcs = ["attribute_map.cc"],
    hdrs = ["attribute_map.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":attribute_map_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "attribute_map_test",
    size = "small",
    srcs = ["attribute_map_test.cc"],
    deps = [
        ":attribute_map",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    protodeps = [":attribute_map_proto"],
)

xla_cc_test(
    name = "executable_test",
    size = "small",
    srcs = ["executable_test.cc"],
    deps = [
        ":attribute_map",
        ":ifrt",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "future_test",
    size = "small",
    srcs = ["future_test.cc"],
    deps = [
        ":ifrt",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

xla_cc_test(
    name = "index_domain_test",
    size = "small",
    srcs = ["index_domain_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "memory_test",
    size = "small",
    srcs = ["memory_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/hash:hash_testing",
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
        ":device_test_util",
        ":ifrt",
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/concurrency:ref_count",
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
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "device_test_util",
    testonly = True,
    srcs = ["device_test_util.cc"],
    hdrs = ["device_test_util.h"],
    deps = [
        ":ifrt",
        ":mock",
        ":test_util",
        "//xla:util",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//xla/python/ifrt/ir:sharding_param",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
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

tf_proto_library(
    name = "attribute_map_proto",
    srcs = ["attribute_map.proto"],
)

cc_library(
    name = "client_impl_test_lib",
    testonly = True,
    srcs = ["client_impl_test_lib.cc"],
    deps = [
        ":ifrt",
        ":test_util",
        "@local_tsl//tsl/platform:statusor",
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
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/types:span",
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
        ":attribute_map",
        ":ifrt",
        "//xla:test",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
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
        "//xla/python/ifrt/ir:sharding_param",
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
        ":device_test_util",
        ":ifrt",
        ":serdes",
        ":serdes_proto_cc",
        ":sharding_serdes",
        "//xla/python/ifrt/ir:sharding_param",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "array_spec_proto",
    srcs = ["array_spec.proto"],
    protodeps = [
        ":dtype_proto",
        ":shape_proto",
        ":sharding_proto",
    ],
)

xla_cc_test(
    name = "array_spec_test",
    size = "small",
    srcs = ["array_spec_test.cc"],
    deps = [
        ":array_spec_proto_cc",
        ":device_test_util",
        ":ifrt",
        ":sharding_serdes",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_proto_library(
    name = "device_proto",
    srcs = ["device.proto"],
)

xla_cc_test(
    name = "device_list_test",
    size = "small",
    srcs = ["device_list_test.cc"],
    deps = [
        ":device_proto_cc",
        ":device_test_util",
        ":ifrt",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
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
    name = "remap_plan_proto",
    srcs = ["remap_plan.proto"],
    protodeps = [":array_spec_proto"],
)

xla_cc_test(
    name = "remap_plan_test",
    size = "small",
    srcs = ["remap_plan_test.cc"],
    deps = [
        ":device_test_util",
        ":ifrt",
        ":sharding_serdes",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "remap_impl_test_lib",
    testonly = True,
    srcs = ["remap_impl_test_lib.cc"],
    deps = [
        ":ifrt",
        ":test_util",
        "//xla:status_macros",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
    alwayslink = True,
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
        "//xla/python/ifrt/ir:sharding_param_proto",
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
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/protobuf:status_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "custom_call_program",
    srcs = ["custom_call_program.cc"],
    hdrs = ["custom_call_program.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":ifrt",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/strings:cord",
        "@llvm-project//llvm:Support",
    ],
)

tf_proto_library(
    name = "custom_call_program_proto",
    srcs = ["custom_call_program.proto"],
    protodeps = [
        ":array_spec_proto",
        ":device_proto",
    ],
)

cc_library(
    name = "custom_call_program_serdes",
    srcs = ["custom_call_program_serdes.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":array_spec_proto_cc",
        ":custom_call_program",
        ":custom_call_program_proto_cc",
        ":ifrt",
        ":program_serdes",
        ":serdes",
        ":sharding_proto_cc",
        ":sharding_serdes",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "custom_call_program_serdes_test",
    srcs = ["custom_call_program_serdes_test.cc"],
    deps = [
        ":custom_call_program",
        ":custom_call_program_serdes",
        ":device_test_util",
        ":ifrt",
        ":program_serdes",
        ":serdes",
        ":serdes_proto_cc",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)
