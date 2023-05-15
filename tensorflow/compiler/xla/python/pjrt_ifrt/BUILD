load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

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
    default_visibility = [
        ":friends",
        ":internal",
    ],
    licenses = ["notice"],
)

exports_files([
    "BUILD",
])

cc_library(
    name = "pjrt_ifrt",
    srcs = [
        "pjrt_array.cc",
        "pjrt_client.cc",
        "pjrt_compiler.cc",
        "pjrt_executable.cc",
        "pjrt_tuple.cc",
    ],
    hdrs = [
        "pjrt_array.h",
        "pjrt_client.h",
        "pjrt_compiler.h",
        "pjrt_executable.h",
        "pjrt_tuple.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_executable",
        "//tensorflow/compiler/xla/pjrt:utils",
        "//tensorflow/compiler/xla/python/ifrt",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@tf_runtime//:ref_count",
    ],
)

cc_library(
    name = "tfrt_cpu_client_test_lib",
    testonly = 1,
    srcs = ["tfrt_cpu_client_test_lib.cc"],
    deps = [
        ":pjrt_ifrt",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/python/ifrt:test_util",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "pjrt_array_impl_test_tfrt_cpu",
    size = "small",
    srcs = ["pjrt_array_impl_test_tfrt_cpu.cc"],
    deps = [
        ":tfrt_cpu_client_test_lib",
        "//tensorflow/compiler/xla/python/ifrt:array_impl_test_lib",
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
        "//tensorflow/compiler/xla/python/ifrt:executable_impl_test_lib",
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
