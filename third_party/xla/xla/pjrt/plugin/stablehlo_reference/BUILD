load("//xla:xla.bzl", "xla_cc_binary", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    deps = [
        ":logging",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
        "//xla/hlo/translate/mhlo_to_hlo:literal_exporter",
        "//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_future",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    deps = [
        ":buffer",
        ":logging",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/translate:stablehlo",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/service:computation_placer_hdr",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:BytecodeWriter",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_tsl//tsl/platform:statusor",
        "@stablehlo//:reference_api",
        "@stablehlo//:stablehlo_passes",
    ],
)

cc_library(
    name = "device",
    srcs = ["device.cc"],
    hdrs = ["device.h"],
    deps = [
        ":logging",
        "//xla:util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "client_cpp_pjrt",
    srcs = [
        "client_cpp_pjrt.cc",
    ],
    hdrs = [
        "client_cpp_pjrt.h",
    ],
    deps = [
        ":buffer",
        ":device",
        ":executable",
        ":logging",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla/pjrt:host_memory_spaces",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

cc_library(
    name = "client_c_pjrt",
    srcs = [
        "client_c_pjrt.cc",
    ],
    hdrs = ["client_c_pjrt.h"],
    deps = [
        ":client_cpp_pjrt",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_layouts_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    deps = [
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:globals",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:DataLayoutInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "plugin_pjrt_test",
    srcs = ["plugin_pjrt_test.cc"],
    deps = [
        ":client_c_pjrt",
        ":client_cpp_pjrt",
        "//xla/pjrt:pjrt_client_test_common",
        "//xla/pjrt/c:pjrt_c_api_test_common",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_binary(
    name = "stablehlo_reference_plugin.so",
    linkshared = True,
    deps = [
        ":client_c_pjrt",
    ],
)
