load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "kernel_runner",
    testonly = 1,
    srcs = ["kernel_runner.cc"],
    hdrs = ["kernel_runner.h"],
    deps = [
        ":llvm_ir_kernel_spec",
        "//xla/backends/cpu/codegen:jit_compiler",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/backends/cpu/runtime:kernel",
        "//xla/backends/cpu/runtime:kernel_c_api",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/codegen/testlib:kernel_runner",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Target",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "kernel_runner_test",
    srcs = ["kernel_runner_test.cc"],
    deps = [
        ":kernel_runner",
        ":llvm_ir_kernel_emitter",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla/codegen:kernel_spec",
        "//xla/codegen/testlib:kernel_runner",
        "//xla/runtime:buffer_use",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "llvm_ir_kernel_emitter",
    testonly = 1,
    srcs = ["llvm_ir_kernel_emitter.cc"],
    hdrs = ["llvm_ir_kernel_emitter.h"],
    deps = [
        ":llvm_ir_kernel_spec",
        "//xla:util",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "llvm_ir_kernel_spec",
    srcs = ["llvm_ir_kernel_spec.cc"],
    hdrs = ["llvm_ir_kernel_spec.h"],
    deps = [
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
    ],
)

xla_cc_test(
    name = "llvm_ir_kernel_emitter_test",
    srcs = ["llvm_ir_kernel_emitter_test.cc"],
    deps = [
        ":llvm_ir_kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/runtime:buffer_use",
        "//xla/stream_executor:launch_dim",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
