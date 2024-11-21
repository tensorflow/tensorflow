load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

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
    name = "llvm_ir_kernel_emitter",
    testonly = 1,
    srcs = ["llvm_ir_kernel_emitter.cc"],
    hdrs = ["llvm_ir_kernel_emitter.h"],
    deps = [
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

xla_cc_test(
    name = "llvm_ir_kernel_emitter_test",
    srcs = ["llvm_ir_kernel_emitter_test.cc"],
    deps = [
        ":llvm_ir_kernel_emitter",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/runtime:buffer_use",
        "//xla/stream_executor:launch_dim",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
