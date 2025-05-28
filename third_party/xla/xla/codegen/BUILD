load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")
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
    name = "autotuner",
    hdrs = ["autotuner.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/runtime:thunk",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "device_spec",
    hdrs = ["device_spec.h"],
    deps = [
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "emitter_loc_op_builder",
    srcs = ["emitter_loc_op_builder.cc"],
    hdrs = ["emitter_loc_op_builder.h"],
    visibility = [
        "//xla/backends:__subpackages__",
        "//xla/codegen/emitters/transforms:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform",
    ] + if_google(["@com_google_absl//absl/types:source_location"]),
)

xla_cc_test(
    name = "emitter_loc_op_builder_test",
    srcs = ["emitter_loc_op_builder_test.cc"],
    tags = ["gpu"],
    deps = [
        ":emitter_loc_op_builder",
        "//xla/backends/gpu/codegen/triton:fusion_emitter",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "kernel_emitter",
    hdrs = ["kernel_emitter.h"],
    deps = [
        ":kernel_definition",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//xla/runtime:work_cluster",
        "//xla/runtime:work_group",
        "//xla/runtime:work_item",
        "//xla/service:buffer_assignment",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "llvm_ir_kernel_source",
    srcs = ["llvm_ir_kernel_source.cc"],
    hdrs = ["llvm_ir_kernel_source.h"],
    deps = [
        ":kernel_source",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:JITLink",
    ],
)

cc_library(
    name = "kernel_source",
    hdrs = ["kernel_source.h"],
)

cc_library(
    name = "kernel_definition",
    hdrs = ["kernel_definition.h"],
    deps = [
        ":kernel_source",
        ":kernel_spec",
        "//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "mlir_kernel_source",
    srcs = ["mlir_kernel_source.cc"],
    hdrs = ["mlir_kernel_source.h"],
    deps = [
        ":kernel_source",
        "//xla:util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_kernel_definition",
    hdrs = ["mlir_kernel_definition.h"],
    deps = [
        ":kernel_definition",
        ":kernel_source",
        ":kernel_spec",
        ":mlir_kernel_source",
    ],
)

cc_library(
    name = "mlir_kernel_emitter",
    hdrs = ["mlir_kernel_emitter.h"],
    deps = [
        ":kernel_emitter",
        ":mlir_kernel_definition",
    ],
)

cc_library(
    name = "llvm_kernel_emitter",
    hdrs = ["llvm_kernel_emitter.h"],
    deps = [
        ":kernel_emitter",
        ":llvm_kernel_definition",
    ],
)

cc_library(
    name = "llvm_kernel_definition",
    hdrs = ["llvm_kernel_definition.h"],
    deps = [
        ":kernel_definition",
        ":llvm_ir_kernel_source",
    ],
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)
