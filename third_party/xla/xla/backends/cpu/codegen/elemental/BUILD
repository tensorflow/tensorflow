load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/backends/cpu:xla_backend_cpu_internal_access"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "concatenate_kernel_emitter",
    srcs = ["concatenate_kernel_emitter.cc"],
    hdrs = ["concatenate_kernel_emitter.h"],
    deps = [
        ":elemental_kernel_emitter",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/cpu/codegen:kernel_api_ir_builder",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:ir_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ir_headers",
    ],
)

py_strict_test(
    name = "concatenate_kernel_emitter_test",
    srcs = ["concatenate_kernel_emitter_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "elemental_kernel_emitter",
    srcs = ["elemental_kernel_emitter.cc"],
    hdrs = ["elemental_kernel_emitter.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/cpu/codegen:kernel_api_ir_builder",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:elemental_ir_emitter",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:elemental_ir_emitter",
        "//xla/service/cpu:ir_emitter",
        "//xla/service/cpu:parallel_loop_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/service/llvm_ir:loop_emitter",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "elemental_kernel_emitter_test",
    srcs = ["elemental_kernel_emitter_test.cc"],
    deps = [
        ":elemental_kernel_emitter",
        "//xla:xla_data_proto_cc",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_spec",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:buffer_assignment",
        "//xla/service:logical_buffer",
        "//xla/service/cpu:target_machine_features_stub",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ir_headers",
    ],
)
