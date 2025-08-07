load("//xla:py_strict.bzl", "py_strict_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/backends/cpu:xla_backend_cpu_internal_access"],
    licenses = ["notice"],
)

cc_library(
    name = "dot_kernel_emitter",
    srcs = ["dot_kernel_emitter.cc"],
    hdrs = ["dot_kernel_emitter.h"],
    deps = [
        "//xla:util",
        "//xla/backends/cpu/codegen:kernel_api_ir_builder",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/codegen:llvm_kernel_definition",
        "//xla/codegen:llvm_kernel_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:work_group",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:dot_op_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ir_headers",
    ],
)

py_strict_test(
    name = "dot_kernel_emitter_test",
    srcs = ["dot_kernel_emitter_test.py"],
    tags = ["no_oss"],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
