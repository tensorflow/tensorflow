load("//xla:pytype.default.bzl", "pytype_strict_library")
load("//xla:strict.default.bzl", "py_strict_test")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
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
    testonly = 1,
    srcs = ["llvm_ir_kernel_spec.cc"],
    hdrs = ["llvm_ir_kernel_spec.h"],
    deps = [
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
    ],
)

tsl_pybind_extension(
    name = "kernel_runner_extention",
    testonly = 1,
    srcs = ["kernel_runner_extention.cc"],
    visibility = ["//visibility:private"],  # the extention should always be linked via kernel_runner_pylib
    deps = [
        ":kernel_runner",
        ":llvm_ir_kernel_emitter",
        ":llvm_ir_kernel_spec",
        # placeholder for index annotation deps
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/codegen:kernel_spec",
        "//xla/stream_executor:launch_dim",
    ],
)

pytype_strict_library(
    name = "kernel_runner_pylib",
    testonly = 1,
    srcs = ["kernel_runner.py"],
    srcs_version = "PY3",
    deps = [
        ":kernel_runner_extention",
        "//xla/codegen/testlib:kernel_runner_pylib",
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

py_strict_test(
    name = "kernel_runner_test_py",
    srcs = ["kernel_runner_test.py"],
    main = "kernel_runner_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss",
    ],
    deps = [
        ":kernel_runner_pylib",
        "//third_party/py/numpy",
        "//xla/codegen/testlib:kernel_runner_pylib",
        "@absl_py//absl/testing:absltest",
    ],
)
