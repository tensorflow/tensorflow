load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:pytype.bzl", "pytype_strict_library")
load("//xla:xla.default.bzl", "xla_cc_test")
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
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:cpu_features",
        "//xla/backends/cpu/codegen:execution_engine",
        "//xla/backends/cpu/codegen:ir_compiler",
        "//xla/backends/cpu/codegen:jit_compiler",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/backends/cpu/runtime:kernel",
        "//xla/backends/cpu/runtime:kernel_c_api",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/codegen/testlib:kernel_runner",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_options",
        "//xla/service/cpu:runtime_symbol_generator",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:ir_headers",
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
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:jit_compiler",
        "//xla/codegen:kernel_definition",
        "//xla/codegen/testlib:kernel_runner",
        "//xla/hlo/testlib:test",
        "//xla/runtime:buffer_use",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "llvm_ir_kernel_emitter",
    testonly = 1,  # TODO(willfroom): Move to runtime(?) & plug into ir_emitter2 once the interface is stable.
    srcs = ["llvm_ir_kernel_emitter.cc"],
    hdrs = ["llvm_ir_kernel_emitter.h"],
    deps = [
        "//xla:util",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

tsl_pybind_extension(
    name = "_extension",
    testonly = 1,
    srcs = ["kernel_runner_extension.cc"],
    visibility = ["//visibility:private"],  # the extension should always be linked via testlib
    deps = [
        ":kernel_runner",
        ":llvm_ir_kernel_emitter",
        # placeholder for index annotation deps  # buildcleaner: keep
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla/backends/cpu/codegen:computation_kernel_emitter",
        "//xla/backends/cpu/codegen:jit_compiler",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/backends/cpu/codegen/dot:dot_kernel_emitter",
        "//xla/backends/cpu/codegen/elemental:concatenate_kernel_emitter",
        "//xla/backends/cpu/codegen/elemental:elemental_kernel_emitter",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen/testlib:kernel_runner",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_compiler_pure",
        "//xla/stream_executor:launch_dim",
    ],
)

pytype_strict_library(
    name = "testlib",
    testonly = 1,
    srcs = [
        "__init__.py",
        "utilities.py",
    ],
    deps = [
        ":_extension",
        "//xla/codegen/testlib",  # buildcleaner: keep
    ],
)

xla_cc_test(
    name = "llvm_ir_kernel_emitter_test",
    srcs = ["llvm_ir_kernel_emitter_test.cc"],
    deps = [
        ":llvm_ir_kernel_emitter",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/runtime:buffer_use",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:casts",
    ],
)

py_strict_test(
    name = "kernel_runner_test_py",
    srcs = ["kernel_runner_test.py"],
    main = "kernel_runner_test.py",
    tags = [
        "no_oss",
    ],
    deps = [
        ":testlib",
        "//third_party/py/numpy",
        "//xla/codegen/testlib",
        "@absl_py//absl/testing:absltest",
    ],
)

py_strict_test(
    name = "elemental_kernel_emitter_test",
    srcs = ["elemental_kernel_emitter_test.py"],
    main = "elemental_kernel_emitter_test.py",
    tags = [
        "no_oss",
    ],
    deps = [
        ":testlib",
        "//third_party/py/numpy",
        "//xla/codegen/testlib",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
