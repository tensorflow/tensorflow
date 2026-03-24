load("//xla:xla.default.bzl", "xla_cc_test")
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
    name = "device_spec",
    hdrs = ["device_spec.h"],
    deps = [
        "//xla/stream_executor:device_description",
        "//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "kernel_emitter",
    hdrs = ["kernel_emitter.h"],
    deps = [
        ":kernel_definition",
        ":kernel_source",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "ir_printing",
    srcs = ["ir_printing.cc"],
    hdrs = ["ir_printing.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:dump",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "kernel_spec",
    srcs = ["kernel_spec.cc"],
    hdrs = ["kernel_spec.h"],
    deps = [
        "//xla/runtime:work_cluster",
        "//xla/runtime:work_dimensions",
        "//xla/runtime:work_group",
        "//xla/runtime:work_item",
        "//xla/service:buffer_assignment",
        "//xla/service:shaped_slice",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "llvm_kernel_source",
    srcs = ["llvm_kernel_source.cc"],
    hdrs = ["llvm_kernel_source.h"],
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
    ],
)

cc_library(
    name = "mlir_kernel_source",
    srcs = ["mlir_kernel_source.cc"],
    hdrs = ["mlir_kernel_source.h"],
    deps = [
        ":kernel_source",
        "//xla:util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "ir_emission_utils",
    srcs = ["ir_emission_utils.cc"],
    hdrs = ["ir_emission_utils.h"],
    deps = [
        ":hlo_fusion_spec",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "ir_emission_utils_test",
    srcs = ["ir_emission_utils_test.cc"],
    deps = [
        ":ir_emission_utils",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "intrinsic_function",
    hdrs = ["intrinsic_function.h"],
    deps = [
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:type",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "intrinsic_lib",
    srcs = ["intrinsic_lib.cc"],
    hdrs = ["intrinsic_lib.h"],
    deps = [
        ":intrinsic_function",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:erf",
        "//xla/codegen/intrinsic:exp",
        "//xla/codegen/intrinsic:fptrunc",
        "//xla/codegen/intrinsic:ldexp",
        "//xla/codegen/intrinsic:log1p",
        "//xla/codegen/intrinsic:rsqrt",
        "//xla/codegen/intrinsic:string_interner",
        "//xla/codegen/intrinsic:tanh",
        "//xla/codegen/intrinsic:type",
        "//xla/codegen/intrinsic:vec_name_mangler",
        "//xla/codegen/intrinsic/cpp:_cpp_gen_intrinsics",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:JITLink",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "intrinsic_lib_test",
    srcs = ["intrinsic_lib_test.cc"],
    deps = [
        ":intrinsic_lib",
        "//xla/codegen/intrinsic",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Analysis",
    ],
)

cc_library(
    name = "hlo_fusion_spec",
    hdrs = ["hlo_fusion_spec.h"],
    deps = [
        "//xla/hlo/utils:hlo_traversal",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "trace_pass_instrumentation",
    srcs = ["trace_pass_instrumentation.cc"],
    hdrs = ["trace_pass_instrumentation.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)
