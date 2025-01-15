load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "emitter_loc_op_builder",
    srcs = ["emitter_loc_op_builder.cc"],
    hdrs = ["emitter_loc_op_builder.h"],
    visibility = [
        "//xla/backends/gpu/codegen/triton:__subpackages__",
        "//xla/service/gpu/fusions:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform",
    ] + if_google(["@com_google_absl//absl/types:source_location"]),
)

xla_test(
    name = "emitter_loc_op_builder_test",
    srcs = ["emitter_loc_op_builder_test.cc"],
    backends = ["gpu"],
    deps = [
        ":emitter_loc_op_builder",
        "//xla/backends/gpu/codegen/triton:fusion_emitter",
        "//xla/hlo/testlib:filecheck",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "fusion_emitter",
    srcs = ["fusion_emitter.cc"],
    hdrs = ["fusion_emitter.h"],
    visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla/backends/gpu/runtime:kernel_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:kernel_arguments",
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:target_util",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:ir_headers",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "fusions",
    srcs = ["fusions.cc"],
    hdrs = ["fusions.h"],
    visibility = [
        "//xla/backends/gpu/codegen/triton:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    deps = [
        ":fusion_emitter",
        "//xla:shape_util",
        "//xla/backends/gpu/codegen:copy",
        "//xla/backends/gpu/codegen:cudnn",
        "//xla/backends/gpu/codegen:custom",
        "//xla/backends/gpu/codegen/emitters:concatenate",
        "//xla/backends/gpu/codegen/emitters:in_place_dynamic_update_slice",
        "//xla/backends/gpu/codegen/emitters:input_slices",
        "//xla/backends/gpu/codegen/emitters:loop",
        "//xla/backends/gpu/codegen/emitters:reduction",
        "//xla/backends/gpu/codegen/emitters:scatter",
        "//xla/backends/gpu/codegen/emitters:transpose",
        "//xla/backends/gpu/codegen/triton:fusion",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
