load("//xla:py_strict.bzl", "py_strict_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_test(
    name = "tiled_kernel_test",
    srcs = ["tiled_kernel_test.py"],
    main = "tiled_kernel_test.py",
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
    name = "tiled_fusion_emitter",
    srcs = ["tiled_fusion_emitter.cc"],
    hdrs = ["tiled_fusion_emitter.h"],
    visibility = [
        "//xla/backends/cpu/codegen:__pkg__",
        "//xla/service/cpu:__pkg__",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:kernel_api_ir_builder",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:mlir_kernel_source",
        "//xla/codegen/emitters:kernel_api_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/tiling:symbolic_tile_analysis",
        "//xla/codegen/tiling:tiled_hlo_computation",
        "//xla/codegen/tiling:tiled_hlo_instruction",
        "//xla/codegen/tiling:tiling_specification",
        "//xla/codegen/xtile/codegen:fusion_emitter",
        "//xla/codegen/xtile/codegen:tiled_emitter_constraints",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:work_dimensions",
        "//xla/service:buffer_assignment",
        "//xla/service:instruction_fusion",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)
