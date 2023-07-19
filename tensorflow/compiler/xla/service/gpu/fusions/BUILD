cc_library(
    name = "in_place_dynamic_update_slice",
    srcs = ["in_place_dynamic_update_slice.cc"],
    hdrs = ["in_place_dynamic_update_slice.h"],
    deps = [
        ":fusion_emitter",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/llvm_ir:dynamic_update_slice_util",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "copy",
    srcs = ["copy.cc"],
    hdrs = ["copy.h"],
    deps = [
        ":fusion_emitter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter_context",
    ],
)

cc_library(
    name = "fusion_emitter",
    srcs = ["fusion_emitter.cc"],
    hdrs = ["fusion_emitter.h"],
    visibility = ["//tensorflow/compiler/xla/service/gpu:__subpackages__"],
    deps = [
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable",
        "//tensorflow/compiler/xla/service/gpu:hlo_to_ir_bindings",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter_context",
        "//tensorflow/compiler/xla/service/gpu:kernel_reuse_cache",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:target_util",
        "//tensorflow/compiler/xla/service/gpu:thunk",
        "//tensorflow/compiler/xla/service/llvm_ir:buffer_assignment_util",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
    ],
)

cc_library(
    name = "fusions",
    srcs = ["fusions.cc"],
    hdrs = ["fusions.h"],
    visibility = ["//tensorflow/compiler/xla/service/gpu:__subpackages__"],
    deps = [
        ":copy",
        ":fusion_emitter",
        ":in_place_dynamic_update_slice",
        ":loop",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service/gpu:hlo_fusion_analysis",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter_context",
    ],
)

cc_library(
    name = "loop",
    srcs = ["loop.cc"],
    hdrs = ["loop.h"],
    deps = [
        ":fusion_emitter",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service:elemental_ir_emitter",
        "//tensorflow/compiler/xla/service/gpu:hlo_fusion_analysis",
        "//tensorflow/compiler/xla/service/gpu:ir_emitter_context",
        "//tensorflow/compiler/xla/service/gpu:parallel_loop_emitter",
        "//tensorflow/compiler/xla/service/llvm_ir:fused_ir_emitter",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "tiling_util",
    srcs = ["tiling_util.cc"],
    hdrs = ["tiling_util.h"],
    visibility = ["//tensorflow/compiler/xla/service/gpu:__subpackages__"],
    deps = [
        "//tensorflow/compiler/xla/service/gpu:hlo_fusion_analysis",
        "//tensorflow/compiler/xla/service/gpu:ir_emission_utils",
        "//tensorflow/compiler/xla/service/gpu:target_util",
        "//tensorflow/compiler/xla/service/llvm_ir:ir_array",
        "//tensorflow/compiler/xla/service/llvm_ir:kernel_support_library",
        "@llvm-project//llvm:ir_headers",
    ],
)
