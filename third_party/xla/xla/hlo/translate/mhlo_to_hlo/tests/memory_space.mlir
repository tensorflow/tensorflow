// RUN: hlo-translate -mlir-to-hlo -split-input-file -print-layouts %s | FileCheck %s

// Non-default memory spaces (e.g. S(1)/VMEM) carried on a custom-call by the
// mhlo.operand_memory_spaces / mhlo.result_memory_spaces attributes are
// restored onto the HLO custom-call's operand layout constraints and result
// layout when exporting StableHLO -> HLO. This goes through the full
// ConvertStablehloToHlo path, which runs the discardable-attribute sanitizer,
// so it also verifies the attributes are on the sanitizer allow-list.

// CHECK: HloModule
// CHECK: custom-call
// CHECK-SAME: operand_layout_constraints={f32[2,2]{1,0:S(1)}}
func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.custom_call @some_target(%arg0) {backend_config = "", mhlo.operand_memory_spaces = array<i64: 1>, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK: HloModule
// CHECK: f32[2,2]{1,0:S(1)} custom-call
// CHECK-SAME: operand_layout_constraints={f32[2,2]{1,0}}
func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.custom_call @some_target(%arg0) {backend_config = "", mhlo.result_memory_spaces = array<i64: 1>, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.negate %0 : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}

// -----

// CHECK: HloModule
// CHECK: (f32[2,2]{1,0:S(1)}, f32[2,2]{1,0}) custom-call
func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0:2 = stablehlo.custom_call @some_target(%arg0) {backend_config = "", mhlo.result_memory_spaces = array<i64: 1, 0>, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]} : (tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)
  %1 = stablehlo.add %0#0, %0#1 : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
