// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text --print-layouts=true %s | FileCheck %s

// CHECK:  HloModule
func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
  // CHECK:  ROOT
  // CHECK-SAME:  f32[1,2,3]{2,0,1} custom-call
  // CHECK-SAME:  operand_layout_constraints={f32[2,3]{0,1}, f32[5,5]{1,0}}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
    result_layouts = [dense<[2, 0, 1]> : tensor<3xindex>]
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}
