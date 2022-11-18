// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text --print-layouts=true %s | FileCheck %s

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
  // CHECK:  ROOT
  // CHECK-SAME:  f32[1,2,3]{2,0,1} custom-call
  // CHECK-SAME:  operand_layout_constraints={f32[2,3]{0,1}, f32[5,5]{1,0}}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
    result_layouts = [dense<[2, 0, 1]> : tensor<3xindex>]
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
}

// -----

// CHECK:  HloModule
// CHECK: (token[]) custom-call(
module @jit_f {
  func.func public @main(%arg0: tensor<0xi1>, %arg1: tensor<i64>) -> tensor<0xi1> {
    %0 = mhlo.create_token : !mhlo.token
    %1 = mhlo.constant dense<57202498903760> : tensor<i64>
    %2 = "mhlo.custom_call"(%1, %0, %arg1) {api_version = 2 : i32, call_target_name ="xla_python_cpu_callback", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<i64>, !mhlo.token, tensor<i64>) -> tuple<!mhlo.token>
    %3 = mhlo.get_tuple_element %2[0] : (tuple<!mhlo.token>) -> !mhlo.token
    %4 = mhlo.constant dense<> : tensor<0xi1>
    return %4 : tensor<0xi1>
  }
}
