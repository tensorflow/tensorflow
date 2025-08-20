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
    %0 = mhlo.after_all : !mhlo.token
    %1 = mhlo.constant dense<57202498903760> : tensor<i64>
    %2 = "mhlo.custom_call"(%1, %0, %arg1) {api_version = 2 : i32, call_target_name ="xla_python_cpu_callback", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<i64>, !mhlo.token, tensor<i64>) -> tuple<!mhlo.token>
    %3 = mhlo.get_tuple_element %2[0] : (tuple<!mhlo.token>) -> !mhlo.token
    %4 = mhlo.constant dense<> : tensor<0xi1>
    return %4 : tensor<0xi1>
  }
}

// -----

// Testing custom-call ops with buffer types and layouts.
// CHECK: HloModule
// CHECK: ENTRY
func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: %{{.*}} = b(f32[2,4]{0,1}) custom-call(%{{.*}}), custom_call_target="Pin"
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "Pin",
    api_version = 4 : i32
  } : (tensor<2x4xf32>) -> memref<2x4xf32, strided<[1, 2], offset:0>>
  // CHECK: %{{.*}} = b(f32[2,4]{0,1}) custom-call(%{{.*}}), custom_call_target="foo"
  %1 = "mhlo.custom_call"(%0) {
    call_target_name = "foo",
    api_version = 4 : i32,
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
        operand_index = 0,
        operand_tuple_indices = []>]
  } : (memref<2x4xf32, strided<[1, 2], offset:0>>) -> memref<2x4xf32, strided<[1, 2], offset:0>>
  // CHECK: %{{.*}} = f32[2,4]{1,0} custom-call(%{{.*}}), custom_call_target="Unpin"
  %2 = "mhlo.custom_call"(%1) {
    call_target_name = "Unpin",
    api_version = 4 : i32
  } : (memref<2x4xf32, strided<[1, 2], offset:0>>) -> tensor<2x4xf32>
  func.return %2 : tensor<2x4xf32>
}
