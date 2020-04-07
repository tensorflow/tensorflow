// RUN: tf-opt -test-buffer-assignment-converters -split-input-file %s | FileCheck %s -dump-input-on-failure

// CHECK-LABEL: func @func_signature_conversion
func @func_signature_conversion(%arg0: tensor<4x8xf32>) {
    return
}
// CHECK: ({{.*}}: memref<4x8xf32>) {

// -----

// CHECK-LABEL: func @non_void_to_void_return_op_converter
func @non_void_to_void_return_op_converter(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  return %arg0 : tensor<4x8xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]<[[RANK:.*]]>, %[[RESULT:.*]]: [[TYPE]]<[[RANK]]>) {
// CHECK-NEXT: "buffer_assignment_test.copy"(%[[ARG0]], %[[RESULT]])
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @func_and_block_signature_conversion
func @func_and_block_signature_conversion(%arg0 : tensor<2xf32>, %cond : i1, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32>{
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    br ^exit(%arg0 : tensor<2xf32>)
  ^exit(%arg2: tensor<2xf32>):
    return %arg1 : tensor<4x4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[ARG0_TYPE:.*]], %[[COND:.*]]: i1, %[[ARG1:.*]]: [[ARG1_TYPE:.*]], %[[RESULT:.*]]: [[RESULT_TYPE:.*]]) {
//      CHECK: br ^[[EXIT_BLOCK:.*]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: br ^[[EXIT_BLOCK]](%[[ARG0]] : [[ARG0_TYPE]])
//      CHECK: ^[[EXIT_BLOCK]](%{{.*}}: [[ARG0_TYPE]])
// CHECK-NEXT: "buffer_assignment_test.copy"(%[[ARG1]], %[[RESULT]])
// CHECK-NEXT: return
