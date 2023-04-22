// RUN: tfr-opt %s -tfr-decompose -verify-diagnostics -split-input-file | FileCheck %s

tfr.func @tf__my_pack(%values: !tfr.tensor_list,
                      %n: i32 {tfr.name="N"},
                      %axis: i32 {tfr.name="axis"}) -> !tfr.tensor {
  %index = constant 0 : index
  %cst = constant 1 : i32
  %eq = cmpi eq, %n, %cst : i32
  %v1 = tfr.get_element %values[%index] : (!tfr.tensor_list, index) -> !tfr.tensor
  %temp = tfr.call @tf__expand_dims(%v1, %axis) : (!tfr.tensor, i32) -> !tfr.tensor
  %res = scf.if %eq -> !tfr.tensor {
    scf.yield %temp : !tfr.tensor
  } else {
    %step = index_cast %cst : i32 to index
    %end = index_cast %n : i32 to index
    %reduce = scf.for %i = %step to %end step %step iter_args(%reduce_iter=%temp) -> !tfr.tensor {
      %v = tfr.get_element %values[%i] : (!tfr.tensor_list, index) -> !tfr.tensor
      %temp1 =  tfr.call @tf__expand_dims(%v, %axis) : (!tfr.tensor, i32) -> !tfr.tensor
      %reduce_next =  tfr.call @tf__risc_concat(%reduce_iter, %temp1, %axis) : (!tfr.tensor, !tfr.tensor, i32) -> !tfr.tensor
      scf.yield %reduce_next : !tfr.tensor
    }
    scf.yield %reduce : !tfr.tensor
  }
  tfr.return %res : !tfr.tensor
}

// CHECK-LABEL: pack_one
func @pack_one(%arg0: tensor<2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = "tf.MyPack"(%arg0) {N=1:i32, axis=0:i32} : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>

// CHECK-NEXT: %[[AXIS:.*]] = constant 0 : i32
// CHECK-NEXT: %[[CAST:.*]] = "tfr.cast"(%arg0) : (tensor<2x3xf32>) -> !tfr.tensor
// CHECK-NEXT: %[[ED:.*]] = tfr.call @tf__expand_dims(%[[CAST]], %[[AXIS]]) : (!tfr.tensor, i32) -> !tfr.tensor
// CHECK-NEXT: %[[BACK:.*]] = "tfr.cast"(%[[ED]]) : (!tfr.tensor) -> tensor<1x2x3xf32>
// CHECK-NEXT: return %[[BACK]] : tensor<1x2x3xf32>
}

// CHECK-LABEL: pack_multiple
func @pack_multiple(%arg0: tensor<2x3xf32>,
                    %arg1: tensor<2x3xf32>,
                    %arg2: tensor<2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = "tf.MyPack"(%arg0, %arg1, %arg2) {N=3:i32, axis=0:i32} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>

// CHECK-NEXT: %[[AXIS:.*]] = constant 0 : i32
// CHECK-NEXT: %[[CAST0:.*]] = "tfr.cast"(%arg0) : (tensor<2x3xf32>) -> !tfr.tensor
// CHECK-NEXT: %[[CAST1:.*]] = "tfr.cast"(%arg1) : (tensor<2x3xf32>) -> !tfr.tensor
// CHECK-NEXT: %[[CAST2:.*]] = "tfr.cast"(%arg2) : (tensor<2x3xf32>) -> !tfr.tensor
// CHECK-NEXT: %[[EX0:.*]] = tfr.call @tf__expand_dims(%[[CAST0]], %[[AXIS]]) : (!tfr.tensor, i32) -> !tfr.tensor
// CHECK-NEXT: %[[EX1:.*]] = tfr.call @tf__expand_dims(%[[CAST1]], %[[AXIS]]) : (!tfr.tensor, i32) -> !tfr.tensor
// CHECK-NEXT: %[[CONCAT1:.*]] = tfr.call @tf__risc_concat(%[[EX0]], %[[EX1]], %c0_i32) : (!tfr.tensor, !tfr.tensor, i32) -> !tfr.tensor
// CHECK-NEXT: %[[EX2:.*]] = tfr.call @tf__expand_dims(%[[CAST2]], %[[AXIS]]) : (!tfr.tensor, i32) -> !tfr.tensor
// CHECK-NEXT: %[[CONCAT2:.*]] = tfr.call @tf__risc_concat(%[[CONCAT1]], %[[EX2]], %[[AXIS]]) : (!tfr.tensor, !tfr.tensor, i32) -> !tfr.tensor
// CHECK-NEXT: %[[BACK:.*]] = "tfr.cast"(%[[CONCAT2]]) : (!tfr.tensor) -> tensor<3x2x3xf32>
// CHECK-NEXT: return %[[BACK]] : tensor<3x2x3xf32>
}
