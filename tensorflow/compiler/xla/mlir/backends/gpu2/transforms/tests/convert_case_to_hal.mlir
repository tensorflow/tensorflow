// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-convert-to-runtime --split-input-file       \
// RUN:   | FileCheck %s

// -----
// lmhlo.case operation with 2 branches converted to scf.if.

func.func @main(%arg0: memref<4xi8>, %arg1: memref<4xi8>, %arg2: memref<4xi8>) {
  %c0 = arith.constant 0 : index

  %buffer = memref.view %arg0[%c0][] : memref<4xi8> to memref<1xf32>
  %cst = memref.view %arg1[%c0][] : memref<4xi8> to memref<1xf32>
  %case = memref.view %arg2[%c0][] : memref<4xi8> to memref<i32>

  "lmhlo.case"(%case) ({
    "lmhlo.fusion"() ({
      %1 = bufferization.to_tensor %buffer : memref<1xf32>
      %2 = bufferization.to_tensor %cst : memref<1xf32>
      %3 = mhlo.multiply %1, %2 : tensor<1xf32>
      memref.tensor_store %3, %buffer : memref<1xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    "lmhlo.terminator"() : () -> ()
  }, {
    "lmhlo.fusion"() ({
      %1 = bufferization.to_tensor %buffer : memref<1xf32>
      %2 = bufferization.to_tensor %cst : memref<1xf32>
      %3 = mhlo.add %1, %2 : tensor<1xf32>
      memref.tensor_store %3, %buffer : memref<1xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) : (memref<i32>) -> ()

  "lmhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @main(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<4xi8>, %[[ARG1:.*]]: tensor<4xi8>,
// CHECK:   %[[ARG2:.*]]: tensor<4xi8>
// CHECK: ) {

// CHECK-DAG: %[[BUFFER0:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK-DAG: %[[BUFFER1:.*]] = iree_input.tensor.export %[[ARG1]]
// CHECK-DAG: %[[BUFFER2:.*]] = iree_input.tensor.export %[[ARG2]]

// CHECK-DAG: %[[TENSOR:.*]] = iree_input.tensor.import %[[BUFFER0]]
// CHECK-DAG: %[[CST:.*]] = iree_input.tensor.import %[[BUFFER1]]
// CHECK-DAG: %[[CASE:.*]] = iree_input.tensor.import %[[BUFFER2]]

// CHECK:   %[[INDEX:.*]] = iree_input.tensor.load %[[CASE]]

// CHECK:   %[[IF:.*]] = scf.if {{.*}} -> (tensor<1xf32>) {
// CHECK:     %[[RES0:.*]] = iree_input.dispatch {{.*}} -> %[[TENSOR]]
// CHECK:     scf.yield %[[RES0]] : tensor<1xf32>
// CHECK:   } else {
// CHECK:     %[[RES1:.*]] = iree_input.dispatch {{.*}} -> %[[TENSOR]]
// CHECK:     scf.yield %[[RES1]] : tensor<1xf32>
// CHECK:   }
// CHECK: }

// -----
// lmhlo.case operation with 3 branches converted to scf.index_switch.

func.func @main(%arg0: memref<4xi8>, %arg1: memref<4xi8>, %arg2: memref<4xi8>) {
  %c0 = arith.constant 0 : index

  %buffer = memref.view %arg0[%c0][] : memref<4xi8> to memref<1xf32>
  %cst = memref.view %arg1[%c0][] : memref<4xi8> to memref<1xf32>
  %case = memref.view %arg2[%c0][] : memref<4xi8> to memref<i32>

  "lmhlo.case"(%case) ({
    "lmhlo.fusion"() ({
      %1 = bufferization.to_tensor %buffer : memref<1xf32>
      %2 = bufferization.to_tensor %cst : memref<1xf32>
      %3 = mhlo.multiply %1, %2 : tensor<1xf32>
      memref.tensor_store %3, %buffer : memref<1xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    "lmhlo.terminator"() : () -> ()
  }, {
    "lmhlo.fusion"() ({
      %1 = bufferization.to_tensor %buffer : memref<1xf32>
      %2 = bufferization.to_tensor %cst : memref<1xf32>
      %3 = mhlo.add %1, %2 : tensor<1xf32>
      memref.tensor_store %3, %buffer : memref<1xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    "lmhlo.terminator"() : () -> ()
  }, {
    "lmhlo.fusion"() ({
      %1 = bufferization.to_tensor %buffer : memref<1xf32>
      %2 = bufferization.to_tensor %cst : memref<1xf32>
      %3 = mhlo.subtract %1, %2 : tensor<1xf32>
      memref.tensor_store %3, %buffer : memref<1xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) : (memref<i32>) -> ()

  "lmhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @main(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<4xi8>, %[[ARG1:.*]]: tensor<4xi8>,
// CHECK:   %[[ARG2:.*]]: tensor<4xi8>
// CHECK: ) {

// CHECK-DAG: %[[BUFFER0:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK-DAG: %[[BUFFER1:.*]] = iree_input.tensor.export %[[ARG1]]
// CHECK-DAG: %[[BUFFER2:.*]] = iree_input.tensor.export %[[ARG2]]

// CHECK-DAG: %[[TENSOR:.*]] = iree_input.tensor.import %[[BUFFER0]]
// CHECK-DAG: %[[CST:.*]] = iree_input.tensor.import %[[BUFFER1]]
// CHECK-DAG: %[[CASE:.*]] = iree_input.tensor.import %[[BUFFER2]]

// CHECK:   %[[INDEX:.*]] = iree_input.tensor.load %[[CASE]]

// CHECK:   %[[SWITCH:.*]] = scf.index_switch {{.*}} -> tensor<1xf32>
// CHECK:   case 0 {
// CHECK:     %[[RES0:.*]] = iree_input.dispatch {{.*}} -> %[[TENSOR]]
// CHECK:     scf.yield %[[RES0]] : tensor<1xf32>
// CHECK:   }
// CHECK:   case 1 {
// CHECK:     %[[RES1:.*]] = iree_input.dispatch {{.*}} -> %[[TENSOR]]
// CHECK:     scf.yield %[[RES1]] : tensor<1xf32>
// CHECK:   }
// CHECK:   case 2 {
// CHECK:     %[[RES2:.*]] = iree_input.dispatch {{.*}} -> %[[TENSOR]]
// CHECK:     scf.yield %[[RES2]] : tensor<1xf32>
// CHECK:   }
// CHECK:   default {
// CHECK:     scf.yield %[[TENSOR]] : tensor<1xf32>
// CHECK:   }
// CHECK: }
