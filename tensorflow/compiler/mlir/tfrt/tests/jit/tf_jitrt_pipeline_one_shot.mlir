// RUN: tf-tfrt-opt -split-input-file -tf-jitrt-pipeline="one-shot-bufferize" %s --mlir-print-ir-after-all
// | FileCheck %s


// CHECK-LABEL: @tf_binary_with_bcast
func.func @tf_binary_with_bcast(%arg0: tensor<?x1xf32>,
                           %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK-NOT: shape.
  // CHECK: %[[LHS:.*]] = memref.reinterpret_cast
  // CHECK: %[[RHS:.*]] = memref.reinterpret_cast
  // CHECK: linalg.generic {{.*}} ins(%[[LHS]], %[[RHS]] :
  // CHECK:   mulf
  %0 = "tf.Mul"(%arg0, %arg1)
       : (tensor<?x1xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}

