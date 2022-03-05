// RUN: tf-tfrt-opt -tf-jitrt-pipeline="vectorize codegen-transpose" -split-input-file %s | FileCheck %s

// Verify that transpose codegen is working within the pipeline.

func @transpose_2d(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Const"()
       {value = dense<[1, 0]> : tensor<2xi64>,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"}
       : () -> tensor<2xi64>
  %1 = "tf.Transpose"(%arg0, %0)
       {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
       : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @transpose_2d
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// 8x8 tiling.
// CHECK:           scf.parallel {{.*}} step (%[[C8]], %[[C8]]) {
// Vector xfer reads: unrolled second vector dimension.
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_read
// AVX2 shuffle/asm sequence.
// CHECK-COUNT-12:        vector.shuffle
// CHECK-COUNT-8:        llvm.inline_asm
// CHECK-COUNT-8:        vector.shuffle
// Vector xfer write: unrolled second vector dimension.
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write
// CHECK-NEXT:        affine.apply
// CHECK-NEXT:        vector.transfer_write

