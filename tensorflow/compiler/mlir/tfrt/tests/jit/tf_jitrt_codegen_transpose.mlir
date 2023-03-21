// RUN: tf-tfrt-opt -tf-jitrt-pipeline="vectorize" -split-input-file %s | FileCheck %s

func.func @transpose_2d(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tf.Const"()
       {value = dense<[1, 0]> : tensor<2xi64>,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"}
       : () -> tensor<2xi64>
  %1 = "tf.Transpose"(%arg0, %0)
       {device = "/job:localhost/replica:0/task:0/device:CPU:0"}
       : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @transpose_2d
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// 8x8 tiling.
// CHECK:           scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// CHECK:             scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// Vector xfer reads: unrolled second vector dimension.
// CHECK-COUNT-8:       vector.transfer_read
// AVX2 shuffle/asm sequence.
// CHECK-COUNT-12:      vector.shuffle
// CHECK-COUNT-8:       llvm.inline_asm
// CHECK-COUNT-8:       vector.shuffle
// Vector xfer writes: unrolled second vector dimension.

// -----

func.func @transpose_3d_021(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Const"() { value = dense<[0, 2, 1]> : tensor<3xi64> }
    : () -> tensor<3xi64>
  %1 = "tf.Transpose"(%arg0, %0)
    : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL:   func @transpose_3d
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// 1x8x8 tiling.
// CHECK:           scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C1]] {
// CHECK:             scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// CHECK:               scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// Vector xfer reads: unrolled second vector dimension.
// CHECK-COUNT-8:         vector.transfer_read
// AVX2 shuffle/asm sequence.
// CHECK-COUNT-12:        vector.shuffle
// CHECK-COUNT-8:         llvm.inline_asm
// CHECK-COUNT-8:         vector.shuffle
// Vector xfer writes: unrolled second vector dimension.

// -----

func.func @transpose_3d_201(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Const"() { value = dense<[2, 0, 1]> : tensor<3xi64> }
    : () -> tensor<3xi64>
  %1 = "tf.Transpose"(%arg0, %0)
    : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL:   func @transpose_3d_201
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// 8x1x8 tiling.
// CHECK:           scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C1]] {
// CHECK:             scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// CHECK:               scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// Vector xfer reads: unrolled second vector dimension.
// CHECK-COUNT-8:         vector.transfer_read
// AVX2 shuffle/asm sequence.
// CHECK-COUNT-12:        vector.shuffle
// CHECK-COUNT-8:         llvm.inline_asm
// CHECK-COUNT-8:         vector.shuffle
// Vector xfer writes: unrolled second vector dimension.

// -----

func.func @transpose_3d_210(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Const"() { value = dense<[2, 1, 0]> : tensor<3xi64> }
    : () -> tensor<3xi64>
  %1 = "tf.Transpose"(%arg0, %0)
    : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL:   func @transpose_3d_210
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// 8x1x8 tiling.
// CHECK:           scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// CHECK:             scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C1]] {
// CHECK:               scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// Vector xfer reads: unrolled second vector dimension.
// CHECK-COUNT-8:         vector.transfer_read
// AVX2 shuffle/asm sequence.
// CHECK-COUNT-12:        vector.shuffle
// CHECK-COUNT-8:         llvm.inline_asm
// CHECK-COUNT-8:         vector.shuffle
// Vector xfer writes: unrolled second vector dimension.

// -----

func.func @transpose_3d_120(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Const"() { value = dense<[1, 2, 0]> : tensor<3xi64> }
    : () -> tensor<3xi64>
  %1 = "tf.Transpose"(%arg0, %0)
    : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL:   func @transpose_3d_120
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// 1x8x8 tiling.
// CHECK:           scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// CHECK:             scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C1]] {
// CHECK:               scf.for {{.*}} = %[[C0]] to {{.*}} step %[[C8]] {
// Vector xfer reads: unrolled second vector dimension.
// CHECK-COUNT-8:         vector.transfer_read
// AVX2 shuffle/asm sequence.
// CHECK-COUNT-12:        vector.shuffle
// CHECK-COUNT-8:         llvm.inline_asm
// CHECK-COUNT-8:         vector.shuffle
// Vector xfer writes: unrolled second vector dimension.
