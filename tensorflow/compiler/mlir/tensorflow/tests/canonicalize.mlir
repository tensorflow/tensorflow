// RUN: tf-opt %s -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @tfAssertTrue
func.func @tfAssertTrue(%arg0: tensor<1x1x6x2xf32>) {
  %t = arith.constant dense<true> : tensor<i1>
  // CHECK-NOT: tf.Assert
  "tf.Assert"(%t, %arg0) {summarize = 3} : (tensor<i1>, tensor<1x1x6x2xf32>) -> ()
  func.return
}

// CHECK-LABEL: func @tfAssertFalse
func.func @tfAssertFalse(%arg0: tensor<1x1x6x2xf32>) {
  %f = arith.constant dense<false> : tensor<i1>
  // CHECK: tf.Assert
  "tf.Assert"(%f, %arg0) {summarize = 3} : (tensor<i1>, tensor<1x1x6x2xf32>) -> ()
  func.return
}

// CHECK-LABEL: testGatherToV2
// Ensures that axis param and batch_dims attr use their default values of 0.
func.func @testGatherToV2(%params: tensor<4x3xf32>, %indices: tensor<1x2xi32>) -> tensor<2x3xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "tf.GatherV2"(%arg0, %arg1, %[[AXIS]]) {batch_dims = 0 : i64} : (tensor<4x3xf32>, tensor<1x2xi32>, tensor<i32>) -> tensor<2x3xf32>
  %0 = "tf.Gather"(%params, %indices) : (tensor<4x3xf32>, tensor<1x2xi32>) -> tensor<2x3xf32>
  func.return %0: tensor<2x3xf32>
}

// CHECK-LABEL: testBatchMatMulToV2
func.func @testBatchMatMulToV2(%arg0: tensor<2x3x5xf32>, %arg1: tensor<2x5x7xf32>) -> tensor<2x3x7xf32> {
  // CHECK: tf.BatchMatMulV2
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<2x3x5xf32>, tensor<2x5x7xf32>) -> tensor<2x3x7xf32>
  func.return %0: tensor<2x3x7xf32>
}

// CHECK-LABEL: testDynamicBatchMatMulToV2
func.func @testDynamicBatchMatMulToV2(%arg0: tensor<2x3x5xf32>, %arg1: tensor<?x5x7xf32>) -> tensor<2x3x7xf32> {
  // CHECK: tf.BatchMatMul
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<2x3x5xf32>, tensor<?x5x7xf32>) -> tensor<2x3x7xf32>
  func.return %0: tensor<2x3x7xf32>
}

// CHECK-LABEL: testBatchMatMulToMatMul
func.func @testBatchMatMulToMatMul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  func.return %0: tensor<2x2xf32>

// CHECK: %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
// CHECK: return %0
}

// CHECK-LABEL: testBatchMatMulV2ToMatMul
func.func @testBatchMatMulV2ToMatMul(%arg0: tensor<4x3xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = false} : (tensor<4x3xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  func.return %0: tensor<3x5xf32>

// CHECK: %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = true, transpose_b = false} : (tensor<4x3xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
// CHECK: return %0
}

// CHECK-LABEL: testBiasAddV1ToBiasAdd
func.func @testBiasAddV1ToBiasAdd(%arg0: tensor<*xf32>, %arg1: tensor<128xf32>) -> tensor<*xf32> {
  // CHECK: "tf.BiasAdd"(%arg0, %arg1) {data_format = "NHWC"} : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  %0 = "tf.BiasAddV1"(%arg0, %arg1) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>
}

// CHECK-LABEL: func @testLeakyRelu
func.func @testLeakyRelu(%arg0 : tensor<16xf32>) -> (tensor<16xf32>) {
  %2 = "tf.LeakyRelu"(%arg0) {alpha = 1.0 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: return %arg0
  func.return %2 : tensor<16xf32>
}

// CHECK-LABEL: testSameBitcastType
func.func @testSameBitcastType(%arg0: tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  func.return %0: tensor<8x16x32x64xf32>

// CHECK: return %arg0
}

// CHECK-LABEL: testDifferentBitcastType
func.func @testDifferentBitcastType(%arg0: tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
  func.return %0: tensor<8x16x32x64xi32>

// CHECK: %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: return %0
}

// CHECK-LABEL: testDoubleBitcast
func.func @testDoubleBitcast(%arg0: tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64x2xi16>
  %1 = "tf.Bitcast"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64xi32>
  func.return %1: tensor<8x16x32x64xi32>

// CHECK: %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: return %0
}

// CHECK-LABEL: testDoubleBitcastWithDependentArg
func.func @testDoubleBitcastWithDependentArg(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x32x64xi32>, tensor<8x16x32x64x2xi16>) {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64x2xi16>
  %1 = "tf.Bitcast"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64xi32>
  %2 = "tf.Identity"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64x2xi16>
  func.return %1, %2 :  tensor<8x16x32x64xi32>, tensor<8x16x32x64x2xi16>

// CHECK: %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64x2xi16>
// CHECK: %1 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: %2 = "tf.Identity"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64x2xi16>
// CHECK: return %1, %2
}

// CHECK-LABEL: testSameCastType
func.func @testSameCastType(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x32x64xf32>, tensor<8x16x32x64xf32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  func.return %0, %1: tensor<8x16x32x64xf32>, tensor<8x16x32x64xf32>

// CHECK: return %arg0, %arg0
}

// CHECK-LABEL: testDifferentCastType
func.func @testDifferentCastType(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x32x64xi32>, tensor<8x16x32x64xi32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
  func.return %0, %1: tensor<8x16x32x64xi32>, tensor<8x16x32x64xi32>

// CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: return %0, %1
}

// CHECK-LABEL: testCompatibleCastType
func.func @testCompatibleCastType(%arg0: tensor<?xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<?xf32>) -> tensor<10xf32>
  %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<?xf32>) -> tensor<10xf32>
  func.return %0, %1: tensor<10xf32>, tensor<10xf32>

// CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<?xf32>) -> tensor<10xf32>
// CHECK: %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<?xf32>) -> tensor<10xf32>
// CHECK: return %0, %1
}

// CHECK-LABEL: testSameCastTypeAcrossBasicBlocks
func.func @testSameCastTypeAcrossBasicBlocks(tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32> {
^bb0(%arg0: tensor<8x16x32x64xf32>):
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  cf.br ^bb1
^bb1:
  %1 = "tf.Cast"(%0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  cf.br ^exit
^exit:
  func.return %1: tensor<8x16x32x64xf32>

// CHECK: return %arg0
}

// CHECK-LABEL: testConcatCanonicalization
func.func @testConcatCanonicalization(%arg0: tensor<2x1xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x2xi32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"
  %0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>

  // CHECK: "tf.ConcatV2"(%arg0, %arg1, %[[AXIS]])
  %1 = "tf.Concat"(%0, %arg0, %arg1) : (tensor<i32>, tensor<2x1xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>
  func.return %1 : tensor<2x2xi32>
}

// CHECK-LABEL: testConcatCwiseUnary
func.func @testConcatCwiseUnary(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1xf32>, %arg2: tensor<i32>) -> tensor<?x2xf32> {

  // CHECK: %[[CONCAT:.*]] = "tf.ConcatV2"(%arg0, %arg1, %arg2)
  // CHECK: %[[LOG1P:.*]] = "tf.Log1p"(%[[CONCAT]])
  // CHECK: return %[[LOG1P]]
  %0 = "tf.Log1p"(%arg0) : (tensor<?x1xf32>) -> tensor<?x1xf32>
  %1 = "tf.Log1p"(%arg1) : (tensor<?x1xf32>) -> tensor<?x1xf32>
  %2 = "tf.ConcatV2"(%0, %1, %arg2) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<i32>) -> tensor<?x2xf32>

  func.return %2 : tensor<?x2xf32>
}

// CHECK-LABEL: testConcatCwiseBinaryOnInnerDim
func.func @testConcatCwiseBinaryOnInnerDim(%arg0: tensor<?x1xf32>,
  %arg1: tensor<?x1xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<?x2xf32> {

  // CHECK-DAG: %[[LHS_AXIS:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>}

  // CHECK: %[[ADD_LHS_CONCAT:.*]] = "tf.Pack"(%arg2, %arg3) {axis = 0 : i64}
  // CHECK: %[[MUL_LHS_CONCAT:.*]] = "tf.ConcatV2"(%arg0, %arg1, %[[LHS_AXIS]])
  // CHECK: %[[MUL_RHS_CONCAT:.*]] = "tf.Pack"(%arg2, %arg3) {axis = 0 : i64}

  // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[MUL_LHS_CONCAT]], %[[MUL_RHS_CONCAT]])
  // CHECK-SAME: (tensor<?x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[ADD_LHS_CONCAT]], %[[MUL]])
  // CHECK-SAME: (tensor<2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  // CHECK: return %[[ADD]]

  %0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  // Mul of a tensor and a scalar const.
  %1 = "tf.Mul"(%arg0, %arg2) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %2 = "tf.Mul"(%arg1, %arg3) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  // Add of a scalar const and a tensor.
  %3 = "tf.AddV2"(%arg2, %1) : (tensor<f32>, tensor<?x1xf32>) -> tensor<?x1xf32>
  %4 = "tf.AddV2"(%arg3, %2) : (tensor<f32>, tensor<?x1xf32>) -> tensor<?x1xf32>
  %5 = "tf.ConcatV2"(%3, %4, %0) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<i32>) -> tensor<?x2xf32>

  func.return %5 : tensor<?x2xf32>
}

// CHECK-LABEL: testConcatCwiseBinaryPreserveAxisType
func.func @testConcatCwiseBinaryPreserveAxisType(%arg0: tensor<?x1xf32>,
  %arg1: tensor<?x1xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<?x2xf32> {

  // CHECK-DAG: %[[LHS_AXIS:.*]] = "tf.Const"() {value = dense<1> : tensor<i64>}

  // CHECK: %[[ADD_LHS_CONCAT:.*]] = "tf.Pack"(%arg2, %arg3) {axis = 0 : i64}
  // CHECK: %[[MUL_LHS_CONCAT:.*]] = "tf.ConcatV2"(%arg0, %arg1, %[[LHS_AXIS]])
  // CHECK: %[[MUL_RHS_CONCAT:.*]] = "tf.Pack"(%arg2, %arg3) {axis = 0 : i64}

  // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[MUL_LHS_CONCAT]], %[[MUL_RHS_CONCAT]])
  // CHECK-SAME: (tensor<?x2xf32>, tensor<2xf32>) -> tensor<?x2xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[ADD_LHS_CONCAT]], %[[MUL]])
  // CHECK-SAME: (tensor<2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  // CHECK: return %[[ADD]]

  %0 = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
  // Mul of a tensor and a scalar const.
  %1 = "tf.Mul"(%arg0, %arg2) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %2 = "tf.Mul"(%arg1, %arg3) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  // Add of a scalar const and a tensor.
  %3 = "tf.AddV2"(%arg2, %1) : (tensor<f32>, tensor<?x1xf32>) -> tensor<?x1xf32>
  %4 = "tf.AddV2"(%arg3, %2) : (tensor<f32>, tensor<?x1xf32>) -> tensor<?x1xf32>
  %5 = "tf.ConcatV2"(%3, %4, %0) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<i64>) -> tensor<?x2xf32>

  func.return %5 : tensor<?x2xf32>
}

// CHECK-LABEL: testConcatCwiseBinaryInvalidInnerDim
func.func @testConcatCwiseBinaryInvalidInnerDim(%arg0: tensor<?x2xf32>,
  %arg1: tensor<?x2xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<?x4xf32> {
  // Each individual binary operation has an implicit broadcast that will be
  // lost if we would reorder them with the concat.

  // CHECK: %[[CONST:.*]] = "tf.Const"()
  // CHECK-DAG: %[[MUL1:.*]] = "tf.Mul"(%arg0, %arg2)
  // CHECK-DAG: %[[MUL2:.*]] = "tf.Mul"(%arg1, %arg3)
  // CHECK: "tf.ConcatV2"(%[[MUL1]], %[[MUL2]], %[[CONST]])
  %0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Mul"(%arg0, %arg2) : (tensor<?x2xf32>, tensor<f32>) -> tensor<?x2xf32>
  %2 = "tf.Mul"(%arg1, %arg3) : (tensor<?x2xf32>, tensor<f32>) -> tensor<?x2xf32>
  %3 = "tf.ConcatV2"(%1, %2, %0) : (tensor<?x2xf32>, tensor<?x2xf32>, tensor<i32>) -> tensor<?x4xf32>

  func.return %3 : tensor<?x4xf32>
}

// CHECK-LABEL: testConcatCwiseBinaryInvalidExceptions
func.func @testConcatCwiseBinaryInvalidExceptions(%arg0: tensor<?x1xf32>,
  %arg1: tensor<?x1xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<?x2xf32>) -> tensor<?x4xf32> {
  // Each individual binary operation has an implicit broadcast that will be
  // lost if we would reorder them with the concat.

  // CHECK: %[[CONST:.*]] = "tf.Const"()
  // CHECK-DAG: %[[MUL1:.*]] = "tf.Mul"(%arg0, %arg2)
  // CHECK-DAG: %[[MUL2:.*]] = "tf.Mul"(%arg1, %arg3)
  // CHECK: "tf.ConcatV2"(%[[MUL1]], %[[MUL2]],
  %0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Mul"(%arg0, %arg2) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %2 = "tf.Mul"(%arg1, %arg3) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %3 = "tf.ConcatV2"(%1, %2, %arg4, %0) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x2xf32>, tensor<i32>) -> tensor<?x4xf32>

  func.return %3 : tensor<?x4xf32>
}


// CHECK-LABEL: testConcatCwiseBinaryNegativeAxis
func.func @testConcatCwiseBinaryNegativeAxis(%arg0: tensor<f32>,
  %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<2xf32> {
  // The test should not crash with negative axis.
  %0 = "tf.Const"() { value = dense<-1> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Mul"(%arg0, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %2 = "tf.Mul"(%arg1, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %3 = "tf.ConcatV2"(%1, %2, %0) : (tensor<f32>, tensor<f32>, tensor<i32>) -> tensor<2xf32>

  func.return %3 : tensor<2xf32>
}

// Synthesize binary ops when 1 of the 3 concat inputs is a non-binary op.
// CHECK-LABEL: testConcatCwiseBinarySynthMulOp3Inputs
func.func @testConcatCwiseBinarySynthMulOp3Inputs(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1xf32>, %arg2: tensor<?x1xf32>) -> tensor<?x3xf32> {
  // CHECK: %[[CONST:.*]] = "tf.Const"() {value = dense<[2.000000e+00, 3.000000e+00, 1.000000e+00]>
  // CHECK: %[[CONCAT:.*]] = "tf.ConcatV2"(%arg0, %arg1, %arg2,
  // CHECK: "tf.Mul"(%[[CONCAT]], %[[CONST]])
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %mul0_const = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %mul0 = "tf.Mul"(%arg0, %mul0_const) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %mul1_const = "tf.Const"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
  %mul1 = "tf.Mul"(%arg1, %mul1_const) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %ret = "tf.ConcatV2"(%mul0, %mul1, %arg2, %axis) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<i32>) -> tensor<?x3xf32>

  func.return %ret : tensor<?x3xf32>
}

// Similar to the above, with tf.Sub as the binary op kind.
func.func @testConcatCwiseBinarySynthSubOp3Inputs(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1xf32>, %arg2: tensor<?x1xf32>) -> tensor<?x3xf32> {
  // CHECK: %[[CONST:.*]] = "tf.Const"() {value = dense<[2.000000e+00, 3.000000e+00, 0.000000e+00]>
  // CHECK: %[[CONCAT:.*]] = "tf.ConcatV2"(%arg0, %arg1, %arg2,
  // CHECK: "tf.Sub"(%[[CONCAT]], %[[CONST]])
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %mul0_const = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %mul0 = "tf.Sub"(%arg0, %mul0_const) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %mul1_const = "tf.Const"() { value = dense<3.0> : tensor<f32> } : () -> tensor<f32>
  %mul1 = "tf.Sub"(%arg1, %mul1_const) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %ret = "tf.ConcatV2"(%mul0, %mul1, %arg2, %axis) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<i32>) -> tensor<?x3xf32>

  func.return %ret : tensor<?x3xf32>
}

// Do not synthesize binary ops when 1 of the 2 concat inputs is a non-binary op.
// CHECK-LABEL: testConcatCwiseBinarySynthMulOp2Inputs
func.func @testConcatCwiseBinarySynthMulOp2Inputs(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1xf32>) -> tensor<?x2xf32> {
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%arg0,
  // CHECK: "tf.ConcatV2"(%[[MUL]], %arg1,
  %axis = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  %mul0_const = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %mul0 = "tf.Mul"(%arg0, %mul0_const) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %ret = "tf.ConcatV2"(%mul0, %arg1, %axis) : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<i32>) -> tensor<?x2xf32>

  func.return %ret : tensor<?x2xf32>
}

// CHECK-LABEL: testLogOfSoftmax
func.func @testLogOfSoftmax(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Softmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Log"(%0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.LogSoftmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testLogToLog1p
func.func @testLogToLog1p(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<1xf32>
  %1 = "tf.Const"() {value = dense<2.0> : tensor<f32>} : () -> tensor<1xf32>
  %2 = "tf.Const"() {value = dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>

  // CHECK: %[[LOGP:.*]] = "tf.Log1p"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = "tf.AddV2"(%arg0, %0): (tensor<4x4xf32>, tensor<1xf32>) -> tensor<4x4xf32>
  %4 = "tf.Log"(%3): (tensor<4x4xf32>) -> tensor<4x4xf32>

  // CHECK: %[[ADD1:.*]] = "tf.AddV2"
  // CHECK: %[[LOG1:.*]] = "tf.Log"(%[[ADD1]])
  %5 = "tf.AddV2"(%4, %1): (tensor<4x4xf32>, tensor<1xf32>) -> tensor<4x4xf32>
  %6 = "tf.Log"(%5): (tensor<4x4xf32>) -> tensor<4x4xf32>

  // This is a legal canonicalization because constant shape 4xf32 is
  // broadcastable to 4x4xf32, however we currently do not support this case,
  // and canonicalize only if the constant is a scalar.
  // CHECK: %[[ADD2:.*]] = "tf.AddV2"
  // CHECK: %[[LOG2:.*]] = "tf.Log"(%[[ADD2]])
  %7 = "tf.AddV2"(%6, %2): (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %8 = "tf.Log"(%7): (tensor<4x4xf32>) -> tensor<4x4xf32>

  // CHECK: return %[[LOG2]]
  func.return %8: tensor<4x4xf32>
}

// CHECK-LABEL: testSubOfNeg
func.func @testSubOfNeg(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Sub"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testSubOfZero
func.func @testSubOfZero(%arg0: tensor<?x1xf32>, %arg1: tensor<4x1xf32>) -> (tensor<?x1xf32>, tensor<4x1xf32>) {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Sub"(%arg0, %0) : (tensor<?x1xf32>, tensor<f32>) -> tensor<?x1xf32>
  %2 = "tf.Sub"(%arg1, %0) : (tensor<4x1xf32>, tensor<f32>) -> tensor<4x1xf32>
  func.return %1, %2: tensor<?x1xf32>, tensor<4x1xf32>

// CHECK: return %arg0, %arg1
}

// CHECK-LABEL: testSubOfZeroWithBroadcasting
func.func @testSubOfZeroWithBroadcasting(%arg0: tensor<4x1xf32>) -> tensor<4x4xf32> {
  // This is an identity arithmetic operation, however we do not currently fold
  // it because it has a broadcasting.
  %0 = "tf.Const"() {value = dense<[[0.0, 0.0, 0.0, 0.0]]> : tensor<1x4xf32>} : () -> tensor<1x4xf32>
  %1 = "tf.Sub"(%arg0, %0) : (tensor<4x1xf32>, tensor<1x4xf32>) -> tensor<4x4xf32>
  func.return %1 : tensor<4x4xf32>

// CHECK: %[[CONST:.*]] = "tf.Const"()
// CHECK: %[[SUB:.*]] = "tf.Sub"(%arg0, %[[CONST]])
// CHECK: return %[[SUB]]
}

// CHECK-LABEL: testSquareOfSub
func.func @testSquareOfSub(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Square"(%0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddToAddV2
func.func @testAddToAddV2(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %0: tensor<8x16xf32>

// CHECK: %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testNoAddToAddV2ForStringType
func.func @testNoAddToAddV2ForStringType(%arg0: tensor<8x16x!tf_type.string>, %arg1: tensor<8x16x!tf_type.string>) -> tensor<8x16x!tf_type.string> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<8x16x!tf_type.string>, tensor<8x16x!tf_type.string>) -> tensor<8x16x!tf_type.string>
  func.return %0: tensor<8x16x!tf_type.string>

// CHECK: %0 = "tf.Add"(%arg0, %arg1) : (tensor<8x16x!tf_type.string>, tensor<8x16x!tf_type.string>) -> tensor<8x16x!tf_type.string>
// CHECK: return %0
}

// CHECK-LABEL: testAddOfNegLeft
func.func @testAddOfNegLeft(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Add"(%0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Sub"(%arg1, %arg0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddOfNegRight
func.func @testAddOfNegRight(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Add"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Sub"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddV2OfNegLeft
func.func @testAddV2OfNegLeft(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.AddV2"(%0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>
// CHECK: %0 = "tf.Sub"(%arg1, %arg0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddV2OfNegRight
func.func @testAddV2OfNegRight(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.AddV2"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Sub"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddV2IdentityScalar
func.func @testAddV2IdentityScalar(%arg0: tensor<f32>, %arg1: tensor<?xf32>, %arg2: tensor<4xf32>) -> (tensor<f32>, tensor<?xf32>, tensor<4xf32>) {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>

  // Identity scalar (0.0) is foldable with operand of any shape because
  // scalar is safely broadcastable to any shape.

  %1 = "tf.AddV2"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %2 = "tf.AddV2"(%arg1, %0) : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %3 = "tf.AddV2"(%arg2, %0) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

  %4 = "tf.AddV2"(%0, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %5 = "tf.AddV2"(%0, %2) : (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
  %6 = "tf.AddV2"(%0, %3) : (tensor<f32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK: return %arg0, %arg1, %arg2
  func.return %4, %5, %6: tensor<f32>, tensor<?xf32>, tensor<4xf32>
}

// CHECK-LABEL: testAddV2IdentityTensor
func.func @testAddV2IdentityTensor(%arg0: tensor<f32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = "tf.Const"() {value = dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>} : () -> tensor<4xf32>

  // If operand is a scalar, then the identity value (0.0 for addition) can
  // be of any shape, because operand is safely broadcastable to any shape.
  //
  // However we can't fold this arithmetic operation because the operand
  // shape does not match the result shape.

  %1 = "tf.AddV2"(%arg0, %0) : (tensor<f32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "tf.AddV2"(%0, %arg0) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>

  // If operand has the same shape as a result, we can fold it.
  %3 = "tf.AddV2"(%arg1, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "tf.AddV2"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK: %[[CONST:.*]] = "tf.Const"()
  // CHECK-DAG: %[[ADD1:.*]] = "tf.AddV2"(%arg0, %[[CONST]])
  // CHECK-DAG: %[[ADD2:.*]] = "tf.AddV2"(%arg0, %[[CONST]])
  // CHECK: return %[[ADD1]], %[[ADD2]], %arg1, %arg1
  func.return %1, %2, %3, %4: tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: testDoubleConj
func.func @testDoubleConj(%arg0: tensor<8x16x32x64xcomplex<f32>>) -> tensor<8x16x32x64xcomplex<f32>> {
  %0 = "tf.Conj"(%arg0) : (tensor<8x16x32x64xcomplex<f32>>) -> tensor<8x16x32x64xcomplex<f32>>
  %1 = "tf.Conj"(%0) : (tensor<8x16x32x64xcomplex<f32>>) -> tensor<8x16x32x64xcomplex<f32>>
  func.return %1: tensor<8x16x32x64xcomplex<f32>>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleInvert
func.func @testDoubleInvert(%arg0: tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Invert"(%arg0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Invert"(%0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  func.return %1: tensor<8x16x32x64xi32>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleLogicalNot
func.func @testDoubleLogicalNot(%arg0: tensor<8x16x32x64xi1>) -> tensor<8x16x32x64xi1> {
  %0 = "tf.LogicalNot"(%arg0) : (tensor<8x16x32x64xi1>) -> tensor<8x16x32x64xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16x32x64xi1>) -> tensor<8x16x32x64xi1>
  func.return %1: tensor<8x16x32x64xi1>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleNeg
func.func @testDoubleNeg(%arg0: tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Neg"(%arg0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Neg"(%0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  func.return %1: tensor<8x16x32x64xi32>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleReciprocal
func.func @testDoubleReciprocal(%arg0: tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Reciprocal"(%arg0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Reciprocal"(%0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  func.return %1: tensor<8x16x32x64xi32>

// CHECK: return %arg0
}

// CHECK-LABEL: testRedundantReshape
func.func @testRedundantReshape(%arg0: tensor<4x4xi32>) -> tensor<2x8xi32> {
  %0 = "tf.Const"() {value = dense<[8, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<[2, 8]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Reshape"(%arg0, %0) : (tensor<4x4xi32>, tensor<2xi32>) -> tensor<8x2xi32>
  %3 = "tf.Reshape"(%2, %1) : (tensor<8x2xi32>, tensor<2xi32>) -> tensor<2x8xi32>
  func.return %3: tensor<2x8xi32>

  // CHECK: %[[CONST:.*]] = "tf.Const"
  // CHECK-SAME: value = dense<[2, 8]> : tensor<2xi32>
  // CHECK: %[[RES:.*]] = "tf.Reshape"(%arg0, %[[CONST]])
  // CHECK: return %[[RES]] : tensor<2x8xi32>
}

// CHECK-LABEL: testReshapeToSelfShape
func.func @testReshapeToSelfShape(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = "tf.Shape"(%arg0) : (tensor<?x4xf32>) -> tensor<2xi32>
  %1 = "tf.Reshape"(%arg0, %0) : (tensor<?x4xf32>, tensor<2xi32>) -> tensor<?x4xf32>

  // CHECK: return %arg0 : tensor<?x4xf32>
  func.return %1: tensor<?x4xf32>
}

// CHECK-LABEL: func @testReshapeNoOp
func.func @testReshapeNoOp(%arg0: tensor<2x4xf32>, %arg1: tensor<2xi32>) -> tensor<2x4xf32> {
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>

  // CHECK: return %arg0
  func.return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @testBroadcastToNoOp
func.func @testBroadcastToNoOp(%arg0: tensor<2x4xf32>, %arg1: tensor<2xi32>) -> tensor<2x4xf32> {
  %0 = "tf.BroadcastTo"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2xi32>) -> tensor<2x4xf32>

  // CHECK: return %arg0
  func.return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @testPackShapeComputation
func.func @testPackShapeComputation(%arg0: tensor<?x1xf32>, %arg1: tensor<?x1x2xf32>, %arg2: tensor<*xf32>) -> (tensor<2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<*xi32>) {
  // Test dimensions sizes.
  %d1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %d2 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>

  // Slice bounds.
  %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>

  // Fold pack operation if it computes the input tensor shape:
  //
  //   %shape  = tf.Shape(%arg)                    // [? x ...]
  //   %dim0   = tf.StridedSlice(%shape, 0, 1, 1)  // get unknown dim0 value
  //   %pack   = tf.Pack(dim0, ...) { axis = 0 }   // [? x ...]
  //
  // Where `...` are some statically known dimensions. In this case %pack can be
  // replace with a %shape. This is a common pattern in models with a dynamic
  // batch size.

  // Test Rank 2
  // CHECK: %[[SHAPE0:.*]] = "tf.Shape"
  %3 = "tf.Shape"(%arg0) : (tensor<?x1xf32>) -> tensor<2xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %1) {shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %5 = "tf.Pack"(%4, %d1) {axis = 0 : i64} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %6 = "tf.Reshape"(%arg0, %5) : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x1xf32>

  // Test Rank 3.
  // CHECK: %[[SHAPE1:.*]] = "tf.Shape"
  %7 = "tf.Shape"(%arg1) : (tensor<?x1x2xf32>) -> tensor<3xi32>
  %8 = "tf.StridedSlice"(%7, %0, %1, %1) {shrink_axis_mask = 1 : i64} : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %9 = "tf.Pack"(%8, %d1, %d2) {axis = 0 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  %10 = "tf.Reshape"(%arg1, %9) : (tensor<?x1x2xf32>, tensor<3xi32>) -> tensor<?x1x2xf32>

  // Packed dimensions have different order from the reshape operand:
  //   [?, 1, 2] vs [?, 2, 1]
  %14 = "tf.StridedSlice"(%7, %0, %1, %1) {shrink_axis_mask = 1 : i64} : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %15 = "tf.Pack"(%14, %d2, %d1) {axis = 0 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  // CHECK: %[[PACK0:.*]] = "tf.Pack"

  // Packed dimensions have higher rank than the reshape operand:
  //   [?, 1] vs [?, 1, 1]
  %16 = "tf.StridedSlice"(%3, %0, %1, %1) {shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %17 = "tf.Pack"(%16, %d1, %d1) {axis = 0 : i64} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  // CHECK: %[[PACK1:.*]] = "tf.Pack"

  // Make sure a dynamic ranked shape doesn't crash the "canonicalize" pass
  %18 = "tf.Shape"(%arg2) : (tensor<*xf32>) -> tensor<*xi32>
  %19 = "tf.StridedSlice"(%18, %0, %1, %1) {shrink_axis_mask = 1 : i64} : (tensor<*xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32>
  %20 = "tf.Pack"(%19, %d1) {axis = 0 : i64} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  // CHECK: %[[PACK2:.*]] = "tf.Pack"

  // CHECK: return %[[SHAPE0]], %[[SHAPE1]], %[[PACK0]], %[[PACK1]], %[[PACK2]]
  func.return %5, %9, %15, %17, %20 : tensor<2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<*xi32>
}

// CHECK-LABEL: testTileMultiplesAllOnes
func.func @testTileMultiplesAllOnes(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense <[1, 1]> : tensor<2xi32>
  // CHECK: return %arg0
  %0 = "tf.Tile"(%arg0, %cst) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
  func.return %0: tensor<2x3xf32>
}

// CHECK-LABEL: func @testStaticAndIdenticalTypeForEqualOp
func.func @testStaticAndIdenticalTypeForEqualOp(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK:      "tf.Equal"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = true
  %0 = "tf.Equal"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @testStaticAndIdenticalTypeForNotEqualOp
func.func @testStaticAndIdenticalTypeForNotEqualOp(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = true
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @testUnknownBroadcastForNotEqualOp
func.func @testUnknownBroadcastForNotEqualOp(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<*xi1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = false
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<?xi32>, tensor<?xi32>) -> tensor<*xi1>
  func.return %0: tensor<*xi1>
}

// CHECK-LABEL: func @testKnownGoodBroadcastForNotEqualOp
func.func @testKnownGoodBroadcastForNotEqualOp(%arg0: tensor<1x?xi32>, %arg1: tensor<?x1xi32>) -> tensor<?x?xi1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = true
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<1x?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  func.return %0: tensor<?x?xi1>
}

// CHECK-LABEL: func @testKnownBadBroadcastForNotEqualOp
func.func @testKnownBadBroadcastForNotEqualOp(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<i1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = false
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<?xi32>, tensor<?xi32>) -> tensor<i1>
  func.return %0: tensor<i1>
}

// CHECK-LABEL: func @testUnrankedRHSForNotEqualOp
func.func @testUnrankedRHSForNotEqualOp(%arg0: tensor<i32>, %arg1: tensor<*xi32>) -> tensor<i1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = false
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<i32>, tensor<*xi32>) -> tensor<i1>
  func.return %0: tensor<i1>
}

// CHECK-LABEL: func @testUnrankedLHSForNotEqualOp
func.func @testUnrankedLHSForNotEqualOp(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<i1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = false
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  func.return %0: tensor<i1>
}

// CHECK-LABEL: func @testScalarForNotEqualOp
func.func @testScalarForNotEqualOp(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
  // CHECK:      "tf.NotEqual"(%arg0, %arg1)
  // CHECK-SAME:   incompatible_shape_error = true
  %0 = "tf.NotEqual"(%arg0, %arg1) { incompatible_shape_error = false } : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %0: tensor<i1>
}

// CHECK-LABEL: testLogicalNotOfEqual
func.func @testLogicalNotOfEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  func.return %1: tensor<8x16xi1>

// CHECK: %[[NE:.*]] = "tf.NotEqual"(%arg0, %arg1) {incompatible_shape_error = true}
// CHECK: return %[[NE]]
}

// CHECK-LABEL: testLogicalNotOfNotEqual
func.func @testLogicalNotOfNotEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.NotEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  func.return %1: tensor<8x16xi1>

// CHECK: %[[NE:.*]] = "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = true}
// CHECK: return %[[NE]]
}

// CHECK-LABEL: testLogicalNotOfGreater
func.func @testLogicalNotOfGreater(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  func.return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testLogicalNotOfGreaterEqual
func.func @testLogicalNotOfGreaterEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  func.return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.Less"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testLogicalNotOfLess
func.func @testLogicalNotOfLess(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  func.return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testLogicalNotOfLessEqual
func.func @testLogicalNotOfLessEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  func.return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.Greater"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testSizeFolding
func.func @testSizeFolding(%arg0: tensor<3x5x7xf32>) -> tensor<i32> {
  %0 = "tf.Size"(%arg0) : (tensor<3x5x7xf32>) -> tensor<i32>
  func.return %0: tensor<i32>

// CHECK: %[[CONST:.*]] = "tf.Const"() {value = dense<105> : tensor<i32>} : () -> tensor<i32>
// CHECK: return %[[CONST]] : tensor<i32>
}

// CHECK-LABEL: testDivWithSqrtDivisor
func.func @testDivWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Div"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.Mul"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: testRealDivWithSqrtDivisor
func.func @testRealDivWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.RealDiv"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.Mul"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: testRealDivWithConstDivisor
func.func @testRealDivWithConstDivisor(%arg0: tensor<8x2xf32>) -> tensor<8x2xf32> {
  %0 = "tf.Const"() {value = dense<[2.0, 4.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = "tf.RealDiv"(%arg0, %0) : (tensor<8x2xf32>, tensor<2xf32>) -> tensor<8x2xf32>
  func.return %1: tensor<8x2xf32>

  // CHECK: %[[CONST:.*]] = "tf.Const"
  // CHECK-SAME: value = dense<[5.000000e-01, 2.500000e-01]
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%arg0, %[[CONST]])
  // CHECK: return %[[MUL]]
}

// CHECK-LABEL: testTruncateDivWithSqrtDivisor
func.func @testTruncateDivWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.TruncateDiv"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.Mul"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: testXdivyWithSqrtDivisor
func.func @testXdivyWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Xdivy"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.MulNoNan"(%0, %arg0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: @identityTranspose
func.func @identityTranspose(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "tf.Const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x5x6xf32>

  func.return %1 : tensor<2x3x4x5x6xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: @identityTransposeConst
func.func @identityTransposeConst(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = arith.constant dense<[0, 1, 2, 3, 4]> : tensor<5xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x5x6xf32>

  func.return %1 : tensor<2x3x4x5x6xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: @nonIdentityTranspose
func.func @nonIdentityTranspose(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x6x5xf32> {
  %0 = "tf.Const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x6x5xf32>

  func.return %1 : tensor<2x3x4x6x5xf32>
  // CHECK: %[[CONST:.*]] = "tf.Const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
  // CHECK: %[[TRANS:.*]] = "tf.Transpose"(%arg0, %[[CONST]]) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x6x5xf32>
  // CHECK: return %[[TRANS]]
}

// CHECK-LABEL: @cancellableTranspose
func.func @cancellableTranspose(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Transpose"(%2, %1) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x8xf32>

  func.return %3 : tensor<1x4x4x8xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: @cancellableTransposeConst
func.func @cancellableTransposeConst(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %0 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  %2 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Transpose"(%2, %1) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x8xf32>

  func.return %3 : tensor<1x4x4x8xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: @nonCancellableTranspose
func.func @nonCancellableTranspose(%arg0: tensor<1x4x4x8xf32>) -> tensor<4x1x4x8xf32> {
  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Const"() {value = dense<[2, 0, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Transpose"(%2, %1) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<4x1x4x8xf32>

  func.return %3 : tensor<4x1x4x8xf32>

  // CHECK-DAG: %[[CONST1:.*]] = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>}
  // CHECK-DAG: %[[CONST2:.*]] = "tf.Const"() {value = dense<[2, 0, 3, 1]> : tensor<4xi32>}
  // CHECK: %[[TRANS1:.*]] = "tf.Transpose"(%arg0, %[[CONST1]]) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  // CHECK: %[[TRANS2:.*]] = "tf.Transpose"(%[[TRANS1]], %[[CONST2]]) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<4x1x4x8xf32>
  // CHECK: return %[[TRANS2]]
}

// CHECK-LABEL: func @addN
func.func @addN(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: return %arg0
  %0 = "tf.AddN"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @addNWithZerosFloat
func.func @addNWithZerosFloat(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %0 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK-DAG: [[ZERO:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>}
  // CHECK-DAG: [[ONE:%.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<2xf32>}
  // CHECK: [[ADD_N:%.*]] = "tf.AddN"(%arg0, [[ZERO]], [[ONE]])
  // CHECK: return %arg0, %arg0, [[ZERO]], [[ADD_N]]
  %2 = "tf.AddN"(%arg0, %1, %1) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %3 = "tf.AddN"(%1, %arg0, %1) : (tensor<2xf32>, tensor<2xf32> , tensor<2xf32>) -> tensor<2xf32>
  %4 = "tf.AddN"(%1, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %5 = "tf.AddN"(%arg0, %1, %0) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %2, %3, %4, %5: tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: func @addNWithZerosInt
func.func @addNWithZerosInt(%arg0: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK-DAG: [[ZERO:%.*]] = "tf.Const"() {value = dense<0> : tensor<2xi32>}
  // CHECK-DAG: [[ONE:%.*]] = "tf.Const"() {value = dense<1> : tensor<2xi32>}
  // CHECK: [[ADD_N:%.*]] = "tf.AddN"(%arg0, [[ZERO]], [[ONE]])
  // CHECK: return %arg0, %arg0, [[ZERO]], [[ADD_N]]
  %2 = "tf.AddN"(%arg0, %1, %1) : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %3 = "tf.AddN"(%1, %arg0, %1) : (tensor<2xi32>, tensor<2xi32> , tensor<2xi32>) -> tensor<2xi32>
  %4 = "tf.AddN"(%1, %1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %5 = "tf.AddN"(%arg0, %1, %0) : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %2, %3, %4, %5: tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL: func @addNSkipFoldingIfBroadcasting
func.func @addNSkipFoldingIfBroadcasting(%arg0: tensor<1xf32>) -> tensor<10xf32> {
  %0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32>
  // CHECK: [[ZERO:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<10xf32>}
  // CHECK: [[ADD_N:%.*]] = "tf.AddN"(%arg0, [[ZERO]])
  // CHECK: return [[ADD_N]]
  %1 = "tf.AddN"(%arg0, %0) : (tensor<1xf32>, tensor<10xf32>) -> tensor<10xf32>
  func.return %1: tensor<10xf32>
}

// CHECK-LABEL: func @ToBool_0DScalarI1
func.func @ToBool_0DScalarI1(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: return %arg0
  %0 = "tf.ToBool"(%arg0) : (tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_0DScalarInt
func.func @ToBool_0DScalarInt(%arg0: tensor<i32>) -> tensor<i1> {
  // CHECK: [[Zero:%.*]] = "tf.Const"() {value = dense<0> : tensor<i32>}
  // CHECK: [[NE:%.*]] = "tf.NotEqual"(%arg0, [[Zero]])
  // CHECK: return [[NE]]
  %0 = "tf.ToBool"(%arg0) : (tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_0DScalarFloat
func.func @ToBool_0DScalarFloat(%arg0: tensor<f32>) -> tensor<i1> {
  // CHECK: [[Zero:%.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: [[NE:%.*]] = "tf.NotEqual"(%arg0, [[Zero]])
  // CHECK: return [[NE]]
  %0 = "tf.ToBool"(%arg0) : (tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_0DScalarString
func.func @ToBool_0DScalarString(%arg0: tensor<!tf_type.string>) -> tensor<i1> {
  // CHECK: [[EmptyStr:%.*]] = "tf.Const"() {value = dense<""> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  // CHECK: [[NE:%.*]] = "tf.NotEqual"(%arg0, [[EmptyStr]]) {incompatible_shape_error = true} : (tensor<!tf_type.string>, tensor<!tf_type.string>) -> tensor<i1>
  // CHECK: return [[NE]] : tensor<i1>
  %0 = "tf.ToBool"(%arg0) : (tensor<!tf_type.string>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_1DTensor
func.func @ToBool_1DTensor(%arg0: tensor<1xf32>) -> tensor<i1> {
  // CHECK: [[Const:%.*]] = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  // CHECK: return [[Const]]
  %0 = "tf.ToBool"(%arg0) : (tensor<1xf32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_1DTensorZeroDim
func.func @ToBool_1DTensorZeroDim(%arg0: tensor<0xf32>) -> tensor<i1> {
  // CHECK: [[Const:%.*]] = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
  // CHECK: return [[Const]]
  %0 = "tf.ToBool"(%arg0) : (tensor<0xf32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_2DTensor
func.func @ToBool_2DTensor(%arg0: tensor<1x5xf32>) -> tensor<i1> {
  // CHECK: [[Const:%.*]] = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  // CHECK: return [[Const]]
  %0 = "tf.ToBool"(%arg0) : (tensor<1x5xf32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: func @ToBool_2DTensorZeroDim
func.func @ToBool_2DTensorZeroDim(%arg0: tensor<1x0xf32>) -> tensor<i1> {
  // CHECK: [[Const:%.*]] = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
  // CHECK: return [[Const]]
  %0 = "tf.ToBool"(%arg0) : (tensor<1x0xf32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// CHECK-LABEL: testReadVariableOpOfCast
func.func @testReadVariableOpOfCast(%arg0: tensor<!tf_type.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf_type.resource<tensor<8x40xf32>>>) -> tensor<*x!tf_type.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource>) -> tensor<8x40xf32>
  func.return %1: tensor<8x40xf32>

// CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32>
// CHECK: return %0
}

// CHECK-LABEL: testReadVariableOpOfCastWithTruncate
func.func @testReadVariableOpOfCastWithTruncate(%arg0: tensor<!tf_type.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32> {
  %0 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<!tf_type.resource<tensor<8x40xf32>>>) -> tensor<*x!tf_type.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource>) -> tensor<8x40xf32>
  func.return %1: tensor<8x40xf32>

// CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32>
// CHECK: return %0
}

// CHECK-LABEL: testReadVariableOpOfCastMultiUse
func.func @testReadVariableOpOfCastMultiUse(%arg0: tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32> {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<*x!tf_type.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource>) -> tensor<f32>
  "tf.AssignVariableOp"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<f32>) -> ()
  func.return %1: tensor<f32>

 // CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<*x!tf_type.resource>
 // CHECK: %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource>) -> tensor<f32>
 // CHECK: "tf.AssignVariableOp"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<f32>) -> ()
 // CHECK: return %1
}

// CHECK-LABEL: testMultiReadVariableOpsOfCast
func.func @testMultiReadVariableOpsOfCast(%arg0: tensor<!tf_type.resource<tensor<f32>>>) -> (tensor<f32>, tensor<f32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<*x!tf_type.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource>) -> tensor<f32>
  %2 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf_type.resource>) -> tensor<f32>
  func.return %1, %2: tensor<f32>, tensor<f32>

 // CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
 // CHECK: %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
 // CHECK: return %0, %1
}

// CHECK-LABEL: testRankOfRankedTensor
func.func @testRankOfRankedTensor(%arg0 : tensor<4x3x2xf32>) -> tensor<i32> {
  // CHECK:[[VAL0:%.+]] = "tf.Const"() {value = dense<3> : tensor<i32>}
  %0 = "tf.Rank"(%arg0) : (tensor<4x3x2xf32>) -> tensor<i32>

  // CHECK: return [[VAL0]]
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: testRankOfRankedTensorUnrankedOutput
func.func @testRankOfRankedTensorUnrankedOutput(%arg0 : tensor<4x3x2xf32>) -> tensor<*xi32> {
  // Regression test to make sure we don't crash in this case.
  %0 = "tf.Rank"(%arg0) : (tensor<4x3x2xf32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: testRankOfRankedTensorDynamicShapeOutput
func.func @testRankOfRankedTensorDynamicShapeOutput(%arg0 : tensor<4x3x2xf32>) -> tensor<?xi32> {
  // Regression test to make sure we don't crash in this case.
  %0 = "tf.Rank"(%arg0) : (tensor<4x3x2xf32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: @foldFill
func.func @foldFill() -> (tensor<3x2x1xf32>, tensor<*xf32>, tensor<*xcomplex<f32>>) {
  %0 = "tf.Const"() {value = dense<[3, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.Const"() {value = dense<23.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: "tf.Const"() {value = dense<2.300000e+01> : tensor<3x2x1xf32>}
  %2 = "tf.Fill"(%0, %1) : (tensor<3xi32>, tensor<f32>) -> tensor<3x2x1xf32>
  // CHECK-DAG: "tf.Const"() {value = dense<2.300000e+01> : tensor<3x2x1xf32>}
  %3 = "tf.Fill"(%0, %1) : (tensor<3xi32>, tensor<f32>) -> tensor<*xf32>

  %complex_cst = "tf.Const"() {value = dense<(0.000000e+00,1.000000e+00)> : tensor<complex<f32>>} : () -> tensor<complex<f32>>
  // Here, custom folder doesn't handle complex dtypes and it is folded through
  // the constant folding hook.
  // TODO(hinsu): Handle complex dtypes in the custom folder for FillOp.
  // CHECK-DAG: "tf.Const"() {value = dense<(0.000000e+00,1.000000e+00)> : tensor<3x2x1xcomplex<f32>>} : () -> tensor<*xcomplex<f32>>
  %4 = "tf.Fill"(%0, %complex_cst) : (tensor<3xi32>, tensor<complex<f32>>) -> tensor<*xcomplex<f32>>

  func.return %2, %3, %4 : tensor<3x2x1xf32>, tensor<*xf32>, tensor<*xcomplex<f32>>
}

// CHECK-LABEL: foldIf
func.func @foldIf(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> (tensor<f32>) {
  %0 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
  %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>

  // CHECK: %0 = "tf.PartitionedCall"(%arg0, %arg1)
  // CHECK-SAME: device = "noodle"
  // CHECK-SAME: f = @sub
  %2 = "tf.If"(%0, %arg0, %arg1) {then_branch = @add, else_branch = @sub, output_shapes = [#tf_type.shape<>], device = "noodle", is_stateless = true} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: %1 = "tf.StatefulPartitionedCall"(%0, %arg1)
  // CHECK-SAME: _underscore_attr = "something"
  // CHECK-SAME: f = @add
  %3 = "tf.If"(%1, %2, %arg1) {then_branch = @add, else_branch = @sub, output_shapes = [#tf_type.shape<>], _underscore_attr = "something", is_stateless = false} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: %2 = "tf.If"
  %4 = "tf.If"(%arg2, %3, %arg1) {then_branch = @add, else_branch = @sub, is_stateless = false} : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>

  // CHECK: return %2
  func.return %4 : tensor<f32>
}

// CHECK-LABEL: foldIfRegion
func.func @foldIfRegion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> (tensor<f32>, tensor<f32>) {
  %false = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
  %true = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>

  // CHECK: [[Val0:%.*]] = "tf.Mul"(%arg0, %arg1)
  %0 = "tf.IfRegion"(%true) ({
      %true_value = "tf.Mul"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%true_value) : (tensor<f32>) -> ()
    }, {
      %false_value = "tf.Sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%false_value) : (tensor<f32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<f32>

  // CHECK: [[Val1:%.*]] = "tf.Sub"(%arg0, %arg1)
  %1 = "tf.IfRegion"(%false) ({
      %true_value = "tf.Mul"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%true_value) : (tensor<f32>) -> ()
    }, {
      %false_value = "tf.Sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%false_value) : (tensor<f32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<f32>

  // CHECK: return [[Val0]], [[Val1]]
  func.return %0, %1 : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: foldIfRegionMismatchedTypes
func.func @foldIfRegionMismatchedTypes(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<i1>) -> tensor<1xf32> {
  %false = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
  %true = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>

  // CHECK: [[Val0:%.*]] = "tf.Mul"(%arg0, %arg1)
  // CHECK-NEXT: [[Cast:%.*]] = "tf.Cast"([[Val0]])
  // CHECK-NEXT: return [[Cast]]
  %0 = "tf.IfRegion"(%true) ({
      %true_value = "tf.Mul"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      "tf.Yield"(%true_value) : (tensor<?xf32>) -> ()
    }, {
      %false_value = "tf.Sub"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      "tf.Yield"(%false_value) : (tensor<?xf32>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK-LABEL: func @eliminatePassThroughIfRegion(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<!tf_type.resource>
func.func @eliminatePassThroughIfRegion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<!tf_type.resource>) -> (tensor<f32>) {
  // CHECK: %[[PRED:.*]] = "tf._SomeOp"() : () -> tensor<i1>
  %pred = "tf._SomeOp"() : () -> tensor<i1>
  // CHECK: %[[IF_OUTPUT:.*]] = "tf.IfRegion"(%[[PRED]]) ({
  // CHECK:   %[[MUL:.*]] = "tf.Mul"(%[[ARG0]], %[[ARG1]])
  // CHECK:   "tf.Yield"(%[[MUL]]) : (tensor<f32>)
  // CHECK:  },  {
  // CHECK:    %[[SUB:.*]] = "tf.Sub"(%[[ARG0]], %[[ARG1]])
  // CHECK:    "tf.Yield"(%[[SUB]]) : (tensor<f32>)
  // CHECK:  }) {is_stateless = true} : (tensor<i1>) -> tensor<f32>
  %0:4 = "tf.IfRegion"(%pred) ({
      %true_value = "tf.Mul"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%arg1, %arg2, %true_value, %arg2) : (tensor<f32>, tensor<!tf_type.resource>, tensor<f32>, tensor<!tf_type.resource>) -> ()
    }, {
      %false_value = "tf.Sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%arg1, %arg2, %false_value, %arg2) : (tensor<f32>, tensor<!tf_type.resource>, tensor<f32>, tensor<!tf_type.resource>) -> ()
    }) { is_stateless = true}: (tensor<i1>) -> (tensor<f32>, tensor<!tf_type.resource>, tensor<f32>, tensor<!tf_type.resource>)
  // CHECK: "tf._SomeOp"(%[[ARG2]], %[[ARG1]]) : (tensor<!tf_type.resource>, tensor<f32>) -> ()
  "tf._SomeOp"(%0#1, %0#0) : (tensor<!tf_type.resource>, tensor<f32>) -> ()
  // CHECK: "tf._SomeOp"(%[[ARG2]], %[[IF_OUTPUT]]) : (tensor<!tf_type.resource>, tensor<f32>) -> ()
  "tf._SomeOp"(%0#3, %0#2) : (tensor<!tf_type.resource>, tensor<f32>) -> ()
  // CHECK: return %[[IF_OUTPUT]] : tensor<f32>
  func.return %0#2 : tensor<f32>
}

// CHECK-LABEL: func @eliminatePassThroughCaseRegion(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<!tf_type.resource>
func.func @eliminatePassThroughCaseRegion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<!tf_type.resource>) -> (tensor<f32>) {
  // CHECK: %[[INDEX:.*]] = "tf._SomeOp"() : () -> tensor<i32>
  %index = "tf._SomeOp"() : () -> tensor<i32>
  // CHECK: %[[CASE_OUTPUT:.*]] = "tf.CaseRegion"(%[[INDEX]]) ({
  // CHECK:   %[[MUL:.*]] = "tf.Mul"(%[[ARG0]], %[[ARG1]])
  // CHECK:   "tf.Yield"(%[[MUL]]) : (tensor<f32>)
  // CHECK:  },  {
  // CHECK:    %[[SUB:.*]] = "tf.Sub"(%[[ARG0]], %[[ARG1]])
  // CHECK:    "tf.Yield"(%[[SUB]]) : (tensor<f32>)
  // CHECK:  },  {
  // CHECK:    %[[ADD:.*]] = "tf.AddV2"(%[[ARG0]], %[[ARG1]])
  // CHECK:    "tf.Yield"(%[[ADD]]) : (tensor<f32>)
  // CHECK:  }) {is_stateless = true} : (tensor<i32>) -> tensor<f32>
  %0:3 = "tf.CaseRegion"(%index) ({
      %mul = "tf.Mul"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%arg1, %mul, %arg2) : (tensor<f32>, tensor<f32>, tensor<!tf_type.resource>) -> ()
    }, {
      %sub = "tf.Sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%arg1, %sub, %arg2) : (tensor<f32>, tensor<f32>, tensor<!tf_type.resource>) -> ()
    }, {
      %add = "tf.AddV2"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "tf.Yield"(%arg1, %add, %arg2) : (tensor<f32>, tensor<f32>, tensor<!tf_type.resource>) -> ()
    }) { is_stateless = true}: (tensor<i32>) -> (tensor<f32>, tensor<f32>, tensor<!tf_type.resource>)
  // CHECK: "tf._SomeOp"(%[[ARG2]], %[[ARG1]]) : (tensor<!tf_type.resource>, tensor<f32>) -> ()
  "tf._SomeOp"(%0#2, %0#0) : (tensor<!tf_type.resource>, tensor<f32>) -> ()
  // CHECK: return %[[CASE_OUTPUT]] : tensor<f32>
  func.return %0#1 : tensor<f32>
}


// CHECK-LABEL: foldCase
func.func @foldCase(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
  %2 = arith.constant dense<1> : tensor<i32>
  %3 = arith.constant dense<0> : tensor<i32>

  // CHECK: PartitionedCall
  // CHECK-SAME: device = "noodle"
  // CHECK-SAME: f = @add
  %4 = "tf.Case"(%2, %arg0, %arg1) {branches = [@sub, @add], output_shapes = [#tf_type.shape<>], device = "noodle", is_stateless = false} : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK: PartitionedCall
  // CHECK-SAME: _cluster_launch = "not_ready"
  // CHECK-SAME: f = @sub
  %5 = "tf.Case"(%3, %4, %arg1) {branches = [@sub, @add], output_shapes = [#tf_type.shape<>], _cluster_launch = "not_ready", is_stateless = false} : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %5 : tensor<f32>
}

func.func @add(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Add"(%arg0, %arg1): (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @sub(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: testBatchToSpaceToBatchToSpaceND
// CHECK-SAME: ([[INPUT:%.*]]: tensor<?x?x?x?xf32>, [[CROPS:%.*]]: tensor<?x?xi32>)
func.func @testBatchToSpaceToBatchToSpaceND(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?xi32>) -> tensor<*xf32> {
  // CHECK: [[BLOCK_SHAPE:%.*]] = "tf.Const"() {value = dense<8> : tensor<2xi64>}
  // CHECK: [[BATCH_TO_SHAPE_ND:%.*]] = "tf.BatchToSpaceND"([[INPUT]], [[BLOCK_SHAPE]], [[CROPS]])
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 8 : i64} : (tensor<?x?x?x?xf32>, tensor<?x?xi32>) -> tensor<*xf32>
  // CHECK: return [[BATCH_TO_SHAPE_ND]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: testBatchToSpaceDynamicInput
func.func @testBatchToSpaceDynamicInput(%arg0: tensor<*xf32>, %arg1: tensor<?x?xi32>) -> tensor<*xf32> {
  // CHECK-NOT: "tf.BatchToSpaceND"
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 8 : i64} : (tensor<*xf32>, tensor<?x?xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: testBatchToSpaceDynamicCrops
func.func @testBatchToSpaceDynamicCrops(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<*xi32>) -> tensor<*xf32> {
  // CHECK-NOT: "tf.BatchToSpaceND"
  %0 = "tf.BatchToSpace"(%arg0, %arg1) {block_size = 8 : i64} : (tensor<?x?x?x?xf32>, tensor<*xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: @erase_tf_var_is_initialized
func.func @erase_tf_var_is_initialized(%arg0 : tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1> {
  %vh = "tf.VarHandleOp"() {container = "", shape = "tfshape$", shared_name = "x"} : () -> tensor<!tf_type.resource<tensor<f32>>>
  %is = "tf.VarIsInitializedOp"(%vh) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  %res = "tf.UnknownOp"(%vh) : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<i1>
  func.return %res : tensor<i1>
}
// Unused VarIsInitializedOp is erased.
// CHECK: tf.VarHandleOp
// CHECK-NEXT: tf.UnknownOp


// Simple pass through value
// CHECK-LABEL: testWhileRegionSimplePassThrough
func.func @testWhileRegionSimplePassThrough(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: "tf.WhileRegion"(%arg1)
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%barg0, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return %arg0 : tensor<*xf32>
  func.return %0#0 : tensor<*xf32>
}

// Explicit capture and return of extern values is removed.
// CHECK-LABEL: testWhileRegionReturnExternValues
func.func @testWhileRegionReturnExternValues(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // CHECK: "tf.WhileRegion"(%arg1)
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>):
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%arg0, %sub) : (tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // CHECK: return %arg0 : tensor<*xf32>
  func.return %0#0 : tensor<*xf32>
}

// Multiple pass through values
// CHECK-LABEL: testWhileRegionMultiplePassThrough
func.func @testWhileRegionMultiplePassThrough(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<i32>) -> tensor<*xf32> {
  // Verify that first 3 operands are elimiinated.
  // CHECK: "tf.WhileRegion"(%arg3)
  %0:4 = "tf.WhileRegion"(%arg0, %arg1, %arg2, %arg3) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0 : tensor<*xf32>, %carg1 : tensor<*xf32>, %carg2 : tensor<*xf32>, %carg3 : tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg3, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0 : tensor<*xf32>, %barg1 : tensor<*xf32>, %barg2 : tensor<*xf32>, %barg3 : tensor<i32>):
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg3, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%barg0, %barg1, %barg2, %sub) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>)

  // CHECK: %[[SUB0:.*]] = "tf.Sub"(%arg0, %arg1)
  // CHECK: %[[SUB1:.*]] = "tf.Sub"(%arg2, %[[SUB0]])
  // CHECK: return %[[SUB1]]
  %sub0 = "tf.Sub" (%0#0, %0#1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %sub1 = "tf.Sub" (%0#2, %sub0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %sub1 : tensor<*xf32>
}

// Multiple non contiguous pass through values
// CHECK-LABEL: testWhileRegionMultiplePassThroughNonContiguous
func.func @testWhileRegionMultiplePassThroughNonContiguous(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<i32>) -> tensor<*xf32> {
  // Verify arg0 and arg2 are eliminated
  // CHECK: %[[WHILE_OUT:.*]]:2 = "tf.WhileRegion"(%arg1, %arg3)
  %0:4 = "tf.WhileRegion"(%arg0, %arg1, %arg2, %arg3) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0 : tensor<*xf32>, %carg1 : tensor<*xf32>, %carg2 : tensor<*xf32>, %carg3 : tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg3, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0 : tensor<*xf32>, %barg1 : tensor<*xf32>, %barg2 : tensor<*xf32>, %barg3 : tensor<i32>):
      %arg1neg = "tf.Neg"(%barg1) : (tensor<*xf32>) -> tensor<*xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg3, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%barg0, %arg1neg, %barg2, %sub) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<i32>)

  // Verify that use of while loop results corresponding to result #0 and 2 of
  // the while are replaces with corresponding WhileRegion operands
  // CHECK: %[[SUB0:.*]] = "tf.Sub"(%arg0, %[[WHILE_OUT]]#0)
  // CHECK: %[[SUB1:.*]] = "tf.Sub"(%arg2, %[[SUB0]])
  // CHECK: return %[[SUB1]]
  %sub0 = "tf.Sub" (%0#0, %0#1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %sub1 = "tf.Sub" (%0#2, %sub0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %sub1 : tensor<*xf32>
}

// Pass through but with type mismatch (tensor<*xf32> is compatible with
// tensor<?x?xf32> in the body). WhileRegion canonicalization does not handle
// this.
// CHECK-LABEL: testWhileRegionPassThroughTypeMismatch
func.func @testWhileRegionPassThroughTypeMismatch(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>) -> tensor<*xf32> {
  // Verify that the While stay's unchanged
  // CHECK: "tf.WhileRegion"(%arg0, %arg1)
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<?x?xf32>, %barg1: tensor<i32>):
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "tf.Yield"(%barg0, %sub) : (tensor<?x?xf32>, tensor<i32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<i32>)
  // Verify that the result stays uchanged
  // CHECK: return %arg0 : tensor<*xf32>
  func.return %0#0 : tensor<*xf32>
}

// Unused value flowing through the while (operand 2 and 3, is unused in the
// while and the corresponding result is unused as well). Canonicalization will
// eliminate them.
// CHECK-LABEL: testWhileRegionUnusedValue
func.func @testWhileRegionUnusedValue(%arg0 : tensor<*xf32>, %arg1 : tensor<i32>, %arg2: tensor<i32>) -> tensor<*xf32> {
  %cst = arith.constant dense <33.0> : tensor<f32>
  // Verify that last 2 operands of while (unused) are removed
  // CHECK: %[[WHILE_OUT:.*]]:2 = "tf.WhileRegion"(%arg0, %arg1)
  %0:4 = "tf.WhileRegion"(%arg0, %arg1, %arg2, %cst) (
    {
      // condition, check if count has reached 0
      ^bb0(%carg0: tensor<*xf32>, %carg1: tensor<i32>, %carg2:tensor<i32>, %carg3:tensor<f32>):
      %zero = arith.constant dense<0> : tensor<i32>
      %ne = "tf.NotEqual"(%carg1, %zero) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tf.Yield"(%ne) : (tensor<i1>) -> ()
    },
    {
      // loop body
      ^bb0(%barg0: tensor<*xf32>, %barg1: tensor<i32>, %barg2:tensor<i32>, %barg3:tensor<f32>):
      %add = "tf.Add"(%barg0, %barg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      %one = arith.constant dense<1> : tensor<i32>
      %sub = "tf.Sub"(%barg1, %one) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %dummy0 = arith.constant dense<7> : tensor<i32>
      %dummy1 = arith.constant dense<3.0> : tensor<f32>
      "tf.Yield"(%add, %sub, %dummy0, %dummy1) : (tensor<*xf32>, tensor<i32>, tensor<i32>, tensor<f32>) -> ()
    }
  ) { is_stateless = false } : (tensor<*xf32>, tensor<i32>,  tensor<i32>, tensor<f32>) -> (tensor<*xf32>, tensor<i32>,  tensor<i32>, tensor<f32>)

  // Verify that return still uses while result # 0
  // CHECK: return %[[WHILE_OUT]]#0 : tensor<*xf32>
  func.return %0#0 : tensor<*xf32>
}

// Check that output_shapes attribute is removed for tf.If
func.func private @testIfThen(tensor<*xf32>) -> tensor<*xf32>
func.func private @testIfElse(tensor<*xf32>) -> tensor<*xf32>
// CHECK-LABEL: func @testIfDropOutputShapes
func.func @testIfDropOutputShapes(tensor<i1>, tensor<2xf32>) -> tensor<2xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2xf32>):
  // CHECK: "tf.If"
  // CHECK-NOT: output_shapes
  %1 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false, output_shapes = [#tf_type.shape<>]
  } : (tensor<i1>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// CHECK-LABEL: testNMSV3ToNMSV4
func.func @testNMSV3ToNMSV4(%arg0: tensor<3x4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<2xi32> {
  %max_size = arith.constant dense<2> : tensor<i32>
  // CHECK: "tf.NonMaxSuppressionV4"
  %0 = "tf.NonMaxSuppressionV3"(%arg0, %arg1, %max_size, %arg2, %arg3): (tensor<3x4xf32>, tensor<3xf32>, tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<2xi32>)
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: testFusedBatchNormToBatchNormV3
func.func @testFusedBatchNormToBatchNormV3(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // CHECK: "tf.FusedBatchNormV3"
  %0:5 = "tf.FusedBatchNorm"(%arg0, %arg1, %arg2, %arg3, %arg4): (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32> )
  func.return %0#0  : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @testSumFoldBypass
func.func @testSumFoldBypass(%arg0: tensor<4x?xf16>, %arg1: tensor<*xi64>) -> tensor<4x?xf16> {
    // CHECK: return %arg0
  %0 = "tf.Sum"(%arg0, %arg1) { keep_dims = false }: (tensor<4x?xf16>, tensor<*xi64>) -> tensor<4x?xf16>
  func.return %0 : tensor<4x?xf16>
}

// CHECK-LABEL: @testMatrixDiag
func.func @testMatrixDiag(%diag: tensor<2x4xf32>) -> tensor<2x4x4xf32> {
  // CHECK-DAG: %[[MINUS1:.*]] = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[ZEROI:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[ZEROF:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: "tf.MatrixDiagV3"(%arg0, %[[ZEROI]], %[[MINUS1]], %[[MINUS1]], %[[ZEROF]]) {align = "RIGHT_LEFT"} : (tensor<2x4xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>) -> tensor<2x4x4xf32>
  %0 = "tf.MatrixDiag"(%diag) : (tensor<2x4xf32>) -> tensor<2x4x4xf32>
  func.return %0 : tensor<2x4x4xf32>
}

// CHECK-LABEL: @testMatrixSetDiag
func.func @testMatrixSetDiag(%arg0: tensor<3x3xi64>, %arg1: tensor<3xi64>) -> tensor<3x3xi64> {
  %0 = "tf.MatrixSetDiag"(%arg0, %arg1) : (tensor<3x3xi64>, tensor<3xi64>) -> tensor<3x3xi64>
  func.return %0 : tensor<3x3xi64>

  // CHECK: %[[ZERO:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>}
  // CHECK: %[[RES:.*]] = "tf.MatrixSetDiagV3"(%arg0, %arg1, %[[ZERO]])
  // CHECK-SAME: {align = "RIGHT_LEFT"}
  // CHECK-SAME: (tensor<3x3xi64>, tensor<3xi64>, tensor<i32>) -> tensor<3x3xi64>
}

// CHECK-LABEL: @testMatrixSetDiagV2
func.func @testMatrixSetDiagV2(%arg0: tensor<3x3xi64>, %arg1: tensor<3xi64>, %arg2: tensor<i32>) -> tensor<3x3xi64> {
  %0 = "tf.MatrixSetDiagV2"(%arg0, %arg1, %arg2) : (tensor<3x3xi64>, tensor<3xi64>, tensor<i32>) -> tensor<3x3xi64>
  func.return %0 : tensor<3x3xi64>

  // CHECK: %[[RES:.*]] = "tf.MatrixSetDiagV3"(%arg0, %arg1, %arg2)
  // CHECK-SAME: {align = "LEFT_LEFT"}
}

// CHECK-LABEL: @testVariableToVariableV2
func.func @testVariableToVariableV2() {
  // CHECK-NOT: "tf.Variable"

  %0 = "tf.Const"() { value = dense<1> : tensor<i32> } : () -> tensor<i32>
  // CHECK: "tf.VariableV2"
  %1 = "tf.Variable"() {container = "", dtype = i32, shared_name = "var", shape = #tf_type.shape<>} : () -> tensor<!tf_type.int32ref>
  %2 = "tf.Assign"(%1, %0) : (tensor<!tf_type.int32ref>, tensor<i32>) -> (tensor<!tf_type.int32ref>)

  func.return
}

// CHECK-LABEL: testUnpackAndCwiseUnary
func.func @testUnpackAndCwiseUnary(%arg0: tensor<?x2xf32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // CHECK: %[[NEG:.*]] = "tf.Neg"(%arg0)
  // CHECK: %[[UNPACK:.*]]:2 = "tf.Unpack"(%[[NEG]])
  %unpacked:2 = "tf.Unpack"(%arg0) {axis = 1 : i64, device = ""}
                : (tensor<?x2xf32>) -> (tensor<?xf32>, tensor<?xf32>)
  %0 = "tf.Neg"(%unpacked#0): (tensor<?xf32>) -> tensor<?xf32>
  %1 = "tf.Neg"(%unpacked#1): (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return %[[UNPACK]]#0, %[[UNPACK]]#1
  func.return %0, %1 : tensor<?xf32>, tensor<?xf32>
}

// CHECK-LABEL: testFoldStridedSliceShapeI32
func.func @testFoldStridedSliceShapeI32(%arg0: tensor<?x1x2x?xf32>) -> (tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %3 : tensor<2xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeI64
func.func @testFoldStridedSliceShapeI64(%arg0: tensor<?x1x2x?xf32>) -> (tensor<2xi64>) {
  %0 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi64>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi64>
  func.return %3 : tensor<2xi64>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeDynamicOutput
func.func @testFoldStridedSliceShapeDynamicOutput(%arg0: tensor<?x1x2x?xf32>) -> (tensor<?xi32>) {
  %0 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  func.return %3 : tensor<?xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithShrinkAxisMaskI32
func.func @testFoldStridedSliceShapeWithShrinkAxisMaskI32(%arg0: tensor<?x1x2x?xf32>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %3 : tensor<i32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithShrinkAxisMaskI64
func.func @testFoldStridedSliceShapeWithShrinkAxisMaskI64(%arg0: tensor<?x1x2x?xf32>) -> (tensor<i64>) {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi64>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  func.return %3 : tensor<i64>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithShrinkAxisMaskUnrankedOutput
func.func @testFoldStridedSliceShapeWithShrinkAxisMaskUnrankedOutput(%arg0: tensor<?x1x2x?xf32>) -> (tensor<*xi32>) {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32>
  func.return %3 : tensor<*xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<*xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithShrinkAxisMaskNegativeBegin1
func.func @testFoldStridedSliceShapeWithShrinkAxisMaskNegativeBegin1(%arg0: tensor<?x1x2x3xf32>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %4 : tensor<i32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithShrinkAxisMaskNegativeBegin2
func.func @testFoldStridedSliceShapeWithShrinkAxisMaskNegativeBegin2(%arg0: tensor<?x1x2x3xf32>) -> (tensor<i32>) {
  %0 = "tf.Const"() {value = dense<-2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %4 : tensor<i32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testUnfoldedStridedSliceShape
func.func @testUnfoldedStridedSliceShape(%arg0: tensor<?x1x2x?xf32>) -> (tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<?x1x2x?xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %4 : tensor<2xi32>
  // CHECK: %[[SLICE:.*]] = "tf.StridedSlice"
  // CHECK: return %[[SLICE]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithBeginMask
func.func @testFoldStridedSliceShapeWithBeginMask(%arg0: tensor<1x2x3x?xf32>) -> (tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<1x2x3x?xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 1 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %4 : tensor<2xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithEndMask
func.func @testFoldStridedSliceShapeWithEndMask(%arg0: tensor<?x1x2x3xf32>) -> (tensor<3xi32>) {
  %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %3 : tensor<3xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithPositiveStrides
func.func @testFoldStridedSliceShapeWithPositiveStrides(%arg0: tensor<1x2x3x4x?xf32>) -> (tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<1x2x3x4x?xf32>) -> tensor<5xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<5xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %4 : tensor<2xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[2, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithPositiveStridesOutOfBoundEnd
func.func @testFoldStridedSliceShapeWithPositiveStridesOutOfBoundEnd(%arg0: tensor<?x1x2x3xf32>) -> (tensor<3xi32>) {
  %0 = "tf.Const"() {value = dense<20> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %3 = "tf.StridedSlice"(%2, %1, %0, %1) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %3 : tensor<3xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithNegativeStrides
func.func @testFoldStridedSliceShapeWithNegativeStrides(%arg0: tensor<1x2x3x?xf32>) -> (tensor<1xi32>) {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<1x2x3x?xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %4 : tensor<1xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithNegativeStridesOutOfBoundBegin
func.func @testFoldStridedSliceShapeWithNegativeStridesOutOfBoundBegin(%arg0: tensor<?x1x2x3xf32>) -> (tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<20> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %4 : tensor<2xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[3, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithNegativeStridesBeginMask
func.func @testFoldStridedSliceShapeWithNegativeStridesBeginMask(%arg0: tensor<?x1x2x3xf32>) -> (tensor<2xi32>) {
  %0 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 1 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  func.return %4 : tensor<2xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[3, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithNegativeStridesEndMask
func.func @testFoldStridedSliceShapeWithNegativeStridesEndMask(%arg0: tensor<1x2x3x?xf32>) -> (tensor<3xi32>) {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<1x2x3x?xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %4 : tensor<3xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<[3, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldStridedSliceShapeWithEmptySlice
func.func @testFoldStridedSliceShapeWithEmptySlice(%arg0: tensor<?x1x2x3xf32>) -> (tensor<0xi32>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Shape"(%arg0) : (tensor<?x1x2x3xf32>) -> tensor<4xi32>
  %4 = "tf.StridedSlice"(%3, %0, %1, %2) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  func.return %4 : tensor<0xi32>
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: testFoldEnsureShapeOp
func.func @testFoldEnsureShapeOp(%arg0: tensor<10x20xf32>) -> (tensor<10x20xf32>, tensor<10x20xf32>, tensor<20x10xf32>) {
  %0 = "tf.EnsureShape"(%arg0) {shape = #tf_type.shape<10x20>} : (tensor<10x20xf32>) -> tensor<10x20xf32>
  %1 = "tf.EnsureShape"(%arg0) {shape = #tf_type.shape<?x20>} : (tensor<10x20xf32>) -> tensor<10x20xf32>
  // Failing case which should not be folded.
  // CHECK: %[[NF:.*]] = "tf.EnsureShape"(%arg0) {shape = #tf_type.shape<20x10>}
  %2 = "tf.EnsureShape"(%arg0) {shape = #tf_type.shape<20x10>} : (tensor<10x20xf32>) -> tensor<20x10xf32>
  // CHECK: return %arg0, %arg0, %[[NF]]
  func.return %0, %1, %2: tensor<10x20xf32>, tensor<10x20xf32>, tensor<20x10xf32>
}

// CHECK-LABEL: testConvertPackToReshapeAxis0
func.func @testConvertPackToReshapeAxis0(%arg0: tensor<2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = "tf.Pack"(%arg0) {axis = 0 : i64, _xla_outside_compilation = "1"} : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
  // CHECK: %[[SHAPE:.*]] = "tf.Const"() {value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[SHAPE]]) {_xla_outside_compilation = "1"} : (tensor<2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
  // CHECK: return %[[RESHAPE]] : tensor<1x2x3xf32>
}

// CHECK-LABEL: testConvertPackToReshapeAxis1
func.func @testConvertPackToReshapeAxis1(%arg0: tensor<2x3xf32>) -> tensor<2x1x3xf32> {
  %0 = "tf.Pack"(%arg0) {axis = 1 : i64} : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  func.return %0 : tensor<2x1x3xf32>
  // CHECK: %[[SHAPE:.*]] = "tf.Const"() {value = dense<[2, 1, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK: %[[RESHAPE:.*]] = "tf.Reshape"(%arg0, %[[SHAPE]]) : (tensor<2x3xf32>, tensor<3xi64>) -> tensor<2x1x3xf32>
  // CHECK: return %[[RESHAPE]] : tensor<2x1x3xf32>
}

// CHECK-LABEL: testDontConvertPackToReshapeDynamicShape
func.func @testDontConvertPackToReshapeDynamicShape(%arg0: tensor<2x?xf32>) -> tensor<1x2x?xf32> {
  %0 = "tf.Pack"(%arg0) {axis = 0 : i64} : (tensor<2x?xf32>) -> tensor<1x2x?xf32>
  func.return %0 : tensor<1x2x?xf32>
  // CHECK: %[[PACK:.*]] = "tf.Pack"(%arg0) {axis = 0 : i64} : (tensor<2x?xf32>) -> tensor<1x2x?xf32>
  // CHECK: return %[[PACK]] : tensor<1x2x?xf32>
}

// CHECK-LABEL: while_with_id_passthrough
func.func @while_with_id_passthrough(%arg0: tensor<7xf32> {tf._user_specified_name = "x"}) -> tensor<?xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "x", outputs = "identity_RetVal"}} {
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SHAPE:.*]] = "tf.Const"() {value = dense<7> : tensor<1xi32>}
  %2 = "tf.Const"() {value = dense<7> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  %4 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %5 = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %6:4 = "tf.WhileRegion"(%0, %1, %arg0, %2) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<?xf32>, %arg4: tensor<1xi32>):
      %8 = "tf.Less"(%arg1, %arg2) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = "tf.LogicalAnd"(%8, %3) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "tf.Identity"(%9) {device = ""} : (tensor<i1>) -> tensor<i1>
      "tf.Yield"(%10) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<?xf32>, %arg4: tensor<1xi32>):
      %8 = "tf.Identity"(%arg4) {device = ""} : (tensor<1xi32>) -> tensor<1xi32>
      // CHECK: tf.RandomStandardNormal{{.*}}(%[[SHAPE]])
      %9 = "tf.RandomStandardNormal"(%arg4) {device = "", seed = 87654321 : i64, seed2 = 0 : i64} : (tensor<1xi32>) -> tensor<?xf32>
      %10 = "tf.Pow"(%9, %5) {device = ""} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
      %11 = "tf.AddV2"(%arg3, %10) {device = ""} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      %12 = "tf.Identity"(%11) {device = ""} : (tensor<?xf32>) -> tensor<?xf32>
      %13 = "tf.AddV2"(%arg1, %4) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %14 = "tf.Identity"(%13) {device = ""} : (tensor<i32>) -> tensor<i32>
      %15 = "tf.Identity"(%arg2) {device = ""} : (tensor<i32>) -> tensor<i32>
      "tf.Yield"(%14, %15, %12, %8) : (tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<1xi32>) -> ()
  }) {_num_original_outputs = 4 : i64, _read_only_resource_inputs = [], _xla_propagate_compile_time_consts = true, device = "", is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<7xf32>, tensor<1xi32>) -> (tensor<i32>, tensor<i32>, tensor<?xf32>, tensor<1xi32>)
  %7 = "tf.Identity"(%6#2) {device = ""} : (tensor<?xf32>) -> tensor<?xf32>
  func.return %7 : tensor<?xf32>
}

// CHECK-LABEL: testConvertQuantizeAndDequantizeV2ToQuantizeAndDequantizeV4
func.func @testConvertQuantizeAndDequantizeV2ToQuantizeAndDequantizeV4(%arg0 : tensor<?x?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?x?xf32> {
  %0 = "tf.QuantizeAndDequantizeV2"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
  // CHECK: %[[QUANT:.*]] = "tf.QuantizeAndDequantizeV4"(%arg0, %arg1, %arg2) {axis = -1 : i64, narrow_range = false, num_bits = 8 : i64, range_given = false, round_mode = "HALF_TO_EVEN", signed_input = true} : (tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK: return %[[QUANT]] : tensor<?x?xf32>
}

// CHECK-LABEL: testHashTableAndInitializeTableToV2
func.func @testHashTableAndInitializeTableToV2(%arg0: tensor<!tf_type.string>) {
  // CHECK: [[handle:%.*]] = "tf.HashTableV2"()
  // CHECK-SAME: container = ""
  // CHECK-SAME: key_dtype = !tf_type.string
  // CHECK-SAME: shared_name = "table"
  // CHECK-SAME: value_dtype = i32
  // CHECK-SAME: () -> tensor<!tf_type.resource>
  %handle = "tf.HashTable"() {container = "", device = "", shared_name = "table", key_dtype = !tf_type.string, value_dtype = i32} : () -> tensor<*x!tf_type.stringref>

  // CHECK: "tf.InitializeTableFromTextFileV2"([[handle]]
  "tf.InitializeTableFromTextFile"(%handle, %arg0) {device = "", key_index=1, value_index=1, delimiter="\t"} : (tensor<*x!tf_type.stringref>, tensor<!tf_type.string>) -> ()
  func.return
}

// CHECK-LABEL: @testHashTableAndLookupTableSizeToV2
func.func @testHashTableAndLookupTableSizeToV2() -> tensor<i64> {
  // CHECK: [[handle:%.*]] = "tf.HashTableV2"()
  // CHECK-SAME: container = ""
  // CHECK-SAME: key_dtype = !tf_type.string
  // CHECK-SAME: shared_name = "table"
  // CHECK-SAME: value_dtype = i32
  // CHECK-SAME: () -> tensor<!tf_type.resource>
  %handle = "tf.HashTable"() {container = "", device = "", shared_name = "table", key_dtype = !tf_type.string, value_dtype = i32} : () -> tensor<*x!tf_type.stringref>

  // CHECK: "tf.LookupTableSizeV2"([[handle]]
  %0 = "tf.LookupTableSize"(%handle) {} : (tensor<*x!tf_type.stringref>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// CHECK-LABEL: @testHashTableAndLookupTableFindToV2
func.func @testHashTableAndLookupTableFindToV2(%arg0: tensor<!tf_type.string>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: [[handle:%.*]] = "tf.HashTableV2"()
  // CHECK-SAME: container = ""
  // CHECK-SAME: key_dtype = !tf_type.string
  // CHECK-SAME: shared_name = "table"
  // CHECK-SAME: value_dtype = i32
  // CHECK-SAME: () -> tensor<!tf_type.resource>
  %handle = "tf.HashTable"() {container = "", device = "", shared_name = "table", key_dtype = !tf_type.string, value_dtype = i32} : () -> tensor<*x!tf_type.stringref>

  // CHECK: "tf.LookupTableFindV2"([[handle]]
  %0 = "tf.LookupTableFind"(%handle, %arg0, %arg1) {} : (tensor<*x!tf_type.stringref>, tensor<!tf_type.string>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: testDivNoNanAndMulNoNanWithConstantY
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>)
func.func @testDivNoNanAndMulNoNanWithConstantY(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  // CHECK: %[[CON1:.*]] = "tf.Const"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK-NEXT: %[[CON2:.*]] = "tf.Const"() {value = dense<[1.000000e+01, 0.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK-NEXT: %[[CON3:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK-NEXT: %[[RES1:.*]] = "tf.Div"(%[[ARG0]], %[[CON1]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[RES2:.*]] = "tf.MulNoNan"(%[[ARG0]], %[[CON2]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: return %[[RES1]], %[[RES2]], %[[CON3]] : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
  %con1 = "tf.Const"() { value = dense<[1.0, 2.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con2 = "tf.Const"() { value = dense<[10.0, 0.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con3 = "tf.Const"() { value = dense<[0.0, 0.0]> : tensor<2xf32> } : () -> tensor<2xf32>

  %res1 = "tf.DivNoNan"(%arg0, %con1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %res2 = "tf.MulNoNan"(%arg0, %con2) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %res3 = "tf.DivNoNan"(%arg0, %con3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %res1, %res2, %res3 : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: testComplexDivNoNanAndMulNoNanWithConstantY
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xcomplex<f32>>)
func.func @testComplexDivNoNanAndMulNoNanWithConstantY(%arg0: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
  // CHECK-NEXT: %[[COMP3:.*]] = "tf.Const"() {value = dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>} : () -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: %[[COMP2:.*]] = "tf.Const"() {value = dense<[(0.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00)]> : tensor<2xcomplex<f32>>} : () -> tensor<2xcomplex<f32>>
  // CHECK: %[[COMP1:.*]] = "tf.Const"() {value = dense<[(1.000000e+00,3.000000e+00), (2.000000e+00,4.000000e+00)]> : tensor<2xcomplex<f32>>} : () -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: %[[RES1:.*]] = "tf.Mul"(%[[ARG0]], %[[COMP1]]) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: %[[RES2:.*]] = "tf.DivNoNan"(%[[ARG0]], %[[COMP2]]) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: return %[[RES1]], %[[RES2]], %[[COMP3]] : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
  %con11 = "tf.Const"() { value = dense<[1.0, 2.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con12 = "tf.Const"() { value = dense<[3.0, 4.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con21 = "tf.Const"() { value = dense<[0.0, 2.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con22 = "tf.Const"() { value = dense<[0.0, 0.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con31 = "tf.Const"() { value = dense<[0.0, 0.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %con32 = "tf.Const"() { value = dense<[0.0, 0.0]> : tensor<2xf32> } : () -> tensor<2xf32>

  %comp1 = "tf.Complex"(%con11, %con12) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
  %comp2 = "tf.Complex"(%con21, %con22) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
  %comp3 = "tf.Complex"(%con31, %con32) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>

  %res1 = "tf.MulNoNan"(%arg0, %comp1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  %res2 = "tf.DivNoNan"(%arg0, %comp2) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  %res3 = "tf.MulNoNan"(%arg0, %comp3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %res1, %res2, %res3 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
}

// CHECK-LABEL: testDivNoNanAndMulNoNanWithNonConstantY
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func.func @testDivNoNanAndMulNoNanWithNonConstantY(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  // CHECK: %[[NONCON2:.*]] = "tf.AddV2"(%[[ARG0]], %[[ARG1]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[NONCON3:.*]] = "tf.Square"(%[[NONCON2]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[RES1:.*]] = "tf.DivNoNan"(%[[ARG0]], %[[ARG1]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[RES2:.*]] = "tf.MulNoNan"(%[[ARG0]], %[[NONCON2]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[RES3:.*]] = "tf.DivNoNan"(%[[ARG0]], %[[NONCON3]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-NEXT: return %[[RES1]], %[[RES2]], %[[RES3]] : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
  %noncon2 = "tf.AddV2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %noncon3 = "tf.Square"(%noncon2) : (tensor<2xf32>) -> tensor<2xf32>

  %res1 = "tf.DivNoNan"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %res2 = "tf.MulNoNan"(%arg0, %noncon2) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %res3 = "tf.DivNoNan"(%arg0, %noncon3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %res1, %res2, %res3 : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: testComplexDivNoNanOpWithNonConstantY
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xcomplex<f32>>, %[[ARG1:.*]]: tensor<2xcomplex<f32>>, %[[ARG2:.*]]: tensor<2xf32>)
func.func @testComplexDivNoNanOpWithNonConstantY(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>, %arg2: tensor<2xf32>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
  // CHECK: %[[CON1:.*]] = "tf.Const"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
  // CHECK-NEXT: %[[NONCON2:.*]] = "tf.Sub"(%[[ARG0]], %[[ARG1]]) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: %[[NONCON3:.*]] = "tf.Complex"(%[[CON1]], %[[ARG2]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: %[[RES1:.*]] = "tf.MulNoNan"(%[[ARG0]], %[[ARG1]]) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>
  // CHECK-NEXT: %[[RES2:.*]] = "tf.DivNoNan"(%[[ARG0]], %[[NONCON2]]) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: %[[RES3:.*]] = "tf.MulNoNan"(%[[ARG0]], %[[NONCON3]]) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  // CHECK-NEXT: return %[[RES1]], %[[RES2]], %[[RES3]] : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
  %con1 = "tf.Const"() { value = dense<[1.0, 2.0]> : tensor<2xf32> } : () -> tensor<2xf32>
  %noncon2 = "tf.Sub"(%arg0, %arg1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  %noncon3 = "tf.Complex"(%con1, %arg2) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>

  %res1 = "tf.MulNoNan"(%arg0, %arg1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  %res2 = "tf.DivNoNan"(%arg0, %noncon2) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  %res3 = "tf.MulNoNan"(%arg0, %noncon3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %res1, %res2, %res3 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
}

// CHECK-LABEL: testXlaConvToV2
func.func @testXlaConvToV2(%lhs: tensor<8x4x16x16x16xf32>, %rhs: tensor<4x3x3x16x16xf32>) -> (tensor<8x4x14x14x16xf32>) {
  %feature_group_count = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %lhs_dilation = "tf.Const"() {value = dense<[4, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %rhs_dilation = "tf.Const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %padding = "tf.Const"() {value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %strides = "tf.Const"() {value = dense<[3, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: "tf.XlaConvV2"(%arg0, %arg1, %cst_3, %cst_2, %cst_0, %cst_1, %cst) {batch_group_count = 1 : i64, dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>, tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>
  %0 = "tf.XlaConv"(%lhs, %rhs, %strides, %padding, %lhs_dilation, %rhs_dilation, %feature_group_count) {dimension_numbers = "\18\03 \042\03\00\01\02@\04P\04Z\03\01\02\03b\03\01\02\03", precision_config = ""} : (tensor<8x4x16x16x16xf32>, tensor<4x3x3x16x16xf32>, tensor<3xi32>, tensor<3x2xi32>, tensor<3xi32>, tensor<3xi32>, tensor<i32>) -> tensor<8x4x14x14x16xf32>
  func.return %0 : tensor<8x4x14x14x16xf32>
}


// CHECK-LABEL: testXlaReduceToXlaVariadicReduceV2
func.func @testXlaReduceToXlaVariadicReduceV2(%arg0: tensor<*xbf16>, %arg1: tensor<*xbf16>) -> tensor<*xbf16> {
  // CHECK: "tf.XlaVariadicReduceV2"(%arg0, %arg1) {dimensions_to_reduce = [], operand_segment_sizes = dense<1> : vector<2xi32>, reducer = @sum1} : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
  %0 = "tf.XlaReduce"(%arg0, %arg1) {dimensions_to_reduce = [], reducer = @sum1} : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
  func.return %0 : tensor<*xbf16>
}

func.func private @sum1(%arg0: tensor<*xbf16>, %arg1: tensor<*xbf16>) -> tensor<*xbf16> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
  func.return %0 : tensor<*xbf16>
}

// CHECK-LABEL: testXlaVariadicReduceToV2
func.func @testXlaVariadicReduceToV2(%arg0: tensor<3x4xf32>, %arg1: tensor<f32>) -> tensor<?x?xf32> {
  // CHECK:  "tf.XlaVariadicReduceV2"(%arg0, %arg1) {dimensions_to_reduce = [], operand_segment_sizes = dense<1> : vector<2xi32>, reducer = @sum2} : (tensor<3x4xf32>, tensor<f32>) -> tensor<?x?xf32>
  %0 = "tf.XlaVariadicReduce"(%arg0, %arg1) {dimensions_to_reduce = [], reducer = @sum2} : (tensor<3x4xf32>, tensor<f32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
func.func private @sum2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: testMaximumOfZeroToReluFloat
func.func @testMaximumOfZeroToReluFloat(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %0 = "tf.Relu"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %0
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
  %0 = "tf.Maximum"(%arg0, %cst_0) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: testMaximumOfZeroToReluInt
func.func @testMaximumOfZeroToReluInt(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %0 = "tf.Relu"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %0
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %0 = "tf.Maximum"(%arg0, %cst_0) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: testMaximumOfZeroToReluInt32OnGpu
func.func @testMaximumOfZeroToReluInt32OnGpu(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<i32>
  // CHECK: %[[RESULT:.*]] = "tf.Maximum"(%arg0, %[[CST]]) {device = "/job:localhost/replica:0/task:0/device:GPU:0", dtype = i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK: return %[[RESULT]]
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %0 = "tf.Maximum"(%arg0, %cst_0) {device = "/job:localhost/replica:0/task:0/device:GPU:0", dtype = i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: testMaximumOfZeroToReluInt32OnCpu
func.func @testMaximumOfZeroToReluInt32OnCpu(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[RESULT:.*]] = "tf.Relu"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %[[RESULT]]
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %0 = "tf.Maximum"(%arg0, %cst_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: testMaximumOfZeroToReluInt64OnGpu
func.func @testMaximumOfZeroToReluInt64OnGpu(%arg0: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: %[[RESULT:.*]] = "tf.Relu"(%arg0) : (tensor<4xi64>) -> tensor<4xi64>
  // CHECK: return %[[RESULT]]
  %cst_0 = arith.constant dense<0> : tensor<i64>
  %0 = "tf.Maximum"(%arg0, %cst_0) {device = "/job:localhost/replica:0/task:0/device:GPU:0"} : (tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// CHECK-LABEL: testReluOfMinimum6ToRelu6Float
func.func @testReluOfMinimum6ToRelu6Float(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %0 = "tf.Relu6"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: return %0
  %cst_6 = arith.constant dense<6.000000e+00> : tensor<f32>
  %0 = "tf.Minimum"(%arg0, %cst_6) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = "tf.Relu"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: testReluOfMinimum6ToRelu6Int
func.func @testReluOfMinimum6ToRelu6Int(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %0 = "tf.Relu6"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: return %0
  %cst_6 = arith.constant dense<6> : tensor<i32>
  %0 = "tf.Minimum"(%arg0, %cst_6) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %1 = "tf.Relu"(%0) : (tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: testTensorListGetItem
func.func @testTensorListGetItem(%arg0: tensor<1600x1x32xf32>, %arg1: tensor<2xi32>, %arg2: tensor<i32>) -> tensor<1x32xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<1600x1x32xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<1x32xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf_type.variant<tensor<1x32xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<1x32xf32>
  func.return %1 : tensor<1x32xf32>

  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[RES:.*]] = "tf.GatherV2"(%arg0, %arg2, %cst) {batch_dims = 0 : i64} : (tensor<1600x1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
}

// CHECK-LABEL: testTensorListGetItemMultipleUsers
func.func @testTensorListGetItemMultipleUsers(%arg0: tensor<1600x1x32xf32>, %arg1: tensor<2xi32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<1x32xf32>, tensor<1x32xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<1600x1x32xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<1x32xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf_type.variant<tensor<1x32xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<1x32xf32>
  %2 = "tf.TensorListGetItem"(%0, %arg3, %arg1) : (tensor<!tf_type.variant<tensor<1x32xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<1x32xf32>
  func.return %1, %2 : tensor<1x32xf32>, tensor<1x32xf32>

  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[RES0:.*]] = "tf.GatherV2"(%arg0, %arg2, %cst) {batch_dims = 0 : i64} : (tensor<1600x1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
  // CHECK: %[[RES1:.*]] = "tf.GatherV2"(%arg0, %arg3, %cst) {batch_dims = 0 : i64} : (tensor<1600x1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
}
