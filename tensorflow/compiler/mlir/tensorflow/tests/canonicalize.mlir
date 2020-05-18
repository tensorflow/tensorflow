// RUN: tf-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s -dump-input-on-failure

// CHECK-LABEL: func @tfAssertTrue
func @tfAssertTrue(%arg0: tensor<1x1x6x2xf32>) {
  %t = constant dense<true> : tensor<i1>
  // CHECK-NOT: tf.Assert
  "tf.Assert"(%t, %arg0) {summarize = 3} : (tensor<i1>, tensor<1x1x6x2xf32>) -> ()
  return
}

// CHECK-LABEL: func @tfAssertFalse
func @tfAssertFalse(%arg0: tensor<1x1x6x2xf32>) {
  %f = constant dense<false> : tensor<i1>
  // CHECK: tf.Assert
  "tf.Assert"(%f, %arg0) {summarize = 3} : (tensor<i1>, tensor<1x1x6x2xf32>) -> ()
  return
}

// CHECK-LABEL: testBatchMatMulToMatMul
func @testBatchMatMulToMatMul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.BatchMatMul"(%arg0, %arg1) {adj_x = false, adj_y = false} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %0: tensor<2x2xf32>

// CHECK: %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
// CHECK: return %0
}

// CHECK-LABEL: testBatchMatMulV2ToMatMul
func @testBatchMatMulV2ToMatMul(%arg0: tensor<4x3xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = false} : (tensor<4x3xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  return %0: tensor<3x5xf32>

// CHECK: %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = true, transpose_b = false} : (tensor<4x3xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
// CHECK: return %0
}

// CHECK-LABEL: func @testLeakyRelu
func @testLeakyRelu(%arg0 : tensor<16xf32>) -> (tensor<16xf32>) {
  %2 = "tf.LeakyRelu"(%arg0) {alpha = 1.0 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: return %arg0
  return %2 : tensor<16xf32>
}

// CHECK-LABEL: testSameBitcastType
func @testSameBitcastType(%arg0: tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  return %0: tensor<8x16x32x64xf32>

// CHECK: return %arg0
}

// CHECK-LABEL: testDifferentBitcastType
func @testDifferentBitcastType(%arg0: tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
  return %0: tensor<8x16x32x64xi32>

// CHECK: %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: return %0
}

// CHECK-LABEL: testDoubleBitcast
func @testDoubleBitcast(%arg0: tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64x2xi16>
  %1 = "tf.Bitcast"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64xi32>
  return %1: tensor<8x16x32x64xi32>

// CHECK: %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: return %0
}

// CHECK-LABEL: testDoubleBitcastWithDependentArg
func @testDoubleBitcastWithDependentArg(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x32x64xi32>, tensor<8x16x32x64x2xi16>) {
  %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64x2xi16>
  %1 = "tf.Bitcast"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64xi32>
  %2 = "tf.Identity"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64x2xi16>
  return %1, %2 :  tensor<8x16x32x64xi32>, tensor<8x16x32x64x2xi16>

// CHECK: %0 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64x2xi16>
// CHECK: %1 = "tf.Bitcast"(%arg0) : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: %2 = "tf.Identity"(%0) : (tensor<8x16x32x64x2xi16>) -> tensor<8x16x32x64x2xi16>
// CHECK: return %1, %2
}

// CHECK-LABEL: testSameCastType
func @testSameCastType(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x32x64xf32>, tensor<8x16x32x64xf32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  return %0, %1: tensor<8x16x32x64xf32>, tensor<8x16x32x64xf32>

// CHECK: return %arg0, %arg0
}

// CHECK-LABEL: testDifferentCastType
func @testDifferentCastType(%arg0: tensor<8x16x32x64xf32>) -> (tensor<8x16x32x64xi32>, tensor<8x16x32x64xi32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
  return %0, %1: tensor<8x16x32x64xi32>, tensor<8x16x32x64xi32>

// CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xi32>
// CHECK: return %0, %1
}

// CHECK-LABEL: testCompatibleCastType
func @testCompatibleCastType(%arg0: tensor<?xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<?xf32>) -> tensor<10xf32>
  %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<?xf32>) -> tensor<10xf32>
  return %0, %1: tensor<10xf32>, tensor<10xf32>

// CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<?xf32>) -> tensor<10xf32>
// CHECK: %1 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<?xf32>) -> tensor<10xf32>
// CHECK: return %0, %1
}

// CHECK-LABEL: testSameCastTypeAcrossBasicBlocks
func @testSameCastTypeAcrossBasicBlocks(tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32> {
^bb0(%arg0: tensor<8x16x32x64xf32>):
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  br ^bb1
^bb1:
  %1 = "tf.Cast"(%0) {Truncate = true} : (tensor<8x16x32x64xf32>) -> tensor<8x16x32x64xf32>
  br ^exit
^exit:
  return %1: tensor<8x16x32x64xf32>

// CHECK: return %arg0
}

// CHECK-LABEL: testLogOfSoftmax
func @testLogOfSoftmax(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Softmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Log"(%0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.LogSoftmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testSubOfNeg
func @testSubOfNeg(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Sub"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testSquareOfSub
func @testSquareOfSub(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Square"(%0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddToAddV2
func @testAddToAddV2(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0: tensor<8x16xf32>

// CHECK: %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testNoAddToAddV2ForStringType
func @testNoAddToAddV2ForStringType(%arg0: tensor<8x16x!tf.string>, %arg1: tensor<8x16x!tf.string>) -> tensor<8x16x!tf.string> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<8x16x!tf.string>, tensor<8x16x!tf.string>) -> tensor<8x16x!tf.string>
  return %0: tensor<8x16x!tf.string>

// CHECK: %0 = "tf.Add"(%arg0, %arg1) : (tensor<8x16x!tf.string>, tensor<8x16x!tf.string>) -> tensor<8x16x!tf.string>
// CHECK: return %0
}

// CHECK-LABEL: testAddOfNegLeft
func @testAddOfNegLeft(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Add"(%0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Sub"(%arg1, %arg0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddOfNegRight
func @testAddOfNegRight(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Add"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Sub"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddV2OfNegLeft
func @testAddV2OfNegLeft(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.AddV2"(%0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>
// CHECK: %0 = "tf.Sub"(%arg1, %arg0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testAddV2OfNegRight
func @testAddV2OfNegRight(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Neg"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.AddV2"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Sub"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %0
}

// CHECK-LABEL: testDoubleConj
func @testDoubleConj(%arg0: tensor<8x16x32x64xcomplex<f32>>) -> tensor<8x16x32x64xcomplex<f32>> {
  %0 = "tf.Conj"(%arg0) : (tensor<8x16x32x64xcomplex<f32>>) -> tensor<8x16x32x64xcomplex<f32>>
  %1 = "tf.Conj"(%0) : (tensor<8x16x32x64xcomplex<f32>>) -> tensor<8x16x32x64xcomplex<f32>>
  return %1: tensor<8x16x32x64xcomplex<f32>>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleInvert
func @testDoubleInvert(%arg0: tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Invert"(%arg0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Invert"(%0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  return %1: tensor<8x16x32x64xi32>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleLogicalNot
func @testDoubleLogicalNot(%arg0: tensor<8x16x32x64xi1>) -> tensor<8x16x32x64xi1> {
  %0 = "tf.LogicalNot"(%arg0) : (tensor<8x16x32x64xi1>) -> tensor<8x16x32x64xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16x32x64xi1>) -> tensor<8x16x32x64xi1>
  return %1: tensor<8x16x32x64xi1>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleNeg
func @testDoubleNeg(%arg0: tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Neg"(%arg0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Neg"(%0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  return %1: tensor<8x16x32x64xi32>

// CHECK: return %arg0
}

// CHECK-LABEL: testDoubleReciprocal
func @testDoubleReciprocal(%arg0: tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32> {
  %0 = "tf.Reciprocal"(%arg0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  %1 = "tf.Reciprocal"(%0) : (tensor<8x16x32x64xi32>) -> tensor<8x16x32x64xi32>
  return %1: tensor<8x16x32x64xi32>

// CHECK: return %arg0
}

// CHECK-LABEL: testLogicalNotOfEqual
func @testLogicalNotOfEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  return %1: tensor<8x16xi1>

// CHECK: %[[NE:.*]] = "tf.NotEqual"(%arg0, %arg1) {incompatible_shape_error = true}
// CHECK: return %[[NE]]
}

// CHECK-LABEL: testLogicalNotOfNotEqual
func @testLogicalNotOfNotEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.NotEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  return %1: tensor<8x16xi1>

// CHECK: %[[NE:.*]] = "tf.Equal"(%arg0, %arg1) {incompatible_shape_error = true}
// CHECK: return %[[NE]]
}

// CHECK-LABEL: testLogicalNotOfGreater
func @testLogicalNotOfGreater(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testLogicalNotOfGreaterEqual
func @testLogicalNotOfGreaterEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.Less"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testLogicalNotOfLess
func @testLogicalNotOfLess(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Less"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testLogicalNotOfLessEqual
func @testLogicalNotOfLessEqual(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  %1 = "tf.LogicalNot"(%0) : (tensor<8x16xi1>) -> tensor<8x16xi1>
  return %1: tensor<8x16xi1>

// CHECK: %0 = "tf.Greater"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK: return %0
}

// CHECK-LABEL: testDivWithSqrtDivisor
func @testDivWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Div"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.Mul"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: testRealDivWithSqrtDivisor
func @testRealDivWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.RealDiv"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.Mul"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: testTruncateDivWithSqrtDivisor
func @testTruncateDivWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.TruncateDiv"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.Mul"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: testXdivyWithSqrtDivisor
func @testXdivyWithSqrtDivisor(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Sqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = "tf.Xdivy"(%arg0, %0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1: tensor<8x16xf32>

// CHECK: %0 = "tf.Rsqrt"(%arg1) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: %1 = "tf.MulNoNan"(%0, %arg0) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %1
}

// CHECK-LABEL: @identityTranspose
func @identityTranspose(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "tf.Const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x5x6xf32>

  return %1 : tensor<2x3x4x5x6xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: @nonIdentityTranspose
func @nonIdentityTranspose(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x3x4x6x5xf32> {
  %0 = "tf.Const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tf.Transpose"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x6x5xf32>

  return %1 : tensor<2x3x4x6x5xf32>
  // CHECK: %0 = "tf.Const"() {value = dense<[0, 1, 2, 4, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
  // CHECK: %1 = "tf.Transpose"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<5xi32>) -> tensor<2x3x4x6x5xf32>
  // CHECK: return %1
}

// CHECK-LABEL: @cancellableTranspose
func @cancellableTranspose(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Transpose"(%2, %1) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x8xf32>

  return %3 : tensor<1x4x4x8xf32>
  // CHECK: return %arg0
}

// CHECK-LABEL: @nonCancellableTranspose
func @nonCancellableTranspose(%arg0: tensor<1x4x4x8xf32>) -> tensor<4x1x4x8xf32> {
  %0 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tf.Const"() {value = dense<[2, 0, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
  %2 = "tf.Transpose"(%arg0, %0) : (tensor<1x4x4x8xf32>, tensor<4xi32>) -> tensor<1x8x4x4xf32>
  %3 = "tf.Transpose"(%2, %1) : (tensor<1x8x4x4xf32>, tensor<4xi32>) -> tensor<4x1x4x8xf32>

  return %3 : tensor<4x1x4x8xf32>
  // CHECK: return %3
}

// CHECK-LABEL: func @addN
func @addN(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: return %arg0
  %0 = "tf.AddN"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @ToBool_0DScalar
func @ToBool_0DScalar(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: return %arg0
  %0 = "tf.ToBool"(%arg0) : (tensor<i1>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: testReadVariableOpOfCast
func @testReadVariableOpOfCast(%arg0: tensor<!tf.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32> {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.resource<tensor<8x40xf32>>>) -> tensor<*x!tf.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf.resource>) -> tensor<8x40xf32>
  return %1: tensor<8x40xf32>

// CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32>
// CHECK: return %0
}

// CHECK-LABEL: testReadVariableOpOfCastWithTruncate
func @testReadVariableOpOfCastWithTruncate(%arg0: tensor<!tf.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32> {
  %0 = "tf.Cast"(%arg0) {Truncate = true} : (tensor<!tf.resource<tensor<8x40xf32>>>) -> tensor<*x!tf.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf.resource>) -> tensor<8x40xf32>
  return %1: tensor<8x40xf32>

// CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<8x40xf32>>>) -> tensor<8x40xf32>
// CHECK: return %0
}

// CHECK-LABEL: testReadVariableOpOfCastMultiUse
func @testReadVariableOpOfCastMultiUse(%arg0: tensor<!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf.resource>) -> tensor<f32>
  "tf.AssignVariableOp"(%0, %1) : (tensor<*x!tf.resource>, tensor<f32>) -> ()
  return %1: tensor<f32>

 // CHECK: %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource>
 // CHECK: %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf.resource>) -> tensor<f32>
 // CHECK: "tf.AssignVariableOp"(%0, %1) : (tensor<*x!tf.resource>, tensor<f32>) -> ()
 // CHECK: return %1
}

// CHECK-LABEL: testMultiReadVariableOpsOfCast
func @testMultiReadVariableOpsOfCast(%arg0: tensor<!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource>
  %1 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf.resource>) -> tensor<f32>
  %2 = "tf.ReadVariableOp"(%0) : (tensor<*x!tf.resource>) -> tensor<f32>
  return %2: tensor<f32>

 // CHECK: %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
 // CHECK: %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf.resource<tensor<f32>>>) -> tensor<f32>
 // CHECK: return %1
}

// CHECK-LABEL: testRankOfRankedTensor
func @testRankOfRankedTensor(%arg0 : tensor<4x3x2xf32>) -> tensor<i32> {
  // CHECK:[[VAL0:%.+]] = "tf.Const"() {value = dense<3> : tensor<i32>}
  %0 = "tf.Rank"(%arg0) : (tensor<4x3x2xf32>) -> tensor<i32>

  // CHECK: return [[VAL0]]
  return %0 : tensor<i32>
}

// CHECK-LABEL: @foldFill
func @foldFill() -> (tensor<3x2x1xf32>, tensor<*xf32>) {
  %0 = "tf.Const"() {value = dense<[3, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tf.Const"() {value = dense<23.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: "tf.Const"() {value = dense<2.300000e+01> : tensor<3x2x1xf32>}
  %2 = "tf.Fill"(%0, %1) : (tensor<3xi32>, tensor<f32>) -> tensor<3x2x1xf32>
  // CHECK: "tf.Const"() {value = dense<2.300000e+01> : tensor<3x2x1xf32>}
  %3 = "tf.Fill"(%0, %1) : (tensor<3xi32>, tensor<f32>) -> tensor<*xf32>
  return %2, %3 : tensor<3x2x1xf32>, tensor<*xf32>
}
