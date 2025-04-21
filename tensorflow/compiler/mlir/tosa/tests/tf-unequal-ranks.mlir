// RUN: tf-opt --split-input-file --tf-to-tosa-pipeline --verify-each %s | FileCheck %s
// REQUIRES: tf_tosa
// Test tf legalization that produce TOSA ResultsBroadcastableShape operators with unequal ranks

// -----

// CHECK-LABEL: test_add
func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<*xf32> {
  // CHECK: tosa.add
  %2 = "tf.Add"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_add
func.func @test_addn(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>, %arg2: tensor<21x3xf32>, %arg3: tensor<3xf32>) -> tensor<*xf32> {
  // CHECK: tosa.add
  // CHECK: tosa.add
  // CHECK: tosa.add
  %2 = "tf.AddN"(%arg0, %arg1, %arg2, %arg3)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>, tensor<21x3xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_bitwise_and
func.func @test_bitwise_and(%arg0: tensor<8x13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32> {
  // CHECK: tosa.bitwise_and
  %2 = "tf.BitwiseAnd"(%arg0, %arg1)   : (tensor<8x13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32>
  func.return %2 : tensor<8x13x21x3xi32>
}

// -----

// CHECK-LABEL: test_sub
func.func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<*xf32> {
  // CHECK: tosa.sub
  %2 = "tf.Sub"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_bitwise_or
func.func @test_bitwise_or(%arg0: tensor<8x13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32> {
  // CHECK: tosa.bitwise_or
  %2 = "tf.BitwiseOr"(%arg0, %arg1)   : (tensor<8x13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32>
  func.return %2 : tensor<8x13x21x3xi32>
}

// -----

// CHECK-LABEL: test_bitwise_xor
func.func @test_bitwise_xor(%arg0: tensor<8x13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32> {
  // CHECK: tosa.bitwise_xor
  %2 = "tf.BitwiseXor"(%arg0, %arg1)   : (tensor<8x13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32>
  func.return %2 : tensor<8x13x21x3xi32>
}

// -----

// CHECK-LABEL: test_logical_and
func.func @test_logical_and(%arg0: tensor<8x13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1> {
  // CHECK: tosa.logical_and
  %2 = "tf.LogicalAnd"(%arg0, %arg1)   : (tensor<8x13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1>
  func.return %2 : tensor<8x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_or
func.func @test_logical_or(%arg0: tensor<8x13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1> {
  // CHECK: tosa.logical_or
  %2 = "tf.LogicalOr"(%arg0, %arg1)   : (tensor<8x13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1>
  func.return %2 : tensor<8x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_floor_div
// CHECK: tosa.intdiv
// CHECK: tosa.select
func.func @test_floor_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<1x13x1x3xi32>) -> tensor<1x13x21x3xi32> {
  %2 = "tf.FloorDiv"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<1x13x1x3xi32>) -> tensor<1x13x21x3xi32>
  func.return %2 : tensor<1x13x21x3xi32>
}

// -----

// CHECK-LABEL: test_real_div
// CHECK: tosa.intdiv
func.func @test_real_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<1x13x1x3xi32>) -> tensor<1x13x21x3xi32> {
  %2 = "tf.RealDiv"(%arg0, %arg1)   : (tensor<13x21x3xi32>, tensor<1x13x1x3xi32>) -> tensor<1x13x21x3xi32>
  func.return %2 : tensor<1x13x21x3xi32>
}

// -----

// CHECK-LABEL: test_left_shift
func.func @test_left_shift(%arg0: tensor<4x4xi32>, %arg1: tensor<1x1x1xi32>) -> tensor<1x4x4xi32> {
  // CHECK: tosa.logical_left_shift
  %0 = "tf.LeftShift"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<1x1x1xi32>) -> tensor<1x4x4xi32>
  func.return %0 : tensor<1x4x4xi32>
}

// -----

// CHECK-LABEL: test_right_shift
func.func @test_right_shift(%arg0: tensor<4x4xi32>, %arg1: tensor<1x1x1xi32>) -> tensor<1x4x4xi32> {
  // CHECK: tosa.arithmetic_right_shift
  %0 = "tf.RightShift"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<1x1x1xi32>) -> tensor<1x4x4xi32>
  func.return %0 : tensor<1x4x4xi32>
}

// -----

// CHECK-LABEL: test_max
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x21x1xf32>) -> tensor<1x13x21x3xf32> {
  // CHECK: tosa.maximum
  %2 = "tf.Maximum"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<1x13x21x1xf32>) -> tensor<1x13x21x3xf32>
  func.return %2 : tensor<1x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_min
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x21x1xf32>) -> tensor<1x13x21x3xf32> {
  // CHECK: tosa.minimum
  %2 = "tf.Minimum"(%arg0, %arg1)   : (tensor<13x21x3xf32>, tensor<1x13x21x1xf32>) -> tensor<1x13x21x3xf32>
  func.return %2 : tensor<1x13x21x3xf32>
}

// -----

// CHECK-LABEL: test_power
func.func @test_power(%arg0: tensor<8x13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32> {
  // CHECK: tosa.pow
  %2 = "tf.Pow"(%arg0, %arg1)   : (tensor<8x13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32>
  func.return %2 : tensor<8x13x21x3xi32>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: tosa.equal
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x1x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tf.Equal"(%arg0, %arg1)  {incompatible_shape_error = true}  : (tensor<13x21x3xf32>, tensor<1x13x1x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: tosa.greater_equal
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tf.GreaterEqual"(%arg0, %arg1)   : (tensor<13x1x3xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: tosa.greater
func.func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tf.Greater"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK: tosa.greater_equal
// CHECK: tosa.logical_not
func.func @test_less(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tf.Less"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----
// CHECK-LABEL: test_select
// CHECK: tosa.select
func.func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x21x3xf32>, %arg2: tensor<1xi1>) -> tensor<1x13x21x3xf32> {
  %2 = "tf.SelectV2"(%arg2, %arg0, %arg1)   : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xf32>
  func.return %2 : tensor<1x13x21x3xf32>
}
