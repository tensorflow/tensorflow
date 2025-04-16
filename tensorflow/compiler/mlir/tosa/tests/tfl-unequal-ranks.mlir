// RUN: tf-opt --split-input-file --tfl-to-tosa-pipeline --verify-each %s | FileCheck %s
// REQUIRES: tf_tosa
// Test tf legalization that produce TOSA ResultsBroadcastableShape operators with unequal ranks

// -----

// CHECK-LABEL: test_add
func.func @test_add(%arg0: tensor<192x192x3xf32>, %arg1: tensor<16x192x192x3xf32>) -> tensor<16x192x192x3xf32> {
    // CHECK: tosa.add
    %1 = tfl.add(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<192x192x3xf32>, tensor<16x192x192x3xf32>) -> tensor<16x192x192x3xf32>
    func.return %1 : tensor<16x192x192x3xf32>
}

// -----

// CHECK-LABEL: test_add_qi8
func.func @test_add_qi8(%arg0: tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, %arg1: tensor<1x13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>) -> tensor<1x13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>> {
  // CHECK: tosa.add
  %0 = tfl.add(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, tensor<1x13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>) -> tensor<1x13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
  func.return %0 : tensor<1x13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
}

// -----

// CHECK-LABEL: test_sub
func.func @test_sub(%arg0: tensor<192x192x3xf32>, %arg1: tensor<16x192x192x3xf32>) -> tensor<16x192x192x3xf32> {
    // CHECK: tosa.sub
    %1 = tfl.sub(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<192x192x3xf32>, tensor<16x192x192x3xf32>) -> tensor<16x192x192x3xf32>
    func.return %1 : tensor<16x192x192x3xf32>
}

// -----

// CHECK-LABEL: test_sub_qi8
func.func @test_sub_qi8(%arg0: tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, %arg1: tensor<1x13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>) -> tensor<1x13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>> {
  // CHECK: tosa.sub
  %0 = tfl.sub(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<13x21x1x!quant.uniform<i8:f32, 0.01568480022251606:-1>>, tensor<1x13x21x3x!quant.uniform<i8:f32, 0.015686055645346642:-1>>) -> tensor<1x13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
  func.return %0 : tensor<1x13x21x3x!quant.uniform<i8:f32, 0.031318482011556625:-1>>
}

// -----

// CHECK-LABEL: test_equal
// CHECK: tosa.equal
func.func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x1x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tfl.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x13x1x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_not_equal
// CHECK: tosa.equal
// CHECK: tosa.logical_not
func.func @test_not_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x1x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tfl.not_equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x13x1x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater_equal
// CHECK: tosa.greater_equal
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tfl.greater_equal"(%arg0, %arg1)   : (tensor<13x1x3xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_greater
// CHECK: tosa.greater
func.func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tfl.greater"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less
// CHECK: tosa.greater
func.func @test_less(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tfl.less"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_less_equal
// CHECK: tosa.greater_equal
func.func @test_less_equal(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1> {
  %2 = "tfl.less_equal"(%arg0, %arg1)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xi1>
  func.return %2 : tensor<1x13x21x3xi1>
}

// -----
// CHECK-LABEL: test_select
// CHECK: tosa.select
func.func @test_select(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x21x3xf32>, %arg2: tensor<1xi1>) -> tensor<1x13x21x3xf32> {
  %2 = "tfl.select_v2"(%arg2, %arg0, %arg1)   : (tensor<1xi1>, tensor<13x21x3xf32>, tensor<1x13x21x3xf32>) -> tensor<1x13x21x3xf32>
  func.return %2 : tensor<1x13x21x3xf32>
}

// -----
// CHECK-LABEL: test_mul_qi8
// CHECK: tosa.mul
func.func @test_mul_qi8(%arg0: tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, %arg1: tensor<1x13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>> {
  %0 = "tfl.mul"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3x!quant.uniform<i8:f32, 0.015681236982345581>>, tensor<1x13x21x3x!quant.uniform<i8:f32, 0.015647144988179207:-1>>) -> tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>>
  func.return %0 : tensor<*x!quant.uniform<i8:f32, 0.0078376950696110725>>
}

// -----
// CHECK-LABEL: test_floor_div
// CHECK: tosa.int_div
// CHECK: tosa.select
func.func @test_floor_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<1x13x1x3xi32>) -> tensor<1x13x21x3xi32> {
  %0 = "tfl.floor_div"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xi32>, tensor<1x13x1x3xi32>) -> tensor<1x13x21x3xi32>
  func.return %0 : tensor<1x13x21x3xi32>
}

// -----
// CHECK-LABEL: test_div
// CHECK: tosa.int_div
func.func @test_div(%arg0: tensor<13x21x3xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "tfl.div"(%arg0, %arg1)  {fused_activation_function = "NONE"}  : (tensor<13x21x3xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: test_max
// CHECK: tosa.maximum
func.func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x21x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x13x21x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_min
// CHECK: tosa.minimum
func.func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x13x21x1xf32>) -> tensor<*xf32> {
  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x13x21x1xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_add
func.func @test_addn(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1x13x21x3xf32>, %arg2: tensor<21x3xf32>, %arg3: tensor<3xf32>) -> tensor<*xf32> {
  // CHECK: tosa.add
  // CHECK: tosa.add
  // CHECK: tosa.add
  %2 = "tfl.add_n"(%arg0, %arg1, %arg2, %arg3)   : (tensor<13x21x1xf32>, tensor<1x13x21x3xf32>, tensor<21x3xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_logical_and
func.func @test_logical_and(%arg0: tensor<8x13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1> {
  // CHECK: tosa.logical_and
  %2 = "tfl.logical_and"(%arg0, %arg1)   : (tensor<8x13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1>
  func.return %2 : tensor<8x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_logical_or
func.func @test_logical_or(%arg0: tensor<8x13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1> {
  // CHECK: tosa.logical_or
  %2 = "tfl.logical_or"(%arg0, %arg1)   : (tensor<8x13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<8x13x21x3xi1>
  func.return %2 : tensor<8x13x21x3xi1>
}

// -----

// CHECK-LABEL: test_power
func.func @test_power(%arg0: tensor<8x13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32> {
  // CHECK: tosa.pow
  %2 = "tfl.pow"(%arg0, %arg1)   : (tensor<8x13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<8x13x21x3xi32>
  func.return %2 : tensor<8x13x21x3xi32>
}
