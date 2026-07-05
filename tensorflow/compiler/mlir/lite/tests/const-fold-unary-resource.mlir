// RUN: litert-opt %s -canonicalize | FILECHECK_OPTS="" FileCheck %s
// RUN: litert-opt %s --tfl-dense-to-dense-resource-elements -canonicalize | litert-opt --tfl-dense-resource-to-dense-elements | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: @test_logical_not_i1
func.func @test_logical_not_i1() -> tensor<4xi1> {
  // CHECK: arith.constant dense<[false, true, false, true]> : tensor<4xi1>
  %0 = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
  %1 = "tfl.logical_not"(%0) : (tensor<4xi1>) -> tensor<4xi1>
  func.return %1 : tensor<4xi1>
}

// CHECK-LABEL: @test_relu_i32
func.func @test_relu_i32() -> tensor<4xi32> {
  // CHECK: arith.constant dense<[0, 2, 0, 4]> : tensor<4xi32>
  %0 = arith.constant dense<[-1, 2, -3, 4]> : tensor<4xi32>
  %1 = "tfl.relu"(%0) : (tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

// CHECK-LABEL: @test_relu_f32
func.func @test_relu_f32() -> tensor<4xf32> {
  // CHECK: arith.constant dense<[0.000000e+00, 2.000000e+00, 0.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %0 = arith.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %1 = "tfl.relu"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: @test_abs_f32
func.func @test_abs_f32() -> tensor<4xf32> {
  // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %0 = arith.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %1 = "tfl.abs"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: @test_ceil_f32
func.func @test_ceil_f32() -> tensor<4xf32> {
  // CHECK: arith.constant dense<[-1.000000e+00, 2.000000e+00, -3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %0 = arith.constant dense<[-1.5, 1.5, -3.5, 3.5]> : tensor<4xf32>
  %1 = "tfl.ceil"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: @test_floor_f32
func.func @test_floor_f32() -> tensor<4xf32> {
  // CHECK: arith.constant dense<[-2.000000e+00, 1.000000e+00, -4.000000e+00, 3.000000e+00]> : tensor<4xf32>
  %0 = arith.constant dense<[-1.5, 1.5, -3.5, 3.5]> : tensor<4xf32>
  %1 = "tfl.floor"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: @test_neg_f32
func.func @test_neg_f32() -> tensor<4xf32> {
  // CHECK: arith.constant dense<[1.000000e+00, -2.000000e+00, 3.000000e+00, -4.000000e+00]> : tensor<4xf32>
  %0 = arith.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %1 = "tfl.neg"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}

// CHECK-LABEL: @test_sin_f32
func.func @test_sin_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<0.000000e+00> : tensor<1xf32>
  %0 = arith.constant dense<[0.0]> : tensor<1xf32>
  %1 = "tfl.sin"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_cos_f32
func.func @test_cos_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<1.000000e+00> : tensor<1xf32>
  %0 = arith.constant dense<[0.0]> : tensor<1xf32>
  %1 = "tfl.cos"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_log_f32
func.func @test_log_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<0.000000e+00> : tensor<1xf32>
  %0 = arith.constant dense<[1.0]> : tensor<1xf32>
  %1 = "tfl.log"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_sqrt_f32
func.func @test_sqrt_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<2.000000e+00> : tensor<1xf32>
  %0 = arith.constant dense<[4.0]> : tensor<1xf32>
  %1 = "tfl.sqrt"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_rsqrt_f32
func.func @test_rsqrt_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<5.000000e-01> : tensor<1xf32>
  %0 = arith.constant dense<[4.0]> : tensor<1xf32>
  %1 = "tfl.rsqrt"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_square_f32
func.func @test_square_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<1.600000e+01> : tensor<1xf32>
  %0 = arith.constant dense<[4.0]> : tensor<1xf32>
  %1 = "tfl.square"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_tanh_f32
func.func @test_tanh_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<0.000000e+00> : tensor<1xf32>
  %0 = arith.constant dense<[0.0]> : tensor<1xf32>
  %1 = "tfl.tanh"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}

// CHECK-LABEL: @test_exp_f32
func.func @test_exp_f32() -> tensor<1xf32> {
  // CHECK: arith.constant dense<1.000000e+00> : tensor<1xf32>
  %0 = arith.constant dense<[0.0]> : tensor<1xf32>
  %1 = "tfl.exp"(%0) : (tensor<1xf32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}
