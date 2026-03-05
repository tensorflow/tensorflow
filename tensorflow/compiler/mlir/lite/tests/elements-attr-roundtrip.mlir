// RUN: litert-opt %s --tfl-dense-to-dense-resource-elements | FILECHECK_OPTS="" FileCheck %s
// RUN: litert-opt %s --tfl-dense-to-dense-resource-elements | litert-opt --tfl-dense-resource-to-dense-elements | FILECHECK_OPTS="" FileCheck %s --check-prefix=ROUNDTRIP

// CHECK-LABEL: @test_f32
// ROUNDTRIP-LABEL: @test_f32
func.func @test_f32() -> tensor<4xf32> {
  // CHECK: arith.constant dense_resource<dense_elements_f32> : tensor<4xf32>
  // ROUNDTRIP: arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @test_i32
// ROUNDTRIP-LABEL: @test_i32
func.func @test_i32() -> tensor<4xi32> {
  // CHECK: arith.constant dense_resource<dense_elements_i32> : tensor<4xi32>
  // ROUNDTRIP: arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @test_i1
// ROUNDTRIP-LABEL: @test_i1
func.func @test_i1() -> tensor<4xi1> {
  // CHECK: arith.constant dense_resource<dense_elements_i1> : tensor<4xi1>
  // ROUNDTRIP: arith.constant dense<[true, false, true, false]> : tensor<4xi1>
  %0 = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// CHECK-LABEL: @test_i64
// ROUNDTRIP-LABEL: @test_i64
func.func @test_i64() -> tensor<2xi64> {
  // CHECK: arith.constant dense_resource<dense_elements_i64> : tensor<2xi64>
  // ROUNDTRIP: arith.constant dense<[9223372036854775807, -9223372036854775808]> : tensor<2xi64>
  %0 = arith.constant dense<[9223372036854775807, -9223372036854775808]> : tensor<2xi64>
  func.return %0 : tensor<2xi64>
}

// CHECK-LABEL: @test_f64
// ROUNDTRIP-LABEL: @test_f64
func.func @test_f64() -> tensor<2xf64> {
  // CHECK: arith.constant dense_resource<dense_elements_f64> : tensor<2xf64>
  // ROUNDTRIP: arith.constant dense<[1.234567, 0.89101112000000005]> : tensor<2xf64>
  %0 = arith.constant dense<[1.234567, 0.89101112]> : tensor<2xf64>
  func.return %0 : tensor<2xf64>
}
