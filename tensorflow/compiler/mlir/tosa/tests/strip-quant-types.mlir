// RUN: tf-opt --split-input-file --tosa-strip-quant-types  --verify-each %s | FileCheck %s

// -----

// CHECK-LABEL: @test_max_pool2d_qi8
// CHECK-SAME: %arg0: tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
func.func @test_max_pool2d_qi8(%arg0: tensor<1x4x4x4x!quant.uniform<i8:f32, 0.1:1>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 0.1:2>> {
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4x!quant.uniform<i8:f32, 0.1:1>>) -> tensor<1x4x4x4x!quant.uniform<i8:f32, 0.1:2>>

  // CHECK: %[[VAR0:.+]] = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x4xi8>) -> tensor<1x4x4x4xi8>
  // CHECK: return %[[VAR0]] : tensor<1x4x4x4xi8>
  func.return %0 : tensor<1x4x4x4x!quant.uniform<i8:f32, 0.1:2>>
}

// -----

// CHECK-LABEL: @test_bitwise_not_qu8
// CHECK-SAME: %arg0: tensor<ui8>) -> tensor<ui8>
func.func @test_bitwise_not_qu8(%arg0: tensor<!quant.uniform<u8:f32, 0.1:1>>) -> tensor<!quant.uniform<u8:f32, 0.1:1>> {
  %0 = "tosa.bitwise_not"(%arg0) : (tensor<!quant.uniform<u8:f32, 0.1:1>>) -> tensor<!quant.uniform<u8:f32, 0.1:1>>

  // CHECK: %[[VAR0:.+]] = tosa.bitwise_not %arg0 : (tensor<ui8>) -> tensor<ui8>
  // CHECK: return %[[VAR0]] : tensor<ui8>
  func.return %0 : tensor<!quant.uniform<u8:f32, 0.1:1>>
}
