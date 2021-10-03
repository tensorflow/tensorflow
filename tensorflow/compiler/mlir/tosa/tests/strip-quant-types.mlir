// RUN: tf-opt --split-input-file --tosa-strip-quant-types  --verify-each %s | FileCheck %s

// CHECK-LABEL: @test_add_qi8
// CHECK-SAME: %arg0: tensor<i8>) -> tensor<i8>
func @test_add_qi8(%arg0: tensor<!quant.uniform<i8:f32, 0.1:1>>) -> tensor<!quant.uniform<i8:f32, 0.1:2>> {
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<!quant.uniform<i8:f32, 0.1:1>>, tensor<!quant.uniform<i8:f32, 0.1:1>>) -> tensor<!quant.uniform<i8:f32, 0.1:2>>

  // CHECK: %[[VAR0:.+]] = "tosa.add"(%arg0, %arg0) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  // CHECK: return %[[VAR0]] : tensor<i8>
  return %0 : tensor<!quant.uniform<i8:f32, 0.1:2>>
}

// ----

// CHECK-LABEL: @test_add_qu8
// CHECK-SAME: %arg0: tensor<ui8>) -> tensor<ui8>
func @test_add_qu8(%arg0: tensor<!quant.uniform<u8:f32, 0.1:1>>) -> tensor<!quant.uniform<u8:f32, 0.1:2>> {
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<!quant.uniform<u8:f32, 0.1:1>>, tensor<!quant.uniform<u8:f32, 0.1:1>>) -> tensor<!quant.uniform<u8:f32, 0.1:2>>

  // CHECK: %[[VAR0:.+]] = "tosa.add"(%arg0, %arg0) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
  // CHECK: return %[[VAR0]] : tensor<ui8>
  return %0 : tensor<!quant.uniform<u8:f32, 0.1:2>>
}
