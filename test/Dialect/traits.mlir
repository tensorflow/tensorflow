// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: @broadcast_scalar_scalar_scalar
func @broadcast_scalar_scalar_scalar(tensor<i32>, tensor<i32>) -> tensor<i32> {
^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @broadcast_tensor_scalar_tensor
func @broadcast_tensor_scalar_tensor(tensor<4xi32>, tensor<i32>) -> tensor<4xi32> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<i32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Check only one dimension has size 1
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x1xi32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check multiple dimensions have size 1
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32> {
^bb0(%arg0: tensor<8x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  return %0 : tensor<8x7x6x5xi32>
}

// -----

// Check leading unknown dimension
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32> {
^bb0(%arg0: tensor<?x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32>
  return %0 : tensor<?x7x6x5xi32>
}

// -----

// Check unknown dimension in the middle
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32> {
^bb0(%arg0: tensor<8x1x?x1xi32>, %arg1: tensor<7x1x5xi32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32>
  return %0 : tensor<8x7x?x5xi32>
}

// -----

// Check incompatible vector and tensor result type
func @broadcast_scalar_vector_vector(tensor<4xf32>, tensor<4xf32>) -> vector<4xf32> {
^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
  // expected-error @+1 {{cannot broadcast vector with tensor}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Check incompatible operand types with known dimension
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x3xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x3xi32>):
  // expected-error @+1 {{operands don't have broadcast-compatible shapes}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x3xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check incompatible result type with known dimension
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x3xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x1xi32>):
  // expected-error @+1 {{does not have the same shape as the one computed}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x3xi32>
  return %0 : tensor<4x3x3xi32>
}

// -----

// Check incompatible result type with known dimension
func @broadcast_tensor_tensor_tensor(tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x1xi32> {
^bb0(%arg0: tensor<8x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // expected-error @+1 {{does not have the same shape as the one computed}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x1xi32>
  return %0 : tensor<8x7x6x1xi32>
}

// -----

func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<?xi32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check incompatible result type with unknown dimension
func @broadcast_tensor_tensor_tensor(tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32> {
^bb0(%arg0: tensor<?x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // expected-error @+1 {{does not have the same shape as the one computed}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  return %0 : tensor<8x7x6x5xi32>
}

// -----

// Check unranked operand but ranked result
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<*xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<*xi32>):
  // expected-error @+1 {{broadcast unranked tensor should result in unranked tensor}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<*xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// CHECK-LABEL: func @broadcastDifferentResultType
func @broadcastDifferentResultType(tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
  // CHECK: %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
