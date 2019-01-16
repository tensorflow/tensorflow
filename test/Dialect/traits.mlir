// RUN: mlir-opt %s -split-input-file -verify | FileCheck %s

// -----

// CHECK-LABEL: @broadcast_scalar_scalar_scalar
func @broadcast_scalar_scalar_scalar(tensor<i32>, tensor<i32>) -> tensor<i32> {
^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
  // CHECK: %0 = tfl.add %arg0, %arg1 {fused_activation_function: "RELU6"} : tensor<i32>
  %0 = tfl.add %arg0, %arg1 {fused_activation_function: "RELU6"} : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: @broadcast_tensor_scalar_tensor
func @broadcast_tensor_scalar_tensor(tensor<4xi32>, tensor<i32>) -> tensor<4xi32> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<i32>):
  // CHECK: %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Check only one dimension has size 1
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x1xi32>):
  // CHECK: %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32>
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check multiple dimensions have size 1
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32> {
^bb0(%arg0: tensor<8x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // CHECK: %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  return %0 : tensor<8x7x6x5xi32>
}

// -----

// Check leading unknown dimension
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32> {
^bb0(%arg0: tensor<?x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // CHECK: %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32>
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32>
  return %0 : tensor<?x7x6x5xi32>
}

// -----

// Check unknown dimension in the middle
// CHECK-LABEL: @broadcast_tensor_tensor_tensor
func @broadcast_tensor_tensor_tensor(tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32> {
^bb0(%arg0: tensor<8x1x?x1xi32>, %arg1: tensor<7x1x5xi32>):
  // CHECK: %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32>
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32>
  return %0 : tensor<8x7x?x5xi32>
}

// -----

// Check incompatible operand types with scalar types
func @broadcast_scalar_scalar_scalar(tensor<i32>, tensor<f32>) -> tensor<i32> {
^bb0(%arg0: tensor<i32>, %arg1: tensor<f32>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<i32>, tensor<f32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Check incompatible operand types with scalar types
func @broadcast_scalar_vector_vector(tensor<i32>, tensor<vector<4xf32>>) -> tensor<vector<4xf32>> {
^bb0(%arg0: tensor<i32>, %arg1: tensor<vector<4xf32>>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<i32>, tensor<vector<4xf32>>) -> tensor<vector<4xf32>>
  return %0 : tensor<vector<4xf32>>
}

// -----

// Check incompatible vector and tensor operand types
func @broadcast_scalar_vector_vector(tensor<4xf32>, tensor<vector<4xf32>>) -> tensor<vector<4xf32>> {
^bb0(%arg0: tensor<4xf32>, %arg1: tensor<vector<4xf32>>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<4xf32>, tensor<vector<4xf32>>) -> tensor<vector<4xf32>>
  return %0 : tensor<vector<4xf32>>
}

// -----

// Check incompatible vector and tensor result type
func @broadcast_scalar_vector_vector(tensor<4xf32>, tensor<4xf32>) -> vector<4xf32> {
^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
  // expected-error @+1 {{result type is not broadcast-compatible with operand types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<4xf32>, tensor<4xf32>) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Check incompatible operand types with known dimension
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x3xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x3xi32>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4x3x2xi32>, tensor<3x3xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check incompatible result type with known dimension
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x3xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x1xi32>):
  // expected-error @+1 {{result type is not broadcast-compatible with operand types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x3xi32>
  return %0 : tensor<4x3x3xi32>
}

// -----

// Check incompatible result type with known dimension
func @broadcast_tensor_tensor_tensor(tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x1xi32> {
^bb0(%arg0: tensor<8x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // expected-error @+1 {{result type is not broadcast-compatible with operand types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x1xi32>
  return %0 : tensor<8x7x6x1xi32>
}

// -----

// Check incompatible operand types with unknown dimension
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<?xi32>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check incompatible result type with unknown dimension
func @broadcast_tensor_tensor_tensor(tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32> {
^bb0(%arg0: tensor<?x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // expected-error @+1 {{result type is not broadcast-compatible with operand types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "Relu6"} : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  return %0 : tensor<8x7x6x5xi32>
}

// -----

// Check tensor of vector
func @broadcast_tensor_tensor_tensor(tensor<vector<4xi32>>, tensor<vector<4xi32>>) -> tensor<vector<4xi32>> {
^bb0(%arg0: tensor<vector<4xi32>>, %arg1: tensor<vector<4xi32>>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<vector<4xi32>>, tensor<vector<4xi32>>) -> tensor<vector<4xi32>>
  return %0 : tensor<vector<4xi32>>
}

// -----

// Check unranked types
func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<*xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<*xi32>):
  // expected-error @+1 {{operands don't have broadcast-compatible types}}
  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function: "RELU6"} : (tensor<4x3x2xi32>, tensor<*xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}
