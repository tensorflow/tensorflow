// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}

//CHECK:func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> attributes {tf.entry_function = {inputs = "arg0", outputs = "arg0"}} {
//CHECK: return %arg0 : tensor<4xi8>
//CHECK:}

func.func @logistic(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

// CHECK:func.func private @logistic(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
// CHECK: %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
// CHECK: return %0 : tensor<1x1x1x96xf32>
// CHECK:}

func.func @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.add %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @multiply(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @multiply(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.multiply %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @divide(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @divide(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.divide %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @maximum(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @maximum(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = stablehlo.maximum %arg0, %arg1 : tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @reshape(%arg0 : tensor<1x128xi32>) -> tensor<4x32x1xi32>{
  %0 = stablehlo.reshape %arg0 : (tensor<1x128xi32>) -> tensor<4x32x1xi32>
  func.return %0 : tensor<4x32x1xi32>
}

//CHECK:func.func private @reshape(%arg0: tensor<1x128xi32>) -> tensor<4x32x1xi32> {
//CHECK-NEXT: %0 = stablehlo.reshape %arg0 : (tensor<1x128xi32>) -> tensor<4x32x1xi32>
//CHECK-NEXT: return %0 : tensor<4x32x1xi32>
//CHECK-NEXT:}

func.func @clamp(%arg0: tensor<f32>, %arg1: tensor<1x256x256x24xf32>, %arg2: tensor<f32>) -> tensor<1x256x256x24xf32>{
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<1x256x256x24xf32>, tensor<f32>) -> tensor<1x256x256x24xf32>
  return %0 : tensor<1x256x256x24xf32>
}

//CHECK:func.func private @clamp(%arg0: tensor<f32>, %arg1: tensor<1x256x256x24xf32>, %arg2: tensor<f32>) -> tensor<1x256x256x24xf32> {
//CHECK-NEXT: %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<1x256x256x24xf32>, tensor<f32>) -> tensor<1x256x256x24xf32>
//CHECK-NEXT: return %0 : tensor<1x256x256x24xf32>
//CHECK-NEXT:}

func.func @concat(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x2xi32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x2xi32>
  func.return %0 : tensor<1x30x2xi32>
}

//CHECK:func.func private @concat(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x2xi32> {
//CHECK-NEXT: %0 = stablehlo.concatenate %arg0, %arg1, dim = 2 : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x2xi32>
//CHECK-NEXT: return %0 : tensor<1x30x2xi32>
//CHECK-NEXT:}