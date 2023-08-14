// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// test stablehlo roundtrip

func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

// CHECK:func.func @main(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> attributes {tf.entry_function = {inputs = "arg0", outputs = "stablehlo.logistic"}} {
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