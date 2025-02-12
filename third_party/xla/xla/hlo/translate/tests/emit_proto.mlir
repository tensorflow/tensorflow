// RUN: hlo-translate -mlir-to-hlo -emit-proto %s | FileCheck %s

// CHECK: name: "foobar
// CHECK: entry_computation_name: "main
// CHECK: computations {
// CHECK: name: "main
// CHECK: instructions {
// CHECK: name: "Arg_
// CHECK: opcode: "parameter"
// CHECK: name: "add
// CHECK: opcode: "add"
// CHECK: name: "dot
// CHECK: opcode: "dot"
module @foobar {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = stablehlo.dot %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
}
