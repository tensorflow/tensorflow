// RUN: xla-translate -mlir-hlo-to-hlo %s | FileCheck %s

module @foobar {
func.func @main(tensor<4xf32>, tensor<4xf32>) -> tensor<f32> {
^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "mhlo.dot"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}
}
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

