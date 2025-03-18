// RUN: hlo-translate -mlir-to-hlo %s.bc | FileCheck %s

// File `vhlo_input.mlir.bc` is created by running the following command:
//  $ stablehlo-translate --serialize --target=1.0.0 --strip-debuginfo vhlo_input.mlir > vhlo_input.mlir.bc
//
// The `.mlir.bc` file is used in the above RUN command, along with the
// filechecks specified in this file.

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.2: f32[4]) -> f32[]
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  // CHECK: %Arg_0.1 = f32[4] parameter(0)
  // CHECK: %Arg_1.2 = f32[4] parameter(1)
  // CHECK: %add.3 = f32[4] add(%Arg_0.1, %Arg_1.2)
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  // CHECK: ROOT %dot.4 = f32[] dot(%add.3, %Arg_1.2), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  %1 = stablehlo.dot %0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}