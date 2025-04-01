// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK-LABEL: op_constant
func.func @op_constant(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: stablehlo.constant
  // CHECK-NOT: mhlo.constant
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  return %cst : tensor<f32>
}
