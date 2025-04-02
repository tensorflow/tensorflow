// RUN: hlo-translate -mlir-to-hlo %s | FileCheck %s
// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false %s | FileCheck %s --check-prefix CHECK-DIRECT

// Tests for all stablehlo ops to validate stablehlo -> hlo conversion.


// CHECK-LABEL: HloModule

// CHECK: %[[ARG0:.*]] = f32[4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[4] parameter(1)
// CHECK: ROOT %add.3 = f32[4] add(%[[ARG0]], %[[ARG1]])
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>  func.return %0 : tensor<4xf32>
}
// CHECK-DIRECT: stablehlo.add
