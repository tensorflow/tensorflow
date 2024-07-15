// RUN: odml-to-stablehlo-opt %s -prepare-hlo -split-input-file | FileCheck %s --dump-input=fail

// Just assert that pass is properly registered.
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0: tensor<f32>
}
// CHECK-LABEL: main
