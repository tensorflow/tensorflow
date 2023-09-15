// COM: This file is there to check that the `tfl-legalize-hlo` pass exists in `odml-to-stablehlo-opt`.

// RUN: odml-to-stablehlo-opt %s -tfl-legalize-hlo -split-input-file | FileCheck %s --dump-input=fail

func.func @main(%arg0: tensor<5x7xf32>) -> tensor<5x7xf32> {
  func.return %arg0: tensor<5x7xf32>
// CHECK-LABEL: main
// CHECK: return %arg0 : tensor<5x7xf32>
}
