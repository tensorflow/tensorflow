// RUN: not hlo-translate -mlir-to-hlo -split-input-file %s 2>&1 | FileCheck %s

// StableHLO ops that has no HLO support. These all must be refined away before
// lowering. See https://openxla.org/stablehlo/dynamism

func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: Shape Error: Invalid element type
  %0 = stablehlo.dynamic_broadcast_in_dim %arg0, %arg1, dims = [0] {known_expanding_dimensions = array<i64>, known_nonexpanding_dimensions = array<i64: 0>} : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
