// RUN: not xla-translate -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: 'stablehlo.composite' op CompositeOp with regions not supported in StableHLO -> HLO conversion
  %0 = "mhlo.composite"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) {
    name = "foo.bar",
    composite_attributes = {},
    decomposition = @add,
    version = 1 : i32
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @add(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}
