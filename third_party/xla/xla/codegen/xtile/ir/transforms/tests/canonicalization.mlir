// RUN: emitters_opt %s -canonicalize -split-input-file

func.func @reshape_noop_folded(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2xi32>) -> tensor<2xi32>
  // CHECK: return %arg0 : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

func.func @real_reshape_is_not_folded(%arg0: tensor<2xi32>) -> tensor<1x2xi32> {
  // CHECK: [[RES:.*]] = stablehlo.reshape %arg0 : (tensor<2xi32>) -> tensor<1x2xi32>
  %0 = stablehlo.reshape %arg0 : (tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK: return [[RES]] : tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}
