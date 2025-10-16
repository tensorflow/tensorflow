// RUN: emitters_opt %s --xtile-cpu-shlo-to-vector -split-input-file | FileCheck %s

func.func @transpose(%input : tensor<1024x32xf32>) -> tensor<32x1024xf32> {
  // CHECK: vector.transpose %{{.*}}, [1, 0] : vector<1024x32xf32> to vector<32x1024xf32>
  %transposed = stablehlo.transpose %input, dims = [1, 0] : (tensor<1024x32xf32>) -> tensor<32x1024xf32>
  return %transposed : tensor<32x1024xf32>
}
// -----
