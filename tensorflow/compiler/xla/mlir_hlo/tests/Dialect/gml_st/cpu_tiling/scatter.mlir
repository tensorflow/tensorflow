// RUN: mlir-hlo-opt %s -xla-cpu-transform-scatter | FileCheck %s

func.func @scatter_small_vector_dim(%indices: tensor<?x2xindex>,
  %updates: tensor<?x?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = thlo.scatter
    ins (%indices: tensor<?x2xindex>, %updates: tensor<?x?x?xf32>)
    outs (%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      %0 = arith.addf %in, %out: f32
      thlo.yield %0: f32
    }
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL: @scatter_small_vector_dim
// CHECK:       scf.for
// CHECK:       thlo.scatter
// CHECK-SAME:    ins(%{{.*}} : tensor<1x2xindex>, %{{.*}} : tensor<1x?x?xf32>)
