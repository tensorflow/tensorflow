// RUN: xla-cpu-opt %s -xla-cpu-transform-scatter | FileCheck %s

#id_map = affine_map<(d0, d1) -> (d0, d1)>

func.func @scatter_small_vector_dim(%indices: tensor<?x2xi32>,
  %updates: tensor<?x?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = thlo.scatter
    ins (%indices: tensor<?x2xi32>, %updates: tensor<?x?x?xf32>)
    outs (%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      %0 = arith.addf %in, %out: f32
      thlo.yield %0: f32
    }
  return %result : tensor<?x?xf32>
}

// CHECK-FOR-LABEL: @scatter_small_vector_dim
// CHECK: gml_st.for
// CHECK: thlo.scatter ins(%{{.*}} : tensor<1x2xi32>, %{{.*}} : tensor<1x?x?xf32>)
