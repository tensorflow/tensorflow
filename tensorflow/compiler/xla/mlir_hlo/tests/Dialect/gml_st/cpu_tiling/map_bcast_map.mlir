// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline | FileCheck %s

func.func @map_bcast_map(%arg0: tensor<?xf32>, %arg1: tensor<?x?x?xf32>,
                              %init0: tensor<?xf32>,
                              %init1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %abs = linalg.map { math.absf }
           ins(%arg0:tensor<?xf32>)
           outs(%init0:tensor<?xf32>)

  %bcast = linalg.broadcast
             ins(%abs : tensor<?xf32>)
             outs(%init1 : tensor<?x?x?xf32>)
             dimensions = [1, 2]

  %mapped = linalg.map { arith.addf }
              ins(%bcast, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
              outs(%init1:tensor<?x?x?xf32>)
  func.return %mapped : tensor<?x?x?xf32>
}

// CHECK-LABEL: func.func @map_bcast_map

// CHECK: gml_st.parallel
// CHECK:   math.absf %{{.*}} : f32
// CHECK:   vector.broadcast %{{.*}} : vector<1xf32> to vector<1x8x1xf32>
// CHECK:   vector.transpose %{{.*}}, [2, 0, 1] : vector<1x8x1xf32> to vector<1x1x8xf32>
// CHECK:   arith.addf %{{.*}} : vector<8xf32>
// CHECK:   gml_st.set_yield

// CHECK: gml_st.parallel
// CHECK:   gml_st.parallel
// CHECK:     math.absf %{{.*}} : f32
// CHECK:     arith.addf %{{.*}} : f32
// CHECK:     gml_st.set_yield
// CHECK:   gml_st.set_yield
