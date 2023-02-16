// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline |\
// RUN: FileCheck %s

func.func @row_reduce_map_fuse_map(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>

  %empty_2D = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %reduce_init = tensor.empty(%dim0) : tensor<?xf32>
  %mapped = linalg.map { arith.addf }
              ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%empty_2D : tensor<?x?xf32>)

  %c0_f32 = arith.constant 0.0 : f32
  %empty_1D = tensor.empty(%dim1) : tensor<?xf32>
  %fill = linalg.fill ins(%c0_f32: f32)
                      outs(%empty_1D: tensor<?xf32>) -> tensor<?xf32>

  %reduce = linalg.reduce { arith.addf }
              ins(%mapped: tensor<?x?xf32>)
              outs(%fill: tensor<?xf32>)
              dimensions = [1]

  %res = linalg.map { math.absf }
           ins(%reduce: tensor<?xf32>)
           outs(%empty_1D : tensor<?xf32>)
  return %res : tensor<?xf32>
}
// CHECK-LABEL: @row_reduce_map_fuse_map

// CHECK: gml_st.parallel
// CHECK:   scf.for
// CHECK:     arith.addf %{{.*}} : vector<4x4xf32>
// CHECK:     vector.multi_reduction <add>
// CHECK:       : vector<4x4xf32> to vector<4xf32>
// CHECK:     scf.yield %{{.*}} : vector<4xf32>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       arith.addf %{{.*}} : vector<4x1xf32>
// CHECK:       vector.multi_reduction <add>
// CHECK:         : vector<4x1xf32> to vector<4xf32>
// CHECK:       scf.yield %{{.*}} : vector<4xf32>
// CHECK:     scf.yield %{{.*}} : vector<4xf32>
// CHECK:   math.absf %{{.*}} : vector<4xf32>
// CHECK:   gml_st.set_yield

// CHECK: gml_st.parallel
// CHECK:   gml_st.parallel
// CHECK:     scf.for
// CHECK:       arith.addf %{{.*}} : f32
// CHECK:       arith.addf %{{.*}} : f32
// CHECK:       scf.yield %{{.*}} : f32
// CHECK:     math.absf %{{.*}} : f32
// CHECK:     gml_st.set_yield
// CHECK:   gml_st.set_yield

// -----

func.func @col_reduce_map_fuse_map(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>

  %empty_2D = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %mapped = linalg.map { arith.addf }
              ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%empty_2D : tensor<?x?xf32>)

  %c0_f32 = arith.constant 0.0 : f32
  %empty_1D = tensor.empty(%dim1) : tensor<?xf32>
  %fill = linalg.fill ins(%c0_f32: f32)
                      outs(%empty_1D: tensor<?xf32>) -> tensor<?xf32>

  %reduce = linalg.reduce { arith.addf }
              ins(%mapped: tensor<?x?xf32>)
              outs(%fill: tensor<?xf32>)
              dimensions = [0]

  %res = linalg.map { math.absf }
           ins(%reduce: tensor<?xf32>)
           outs(%empty_1D : tensor<?xf32>)
  return %res : tensor<?xf32>
}
// CHECK-LABEL: @col_reduce_map_fuse_map

// CHECK: gml_st.parallel
// CHECK:   scf.for
// CHECK:     arith.addf %{{.*}} : vector<4x4xf32>
// CHECK:     vector.multi_reduction <add>
// CHECK:       : vector<4x4xf32> to vector<4xf32>
// CHECK:     scf.yield %{{.*}} : vector<4xf32>
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         arith.addf %{{.*}} : f32
// CHECK:         arith.addf %{{.*}} : f32
// CHECK:         scf.yield %{{.*}} : f32
// CHECK:     scf.yield %{{.*}} : tensor<4xf32>
// CHECK:   scf.yield %{{.*}} : tensor<?xf32>
// CHECK:   gml_st.set_yield

// CHECK: gml_st.parallel
// CHECK:   gml_st.parallel
// CHECK:     scf.for
// CHECK:       arith.addf %{{.*}} : f32
// CHECK:       arith.addf %{{.*}} : f32
// CHECK:       scf.yield %{{.*}} : f32
// CHECK:     math.absf %{{.*}} : f32
// CHECK:     gml_st.set_yield
// CHECK:   gml_st.set_yield
