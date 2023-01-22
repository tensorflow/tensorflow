// RUN: mlir-hlo-opt %s --gml-st-cpu-pipeline --split-input-file \
// RUN: | FileCheck %s

func.func @map_unary(%input: tensor<?x?xf32>, %init: tensor<?x?xf32>)
                  -> tensor<?x?xf32> {
  %abs = linalg.map { math.absf }
           ins(%input:tensor<?x?xf32>)
           outs(%init:tensor<?x?xf32>)
  func.return %abs : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @map_unary
// CHECK:         gml_st.parallel
// CHECK:            math.absf %{{.*}} : vector<8xf32>
// CHECK:         gml_st.parallel
// CHECK:           gml_st.parallel
// CHECK:             affine.apply
// CHECK:             math.absf %{{.*}} : f32
