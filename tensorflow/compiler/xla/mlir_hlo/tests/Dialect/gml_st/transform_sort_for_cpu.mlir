// RUN: mlir-hlo-opt %s --gml-st-cpu-transform-sort | FileCheck %s

func.func @sort_variadic(%input1: tensor<64x8x4xf32>, %input2: tensor<64x8x4xf32>,
                %init1: tensor<64x8x4xf32>, %init2: tensor<64x8x4xf32>) {
  thlo.sort
      ins(%input1: tensor<64x8x4xf32>, %input2: tensor<64x8x4xf32>)
      outs(%init1: tensor<64x8x4xf32>, %init2: tensor<64x8x4xf32>)
      dimension = 1
      is_stable = true
      (%e11: f32, %e12: f32, %e21: f32, %e22: f32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return
}

// CHECK-LABEL: func.func @sort_variadic(
// CHECK-SAME:        %[[INPUT1:[A-Za-z0-9]*]]: tensor<64x8x4xf32>,
// CHECK-SAME:        %[[INPUT2:[A-Za-z0-9]*]]: tensor<64x8x4xf32>,
// CHECK-SAME:        %[[INIT1:[A-Za-z0-9]*]]: tensor<64x8x4xf32>,
// CHECK-SAME:        %[[INIT2:[A-Za-z0-9]*]]: tensor<64x8x4xf32>) {

// CHECK:             gml_st.parallel
// CHECK:               %[[INPUT1_SUB:.*]] = gml_st.materialize %[[INPUT1]]
// CHECK:      : tensor<64x8x4xf32> to tensor<1x?x1xf32>
// CHECK:               %[[INPUT2_SUB:.*]] = gml_st.materialize %[[INPUT2]]
// CHECK:      : tensor<64x8x4xf32> to tensor<1x?x1xf32>
// CHECK:               %[[INIT1_SUB:.*]] = gml_st.materialize %[[INIT1]]
// CHECK:      : tensor<64x8x4xf32> to tensor<1x?x1xf32>
// CHECK:               %[[INIT2_SUB:.*]] = gml_st.materialize %[[INIT2]]
// CHECK:      : tensor<64x8x4xf32> to tensor<1x?x1xf32>

// CHECK:      thlo.sort
// CHECK-SAME:      ins(%[[INPUT1_SUB]] : tensor<1x?x1xf32>, %[[INPUT2_SUB]] : tensor<1x?x1xf32>)
// CHECK-SAME:      outs(%[[INIT1_SUB]] : tensor<1x?x1xf32>, %[[INIT2_SUB]] : tensor<1x?x1xf32>)
// CHECK-SAME:      dimension = 1

