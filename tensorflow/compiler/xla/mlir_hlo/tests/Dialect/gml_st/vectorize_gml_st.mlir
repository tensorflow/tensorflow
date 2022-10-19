// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:     --vectorize-gml-st-loops="vectorize-gml-st-ops=true included-distribution-labels=test" \
// RUN: | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @vectorize_gml_st_parallel_op(
func.func @vectorize_gml_st_parallel_op(
    %arg0: tensor<32xf32>, %arg1: tensor<32xf32>)
    -> tensor<32xf32> {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %space = gml_st.space [32] : !gml_st.tile<32>
  // We need this outer trivial loop to make sure the inner loop has a parent
  // with the correct distribution label.
  %2 = gml_st.parallel (%unused) = (%c0) to (%c1) step (%c1)
          distribution ("test") {
    %arg0tile = gml_st.materialize %arg0[%space]
      : tensor<32xf32>[!gml_st.tile<32>] to tensor<32xf32>
    %arg1tile = gml_st.materialize %arg1[%space]
      : tensor<32xf32>[!gml_st.tile<32>] to tensor<32xf32>
    %3 = gml_st.parallel (%i) = (%c0) to (%c32) step (%c4)
          distribution ("test") {
      %tile = gml_st.tile %space [%i] [4] [1]
        : !gml_st.tile<32> to !gml_st.tile<4>
      %6 = gml_st.materialize %arg0tile[%tile]
        : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
      %7 = gml_st.materialize %arg1tile[%tile]
        : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
      %9 = linalg.generic {indexing_maps = [#map0, #map0],
                          iterator_types = ["parallel"]}
                          ins(%6: tensor<4xf32>)
                          outs(%7 : tensor<4xf32>) {
      ^bb0(%arg5: f32, %arg6: f32):
        %10 = arith.negf %arg5 : f32
        linalg.yield %10 : f32
      } -> tensor<4xf32>
      gml_st.set_yield %9 into %arg1tile[%tile]
        : tensor<4xf32> into tensor<32xf32>[!gml_st.tile<4>]
    } : tensor<32xf32>
    gml_st.set_yield %3 into %arg1[%space]
      : tensor<32xf32> into tensor<32xf32>[!gml_st.tile<32>]
  } : tensor<32xf32>
  func.return %2 : tensor<32xf32>
}
// CHECK:      gml_st.parallel
// CHECK-DAG:  %[[ARG0TILE:.*]] = gml_st.materialize %arg0
// CHECK-DAG:  %[[LHS:.*]] = vector.transfer_read %[[ARG0TILE]][%c0]
// CHECK:      %[[RESULT:.*]] = gml_st.parallel
// CHECK-DAG:    %[[LHSTILE:.*]] = gml_st.materialize %[[LHS]]
// CHECK:        %[[NEG:.*]] = arith.negf %[[LHSTILE]] : vector<4xf32>
// CHECK:        gml_st.set_yield %[[NEG]]
// CHECK-SAME:   vector<4xf32> into vector<32xf32>
// CHECK:      vector.transfer_write %[[RESULT]], {{%.*}}[%c0]

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @skip_vectorization_with_wrong_label(
func.func @skip_vectorization_with_wrong_label(
    %arg0: tensor<32xf32>, %arg1: tensor<32xf32>)
    -> tensor<32xf32> {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %space = gml_st.space [32] : !gml_st.tile<32>
  %2 = gml_st.parallel (%unused) = (%c0) to (%c1) step (%c1)
          distribution ("no_vec") {
    %3 = gml_st.parallel (%i) = (%c0) to (%c32) step (%c4)
            distribution ("no_vec") {
      %tile = gml_st.tile %space [%i] [4] [1]
        : !gml_st.tile<32> to !gml_st.tile<4>
      %6 = gml_st.materialize %arg0[%tile]
        : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
      %7 = gml_st.materialize %arg1[%tile]
        : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
      %9 = linalg.generic {indexing_maps = [#map0, #map0],
                          iterator_types = ["parallel"]}
                          ins(%6 : tensor<4xf32>)
                          outs(%7 : tensor<4xf32>) {
      ^bb0(%arg5: f32, %arg6: f32):
        %10 = arith.negf %arg5 : f32
        linalg.yield %10 : f32
      } -> tensor<4xf32>
      gml_st.set_yield %9 into %arg1[%tile]
        : tensor<4xf32> into tensor<32xf32>[!gml_st.tile<4>]
    } : tensor<32xf32>
    gml_st.set_yield %3 into %arg1[%space]
      : tensor<32xf32> into tensor<32xf32>[!gml_st.tile<32>]
  } : tensor<32xf32>
  func.return %2 : tensor<32xf32>
}
// CHECK-NOT: vector.transfer_read