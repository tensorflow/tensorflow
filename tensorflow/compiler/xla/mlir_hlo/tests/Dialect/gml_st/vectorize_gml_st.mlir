// RUN: mlir-hlo-opt %s --vectorize-gml-st-loops="vectorize-gml-st-ops=true" \
// RUN: | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @vectorize_gml_st_parallel_op(
func.func @vectorize_gml_st_parallel_op(
    %arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<32xf32>)
    -> tensor<32xf32> {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %space = gml_st.space [32] : !gml_st.tile<32>
  %2 = gml_st.parallel (%i) = (%c0) to (%c32) step (%c4) {
    %tile = gml_st.tile %space [%i] [4] [1] : !gml_st.tile<32> to !gml_st.tile<4>
    %6 = gml_st.materialize %arg0[%tile]
      : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
    %7 = gml_st.materialize %arg1[%tile]
      : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
    %8 = gml_st.materialize %arg2[%tile]
      : tensor<32xf32>[!gml_st.tile<4>] to tensor<4xf32>
    %9 = linalg.generic {indexing_maps = [#map0, #map0, #map0],
                        iterator_types = ["parallel"]}
                        ins(%6, %7 : tensor<4xf32>, tensor<4xf32>)
                        outs(%8 : tensor<4xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):
      %10 = arith.addf %arg5, %arg6 : f32
      linalg.yield %10 : f32
    } -> tensor<4xf32>
    gml_st.set_yield %9 into %arg2[%tile]
      : tensor<4xf32> into tensor<32xf32>[!gml_st.tile<4>]
  } : tensor<32xf32>
  func.return %2 : tensor<32xf32>
}
// CHECK-DAG: %[[LHS:.*]] = vector.transfer_read %arg0[%c0]
// CHECK-DAG: %[[RHS:.*]] = vector.transfer_read %arg1[%c0]
// CHECK:     %[[RESULT:.*]] = gml_st.parallel
// CHECK-DAG: %[[LHSTILE:.*]] = gml_st.materialize %[[LHS]]
// CHECK-DAG: %[[RHSTILE:.*]] = gml_st.materialize %[[RHS]]
// CHECK-NOT: linalg.generic
// CHECK:     %[[ADD:.*]] = arith.addf %[[LHSTILE]], %[[RHSTILE]] : vector<4xf32>
// CHECK:     gml_st.set_yield %[[ADD]]
// CHECK:     vector.transfer_write %[[RESULT]], {{%.*}}[%c0]