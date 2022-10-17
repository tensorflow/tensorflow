// RUN: mlir-hlo-opt --gml-tiling-reduction %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: func @tile_reduction
func.func @tile_reduction(%arg0 : tensor<1x?xf32>) -> tensor<1xf32> {

  %zero = arith.constant 0.0 : f32
  %result0 = tensor.empty() : tensor<1xf32>
  // CHECK: %[[RESULT1:.*]] = linalg.fill
  %result1 = linalg.fill ins(%zero : f32) outs(%result0 : tensor<1xf32>) -> tensor<1xf32>

  // CHECK:      %[[RDIM:.*]] = tensor.dim %arg0, %c1 : tensor<1x?xf32>
  // CHECK:      %[[PARTIAL0:.*]] = tensor.empty() : tensor<32xf32>
  // CHECK:      %[[PARTIAL1:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[LANE:.*]]) = (%c0) to (%c32) step (%c1)
  // CHECK-SAME:     distribution ("warp")
  // CHECK:        gml_st.set_yield %[[RESULT1]] into %[[PARTIAL0]]
  // CHECK:      %[[PARTIAL2:.*]] = gml_st.parallel
  // CHECK-SAME:     (%[[LANE:.*]]) = (%c0) to (%c32) step (%c1)
  // CHECK-SAME:     distribution ("warp")
  // CHECK:        %[[INITVAL:.*]] = gml_st.materialize %[[PARTIAL1]]
  // CHECK:        %[[PARTVAL:.*]] = gml_st.for
  // CHECK-SAME:       (%[[COL:.*]]) = (%[[LANE]]) to (%[[RDIM]]) step (%c32)
  // CHECK-SAME:       outs (%[[OUTVAL:.*]] = %[[INITVAL]]: tensor<1xf32>)
  // CHECK:          %[[SPACE:.*]] = gml_st.space [1, %[[RDIM]]] : !gml_st.tile<1x?>
  // CHECK:          %[[TILE:.*]] = gml_st.tile %[[SPACE]] [0, %arg2] [1, 1] [1, 1]
  // CHECK:          %[[INVAL:.*]] = gml_st.materialize %arg0
  // CHECK:          %[[ACCVAL:.*]] = linalg.generic
  // CHECK-SAME:         ins(%[[INVAL]] : tensor<1x1xf32>)
  // CHECK-SAME:         outs(%[[OUTVAL]] : tensor<1xf32>)
  // CHECK:          gml_st.set_yield %[[ACCVAL]] into %[[OUTVAL]]
  // CHECK:        gml_st.set_yield %[[PARTVAL]] into %[[PARTIAL1]]
  // CHECK:      %[[EXPAND:.*]] = tensor.expand_shape %[[PARTIAL2]]
  // CHECK:      %[[RESULT2:.*]] = linalg.generic
  // CHECK-SAME:     ins(%[[EXPAND]] : tensor<1x32xf32>)
  // CHECK-SAME:     outs(%[[RESULT1]] : tensor<1xf32>)
  %result2 = linalg.generic {
    indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%arg0 : tensor<1x?xf32>) outs(%result1 : tensor<1xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %24 = arith.addf %arg4, %arg3 : f32
    linalg.yield %24 : f32
  } -> tensor<1xf32>

  // CHECK: return %[[RESULT2]]
  func.return %result2 : tensor<1xf32>
}


