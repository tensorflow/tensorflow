// RUN: tf-tfrt-opt -tf-jitrt-tile-cwise %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @tanh_2d(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%dim0, %dim1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%input : tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>)
  {
  ^bb0(%arg1: f32, %arg2: f32):
    %2 = math.tanh %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  func.return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @tanh_2d(
// CHECK-SAME:                  %[[INPUT:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[STEP:.*]] = arith.constant 8 : index
// CHECK-NOT:       tensor.dim
// CHECK-DAG:       %[[DIM0:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK-DAG:       %[[DIM1:.*]] = tensor.dim %[[INPUT]], %[[C1]]
// CHECK:           %[[INIT:.*]] = linalg.init_tensor {{\[}}%[[DIM0]], %[[DIM1]]]
// CHECK-DAG:       %[[DIM0_OUT:.*]] = tensor.dim %[[INPUT]], %[[C0]]
// CHECK-DAG:       %[[DIM1_OUT:.*]] = tensor.dim %[[INPUT]], %[[C1]]
// CHECK:           %[[OUTPUT:.*]] = gml_st.loop
// CHECK-SAME:          (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:          to (%[[DIM0_OUT]], %[[DIM1_OUT]])
// CHECK-SAME:          step (%[[C1]], %[[STEP]])
// CHECK-SAME:          ins (%[[IN_TENS:.*]] = %[[INPUT]]: tensor<?x?xf32>)
// CHECK-SAME:          outs (%[[OUT_TENS:.*]] = %[[INIT]]: tensor<?x?xf32>) {
// CHECK:           %[[IN_SLICE:.*]] = tensor.extract_slice
// CHECK-SAME:          %[[IN_TENS]]{{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK-SAME:          {{\[}}1, %{{.*}}] [1, 1]
// CHECK:           %[[OUT_SLICE:.*]] = tensor.extract_slice
// CHECK-SAME:          %[[OUT_TENS]]{{\[}}%[[ARG1]], %[[ARG2]]]
// CHECK-SAME:          {{\[}}1, %{{.*}}] [1, 1]
// CHECK:           %[[VECTOR_RESULT:.*]] = linalg.generic
// CHECK-SAME:          {indexing_maps = [#map1, #map1],
// CHECK-SAME:          iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:          ins(%[[IN_SLICE]] : tensor<1x?xf32>)
// CHECK-SAME:          outs(%[[OUT_SLICE]] : tensor<1x?xf32>) {
// CHECK-NEXT:        ^bb0(%[[SCALAR_INPUT:.*]]: f32, %[[VAL_20:.*]]: f32):
// CHECK-NEXT:          %[[TANH_OUT:.*]] = math.tanh %[[SCALAR_INPUT]] : f32
// CHECK-NEXT:          linalg.yield %[[TANH_OUT]] : f32
// CHECK-NEXT:        } -> tensor<1x?xf32>
// CHECK-NEXT:        %[[INSERT_RESULT:.*]] = tensor.insert_slice
// CHECK-SAME:            %[[VAL_23:.*]] into %[[OUT_TENS]]
// CHECK-SAME:            {{\[}}%[[ARG1]], %[[ARG2]]] [1,
// CHECK-SAME:            %{{.*}}] [1, 1]
// CHECK-NEXT:        gml_st.yield %[[INSERT_RESULT]] : tensor<?x?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[FINAL_OUTPUT:.*]] : tensor<?x?xf32>
// CHECK-NEXT:    }
