// RUN: tf-tfrt-opt -tf-jitrt-tile-cwise %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @tanh_2d(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %input, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %input, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
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
// CHECK:           %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]])
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

// -----

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
  %dim_1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim_2 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dim_3 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %1 = gml_st.parallel (%arg2, %arg3) = (%c0, %c0) to (%dim_1, %dim_3) step (%c8, %c4) {
    %2 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 8)>(%arg2)[%dim_1]
    %3 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%arg3)[%dim_3]
    %4 = gml_st.tile [%arg2, 0] [%2, %dim_2] [1, 1] : !gml_st.tile<?x?>
    %5 = gml_st.materialize %arg0[%4] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %6 = gml_st.tile [0, %arg3] [%dim_2, %3] [1, 1] : !gml_st.tile<?x?>
    %7 = gml_st.materialize %arg1[%6] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %8 = gml_st.tile [%arg2, %arg3] [%2, %3] [1, 1] : !gml_st.tile<?x?>
    %9 = gml_st.tile [%arg2, %arg3] [%2, %3] [1, 1] : !gml_st.tile<?x?>
    %10 = gml_st.materialize %0[%9] : tensor<?x?xf32>[!gml_st.tile<?x?>] to tensor<?x?xf32>
    %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %12 = linalg.matmul ins(%5, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
    gml_st.set_yield %12 into %0[%8] : tensor<?x?xf32> into tensor<?x?xf32>[!gml_st.tile<?x?>]
  } : tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func @matmul(

// CHECK:           %[[OUTPUT:.*]] = gml_st.parallel
// CHECK-NOT:         gml_st.loop
// CHECK:             %[[FILL:.*]] = linalg.fill
// CHECK-NEXT:        %[[MATMUL:.*]] = linalg.matmul{{.*}}outs(%[[FILL]]
// CHECK-NEXT:        gml_st.set_yield %[[MATMUL]]
// CHECK-NEXT:      }
// CHECK-NEXT:      return %[[OUTPUT:.*]] : tensor<?x?xf32>
