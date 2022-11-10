// RUN: tf-tfrt-opt %s -split-input-file \
// RUN:   -xla-cpu-transform-matmul="tile-sizes=8,4,2" \
// RUN: | FileCheck %s

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul(
// CHECK-SAME:      %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index

// CHECK:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])

// CHECK:           %[[FILL:.*]] = linalg.fill
// CHECK:           %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK-SAME:          outs (%[[OUT_SUB_ARG:.*]] = %[[FILL]]:

// CHECK:             %[[MATMUL:.*]] = linalg.matmul

// CHECK-NEXT:        gml_st.set_yield %[[MATMUL]]
// CHECK:           gml_st.set_yield %[[FOR]]
