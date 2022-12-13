// RUN: mlir-hlo-opt %s \
// RUN:  -xla-triton-transform-matmul="tile-sizes=8,4,2 distribution-label=test" \
// RUN: | FileCheck %s

func.func @matmul_static(%arg0: tensor<128x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}

// CHECK-LABEL:    func @matmul_static(
// CHECK-SAME:       %[[LHS:.*]]: tensor<128x16xf32>,
// CHECK-SAME:       %[[RHS:.*]]: tensor<16x64xf32>,
// CHECK-SAME:       %[[OUT:.*]]: tensor<128x64xf32>)

// CHECK:      %[[C0:.*]] = arith.constant 0 : index
// CHECK:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK-SAME:     distribution ("test")
// CHECK:        %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:       -> tensor<8x4xf32>
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        gml_st.set_yield %[[FOR]]
