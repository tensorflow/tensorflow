// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:  -xla-triton-transform-matmul="tile-sizes=8,4,2 distribution-label=test" \
// RUN: | FileCheck %s --dump-input=always

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

// -----

func.func @matmul_fuse_output(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                              %arg2: tensor<?x?xf32>)
                              -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32)
                        outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = linalg.matmul ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = linalg.map { math.absf }
         ins(%5 : tensor<?x?xf32>)
         outs(%init : tensor<?x?xf32>)

  %result = linalg.map { arith.addf }
              ins(%4, %6 : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%init : tensor<?x?xf32>)
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL:    func @matmul_fuse_output(
// CHECK:      %[[C0:.*]] = arith.constant 0 : index

// CHECK:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK:        gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]

// CHECK:        gml_st.for
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]

// CHECK:        linalg.map
// CHECK:        linalg.map

// CHECK:        gml_st.set_yield

// -----

func.func @matmul_fuse_input_and_output(
              %arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>,
              %init: tensor<?x?xf32>)
              -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32)
                        outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %mapped = linalg.map { math.absf }
              ins(%arg0 : tensor<?x?xf32>)
              outs(%init : tensor<?x?xf32>)
  %bcast = linalg.broadcast
             ins(%arg1 : tensor<?xf32>)
             outs(%init : tensor<?x?xf32>)
             dimensions = [1]

  %matmul = linalg.matmul
              ins(%mapped, %bcast : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>

  %result = linalg.map { math.absf }
              ins(%matmul : tensor<?x?xf32>)
              outs(%init : tensor<?x?xf32>)
  return %result : tensor<?x?xf32>
}

// CHECK-LABEL:    func @matmul_fuse_input_and_output(
// CHECK:      %[[C0:.*]] = arith.constant 0 : index

// CHECK:      gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]])
// CHECK:        gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK:          linalg.map
// CHECK:          linalg.broadcast
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        linalg.map
// CHECK:        gml_st.set_yield
