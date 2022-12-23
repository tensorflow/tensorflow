// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes=CHECK,TRANSFORMED
// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="tile-sizes=8,4,2" | FileCheck %s --check-prefixes=MARKED
// RUN: mlir-hlo-opt %s -xla-cpu-transform-matmul="lower-to-mmt4d=true" | FileCheck %s --check-prefixes=MMT4D,PAD

#id_map = affine_map<(d0, d1) -> (d0, d1)>

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
// CHECK:        %[[FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK-SAME:       -> tensor<8x4xf32>
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        gml_st.set_yield %[[FOR]]

// -----

// MMT4D-LABEL:    func @matmul_static(

// MMT4D-NOT:        linalg.matmul
// MMT4D:            gml_st.parallel {{.*}} = (%c0, %c0) to (%[[DIM0:.*]], %[[DIM1:.*]]) step (%c1, %c1)
// MMT4D:              gml_st.parallel {{.*}} = (%c0, %c0) to (%c8, %c8) step (%c8, %c8)
// MMT4D:                gml_st.for {{.*}} = (%c0) to (%[[DIM2:.*]]) step (%c1)
// MMT4D:                  gml_st.for {{.*}} = (%c0) to (%c1) step (%c1)
// MMT4D:                    linalg.mmt4d

// -----

func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>)
                  -> tensor<?x?xf32> {
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

// TRANSFORMED-LABEL: func @matmul(
// TRANSFORMED-SAME:      %[[LHS:.*]]: tensor<?x?xf32>, %[[RHS:.*]]: tensor<?x?xf32>)

// TRANSFORMED-DAG:     %[[C0:.*]] = arith.constant 0 : index
// TRANSFORMED:         %[[INIT:.*]] = tensor.empty

// TRANSFORMED:         %[[MAIN_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// TRANSFORMED:           %[[MAIN_SLICE:.*]] = gml_st.materialize %[[INIT]]
// TRANSFORMED:           %[[MAIN_FILL:.*]] = linalg.fill{{.*}}outs(%[[MAIN_SLICE]]
// TRANSFORMED:           %[[MAIN_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) {{.*}} outs ({{.*}} = %[[MAIN_FILL]]:
// TRANSFORMED:             %[[MAIN_PAR_MAIN_FOR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED:             gml_st.set_yield %[[MAIN_PAR_MAIN_FOR_MATMUL]]
// TRANSFORMED:           %[[REM_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[KUB]]) {{.*}} outs ({{.*}} = %[[MAIN_FOR]]:
// TRANSFORMED:             %[[MAIN_PAR_REM_FOR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED     :        gml_st.set_yield %[[MAIN_PAR_REM_FOR_MATMUL]]
// TRANSFORMED:           gml_st.set_yield %[[REM_FOR]]

// TRANSFORMED:         %[[REM_RHS_PAR:.*]] = gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// TRANSFORMED:           %[[REM_RHS_SLICE:.*]] = gml_st.materialize %[[MAIN_PAR]]
// TRANSFORMED:           %[[REM_RHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_RHS_SLICE]]
// TRANSFORMED:           %[[REM_RHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_RHS_FILL]]:
// TRANSFORMED:             %[[REM_RHS_PAR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED:             gml_st.set_yield %[[REM_RHS_PAR_MATMUL]]
// TRANSFORMED:           gml_st.set_yield %[[REM_RHS_FOR]]

// TRANSFORMED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// TRANSFORMED:           %[[REM_LHS_SLICE:.*]] = gml_st.materialize %[[REM_RHS_PAR]]
// TRANSFORMED:           %[[REM_LHS_FILL:.*]] = linalg.fill{{.*}}outs(%[[REM_LHS_SLICE]]
// TRANSFORMED:           %[[REM_LHS_FOR:.*]] = gml_st.for (%[[K:.*]]) = (%[[C0]]) {{.*}} outs ({{.*}} = %[[REM_LHS_FILL]]:
// TRANSFORMED:             %[[REM_LHS_PAR_MATMUL:.*]] = linalg.matmul
// TRANSFORMED:             gml_st.set_yield %[[REM_LHS_PAR_MATMUL]]
// TRANSFORMED:           gml_st.set_yield %[[REM_LHS_FOR]]

// -----

// MARKED-LABEL: func @matmul(

// MARKED:         %[[C0:.*]] = arith.constant 0 : index
// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[C0]]) to (%[[IUB:.*]], %[[JUB:.*]]) step
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]]) to (%[[KUB:.*]]) step
// MARKED:           __perfectly_tiled_loop_label__
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[KUB]])

// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[C0]], %[[JUB]])
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]])

// MARKED:         gml_st.parallel (%[[I:.*]], %[[J:.*]]) = (%[[IUB]], %[[C0]])
// MARKED:           gml_st.for (%[[K:.*]]) = (%[[C0]])

// -----

// MMT4D-LABEL:    func @matmul(

// MMT4D-NOT:        linalg.matmul
// MMT4D:            gml_st.parallel {{.*}} = (%c0, %c0) to (%[[DIM0:.*]], %[[DIM1:.*]]) step (%c1, %c1)
// MMT4D:              gml_st.parallel {{.*}} = (%c0, %c0) to (%c8, %c8) step (%c8, %c8)
// MMT4D:                gml_st.for {{.*}} = (%c0) to (%[[DIM2:.*]]) step (%c1)
// MMT4D:                  gml_st.for {{.*}} = (%c0) to (%c1) step (%c1)
// MMT4D:                    linalg.mmt4d

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
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = linalg.matmul ins(%arg0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%filled : tensor<?x?xf32>) -> tensor<?x?xf32>
  %6 = linalg.map
       ins(%5 : tensor<?x?xf32>)
       outs(%init : tensor<?x?xf32>)
       (%el: f32) {
         %0 = math.absf %el: f32
         linalg.yield %0: f32
       }

  %result = linalg.map
            ins(%4, %6 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%init : tensor<?x?xf32>)
            (%lhs: f32, %rhs: f32) {
              %0 = arith.addf %lhs, %rhs: f32
              linalg.yield %0: f32
            }
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

// CHECK:        gml_st.for (%[[K:.*]]) = (%[[C0]])
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]

// CHECK:        gml_st.for
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]

// CHECK:        linalg.map
// CHECK:        linalg.map

// CHECK:        gml_st.set_yield

// CHECK:      gml_st.parallel
// CHECK:        gml_st.for
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        gml_st.for
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        linalg.map
// CHECK:        linalg.map
// CHECK:        gml_st.set_yield

// CHECK:      gml_st.parallel
// CHECK:        gml_st.for
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        gml_st.for
// CHECK:          %[[MATMUL:.*]] = linalg.matmul
// CHECK:          gml_st.set_yield %[[MATMUL]]
// CHECK:        linalg.map
// CHECK:        linalg.map
// CHECK:        gml_st.set_yield

// -----

func.func @pad(%arg0: tensor<10x10xf32>) -> tensor<16x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %padded = tensor.pad %arg0 low[0, 0] high[6, 0] {
  ^bb0(%arg3: index, %arg4: index):
    tensor.yield %cst : f32
  } : tensor<10x10xf32> to tensor<16x10xf32>

  return %padded : tensor<16x10xf32>
}

// PAD-LABEL:    func @pad(

// PAD:            %[[EMPTY:.*]] = tensor.empty() : tensor<16x10xf32>
// PAD:            %[[FILL:.*]] = linalg.fill {{.*}} outs(%[[EMPTY]]
// PAD:            %[[EXTRACT:.*]] = tensor.extract_slice %[[FILL]][0, 0] [10, 10]
// PAD:            %[[MAP:.*]] = linalg.map ins(%arg0 {{.*}} outs(%[[EXTRACT]]
// PAD:            %[[INSERT:.*]] = tensor.insert_slice %[[MAP]] into  %[[FILL]][0, 0] [10, 10]
// PAD:            return %[[INSERT]]

// -----

func.func @matvec_static(%arg0: tensor<1x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %2 : tensor<1x64xf32>
}

// MMT4D-LABEL:    func @matvec_static(

// MMT4D-NOT:        linalg.matmul
// MMT4D:            gml_st.parallel {{.*}} = (%c0, %c0) to (%[[DIM0:.*]], %[[DIM1:.*]]) step (%c1, %c1)
// MMT4D:              gml_st.parallel {{.*}} = (%c0, %c0) to (%c1, %c8) step (%c1, %c8)
// MMT4D:                gml_st.for {{.*}} = (%c0) to (%[[DIM2:.*]]) step (%c1)
// MMT4D:                  gml_st.for {{.*}} = (%c0) to (%c1) step (%c1) outs ({{.*}}tensor<1x1x1x8xf32>)
// MMT4D:                    linalg.mmt4d

// -----

func.func @matmul_narrow_static(%arg0: tensor<2x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<2x64xf32>) -> tensor<2x64xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<2x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<2x64xf32>) -> tensor<2x64xf32>
  return %2 : tensor<2x64xf32>
}

// MMT4D-LABEL:    func @matmul_narrow_static(

// MMT4D-NOT:        linalg.matmul
// MMT4D:            gml_st.parallel {{.*}} = (%c0, %c0) to (%[[DIM0:.*]], %[[DIM1:.*]]) step (%c1, %c1)
// MMT4D:              gml_st.parallel {{.*}} = (%c0, %c0) to (%c2, %c8) step (%c2, %c8)
// MMT4D:                gml_st.for {{.*}} = (%c0) to (%[[DIM2:.*]]) step (%c1)
// MMT4D:                  gml_st.for {{.*}} = (%c0) to (%c1) step (%c1) outs ({{.*}}tensor<1x1x2x8xf32>)
// MMT4D:                    linalg.mmt4d
