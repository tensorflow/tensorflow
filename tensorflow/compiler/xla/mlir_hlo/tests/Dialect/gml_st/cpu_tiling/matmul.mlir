// RUN: mlir-hlo-opt %s --split-input-file --gml-st-cpu-tiling-pipeline | FileCheck %s
// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline="lower-to-mmt4d=true" | FileCheck %s --check-prefixes=PACKED

func.func @matmul_static(%lhs: tensor<128x16xf32>, %rhs: tensor<16x64xf32>,
                         %output: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<128x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}

// CHECK-LABEL: @matmul_static

// CHECK:         gml_st.parallel
// CHECK:           vector.transfer_read
// CHECK-NEXT:      scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:             scf.yield {{.*}} : vector<4x4xf32>
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield

// PACKED-LABEL: @matmul_static

// PACKED:         tensor.empty() : tensor<16x16x8x1xf32>
// PACKED-COUNT-2: scf.for
// PACKED:           vector.transfer_read
// PACKED:           vector.transfer_write
// PACKED:           scf.yield %{{.*}} : tensor<16x16x8x1xf32>
// PACKED:          scf.yield %{{.*}} : tensor<16x16x8x1xf32>

// PACKED:         tensor.empty() : tensor<8x16x8x1xf32>
// PACKED-COUNT-2:   scf.for
// PACKED:           vector.transfer_read
// PACKED:           vector.transfer_write
// PACKED:            scf.yield %{{.*}} : tensor<8x16x8x1xf32>
// PACKED:           scf.yield %{{.*}} : tensor<8x16x8x1xf32>

// PACKED:         tensor.empty() : tensor<16x8x8x8xf32>
// PACKED-COUNT-2: scf.for
// PACKED:           vector.transfer_read
// PACKED:           vector.transfer_write
// PACKED:          scf.yield
// PACKED:         scf.yield

// PACKED-COUNT-2: scf.for
// PACKED:           scf.for
// PACKED:             vector.transfer_read
// PACKED:             vector.transfer_read
// PACKED:             vector.contract
// PACKED:             scf.yield
// PACKED:           scf.yield
// PACKED:          scf.yield

// PACKED:         tensor.empty() : tensor<128x64xf32>
// PACKED-COUNT-2: scf.for
// PACKED:           vector.transfer_read
// PACKED:           vector.transfer_write
// PACKED:           scf.yield %{{.*}} : tensor<128x64xf32>
// PACKED:          scf.yield %{{.*}} : tensor<128x64xf32>



// -----

func.func @matmul(%lhs: tensor<?x?xf32>,
    %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %3 = linalg.fill ins(%cst : f32)
         outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %4 = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: @matmul

// CHECK:         gml_st.parallel
// CHECK:           scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK-NEXT:        scf.yield %{{.*}} : vector<4x4xf32>
// CHECK:           vector.transfer_write

// CHECK-NEXT:      scf.for
// CHECK:             linalg.matmul {{.*}} -> tensor<4x4xf32>
// CHECK:             scf.yield {{.*}} : tensor<4x4xf32>
// CHECK:           gml_st.set_yield

// CHECK:         gml_st.parallel
// CHECK:           linalg.fill
// CHECK:           scf.for
// CHECK:             linalg.matmul {{.*}} -> tensor<4x?xf32>
// CHECK:             scf.yield {{.*}} : tensor<4x?xf32>
// CHECK:           gml_st.set_yield

// CHECK:         gml_st.parallel
// CHECK:           linalg.fill
// CHECK:           scf.for
// CHECK:             linalg.matmul
// CHECK:             scf.yield {{.*}} : tensor<?x?xf32>
// CHECK:           gml_st.set_yield

// -----

func.func @matmul_narrow_static(%lhs: tensor<2x16xf32>, %rhs: tensor<16x64xf32>,
                         %output: tensor<2x64xf32>) -> tensor<2x64xf32> {
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<2x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<2x64xf32>) -> tensor<2x64xf32>
  return %2 : tensor<2x64xf32>
}
// CHECK-LABEL: @matmul_narrow_static

// CHECK:         gml_st.parallel
// CHECK:           scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:             scf.yield {{.*}} : vector<2x4xf32>
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield

// PACKED-LABEL: @matmul_narrow_static

// PACKED:       tensor.empty() : tensor<1x16x2x1xf32>
// PACKED:       scf.for
// PACKED:         vector.transfer_read
// PACKED:         vector.transfer_write
// PACKED:         scf.yield %{{.*}} : tensor<1x16x2x1xf32>
// PACKED:       }

// PACKED:       tensor.empty() : tensor<8x16x8x1xf32>
// PACKED-COUNT: scf.for
// PACKED:           vector.transpose
// PACKED:           scf.yield %{{.*}} : tensor<8x16x8x1xf32>
// PACKED:         scf.yield %{{.*}} : tensor<8x16x8x1xf32>

// PACKED:       tensor.empty() : tensor<1x8x2x8xf32>
// PACKED:       scf.for
// PACKED:         vector.transfer_read
// PACKED:         vector.transfer_write
// PACKED:         scf.yield %{{.*}} : tensor<1x8x2x8xf32>
// PACKED:       scf.for
// PACKED:         scf.for
// PACKED:           vector.contract
// PACKED:           scf.yield %{{.*}} : vector<1x1x2x8xf32>
// PACKED:         scf.yield

// PACKED:       tensor.empty() : tensor<2x64xf32>
// PACKED:       scf.for
// PACKED:         vector.transfer_read
// PACKED:         vector.transfer_write
// PACKED:         scf.yield %{{.*}} : tensor<2x64xf32>

// -----

func.func @matmul_small_static_peeling(%lhs: tensor<2x4xf32>, %arg1: tensor<4x6xf32>,
                         %output: tensor<2x6xf32>) -> tensor<2x6xf32> {
  %2 = linalg.matmul ins(%lhs, %arg1 : tensor<2x4xf32>, tensor<4x6xf32>)
                     outs(%output : tensor<2x6xf32>) -> tensor<2x6xf32>
  return %2 : tensor<2x6xf32>
}
// CHECK-LABEL: @matmul_small_static_peeling

// CHECK-NOT:     gml_st.parallel
// CHECK-NOT:     scf.for
// CHECK:         vector.contract

// -----

func.func @matvec_static(%lhs: tensor<1x16xf32>, %arg1: tensor<16x64xf32>,
                         %output: tensor<1x64xf32>) -> tensor<1x64xf32> {
  %2 = linalg.matmul ins(%lhs, %arg1 : tensor<1x16xf32>, tensor<16x64xf32>)
                     outs(%output : tensor<1x64xf32>) -> tensor<1x64xf32>
  return %2 : tensor<1x64xf32>
}
// CHECK-LABEL: @matvec_static

// CHECK:         gml_st.parallel
// CHECK:           vector.transfer_read
// CHECK-NEXT:      scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:             scf.yield {{.*}} : vector<1x4xf32>
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield
