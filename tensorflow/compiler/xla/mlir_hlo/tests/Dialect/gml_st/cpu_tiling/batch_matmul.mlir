// RUN: mlir-hlo-opt %s --xla-cpu-transform-batch-matmul | FileCheck %s

func.func @batch_matmul(%lhs: tensor<8x64x32xf32>,
                        %rhs: tensor<8x32x64xf32>) -> tensor<8x64x64xf32> {
  %37 = tensor.empty() : tensor<8x64x64xf32>
  %cst_75 = arith.constant 0.000000e+00 : f32
  %38 = linalg.fill ins(%cst_75 : f32) outs(%37 : tensor<8x64x64xf32>)
    -> tensor<8x64x64xf32>
  %39 = linalg.batch_matmul ins(%lhs, %rhs : tensor<8x64x32xf32>,
    tensor<8x32x64xf32>) outs(%38 : tensor<8x64x64xf32>) -> tensor<8x64x64xf32>

  func.return %39 : tensor<8x64x64xf32>}

// CHECK-LABEL: @batch_matmul
// CHECK:       scf.forall
// CHECK:         linalg.fill
// CHECK:         linalg.matmul
// CHECK-SAME:      tensor<64x32xf32>, tensor<32x64xf32>
// CHECK-SAME:      tensor<64x64xf32>) -> tensor<64x64xf32>
