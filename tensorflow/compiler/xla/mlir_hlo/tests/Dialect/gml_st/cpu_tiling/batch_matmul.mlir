// RUN: mlir-hlo-opt %s \
// RUN:   --gml-st-cpu-tiling-pipeline=matmul-tile-sizes=4,4,4 \
// RUN: | FileCheck %s

func.func @batch_matmul(%lhs: tensor<8x64x32xf32>,
                        %rhs: tensor<8x32x64xf32>) -> tensor<8x64x64xf32> {
  %37 = tensor.empty() : tensor<8x64x64xf32>
  %cst_75 = arith.constant 0.000000e+00 : f32
  %38 = linalg.fill ins(%cst_75 : f32) outs(%37 : tensor<8x64x64xf32>)
    -> tensor<8x64x64xf32>
  %39 = linalg.batch_matmul ins(%lhs, %rhs : tensor<8x64x32xf32>,
    tensor<8x32x64xf32>) outs(%38 : tensor<8x64x64xf32>) -> tensor<8x64x64xf32>

  func.return %39 : tensor<8x64x64xf32>
}
// CHECK-LABEL: @batch_matmul

// CHECK:      scf.for
// CHECK-DAG:    tensor.collapse_shape
// CHECK-SAME:       : tensor<1x64x32xf32> into tensor<64x32xf32>
// CHECK-DAG:    tensor.collapse_shape
// CHECK-SAME:       : tensor<1x32x64xf32> into tensor<32x64xf32>
// CHECK-DAG:    tensor.collapse_shape
// CHECK-SAME:       : tensor<1x64x64xf32> into tensor<64x64xf32>
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            scf.for
// CHECK:              vector.contract
// CHECK-SAME:           : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
// CHECK:              scf.yield %{{.*}} : vector<4x4xf32>
// CHECK:            scf.yield %{{.*}} : tensor<64x64xf32>
// CHECK:          scf.yield %{{.*}} : tensor<64x64xf32>
// CHECK:        %expanded = tensor.expand_shape
// CHECK:          : tensor<64x64xf32> into tensor<1x64x64xf32>
// CHECK:        scf.yield %inserted_slice : tensor<8x64x64xf32>
