// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:   --gml-st-cpu-tiling-pipeline=matmul-tile-sizes=4,4,4 \
// RUN: | FileCheck %s

func.func @map_matmul(%lhs0: tensor<16x16xf32>, %rhs0: tensor<16x16xf32>,
    %lhs1: tensor<16x32xf32>, %rhs1: tensor<32x16xf32>) -> tensor<16x16xf32> {
  %init = tensor.empty() : tensor<16x16xf32>

  %cst = arith.constant 0.000000e+00 : f32
  %filled = linalg.fill ins(%cst : f32)
              outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>

  %4 = linalg.matmul ins(%lhs0, %rhs0 : tensor<16x16xf32>, tensor<16x16xf32>)
                     outs(%filled : tensor<16x16xf32>) -> tensor<16x16xf32>
  %5 = linalg.matmul ins(%lhs1, %rhs1 : tensor<16x32xf32>, tensor<32x16xf32>)
                     outs(%filled : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = linalg.map { math.absf }
         ins(%5 : tensor<16x16xf32>)
         outs(%init : tensor<16x16xf32>)

  %result = linalg.map { arith.addf }
              ins(%4, %6 : tensor<16x16xf32>, tensor<16x16xf32>)
              outs(%init : tensor<16x16xf32>)
  return %result : tensor<16x16xf32>
}

// CHECK-LABEL: @map_matmul

// Fuse this linalg.fill.

// CHECK-NOT:  linalg.fill
// CHECK:      scf.for
// CHECK:        scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:          scf.yield
// CHECK:        math.absf %{{.*}} : vector<4x4xf32>
// CHECK:        vector.transfer_write

// CHECK:      scf.for
// CHECK:        scf.for
// CHECK:        arith.addf %{{.*}} : vector<1x8xf32>
