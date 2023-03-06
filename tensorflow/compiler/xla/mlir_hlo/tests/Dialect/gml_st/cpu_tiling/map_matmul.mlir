// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:   --gml-st-cpu-tiling-pipeline=matmul-tile-sizes=4,4,4 | FileCheck %s

func.func @map_matmul(%arg0: tensor<?x?xf32>,
    %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
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

// CHECK-LABEL: @map_matmul

// CHECK:      scf.forall
// CHECK:        scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK:          linalg.matmul
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK-COUNT-2:     vector.transfer_read
// CHECK:             vector.contract
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK:          linalg.matmul
// CHECK:          scf.yield
// CHECK:        math.absf %{{.*}} : vector<4x4xf32>
// CHECK:        arith.addf %{{.*}} : vector<4x4xf32>
// CHECK:        tensor.parallel_insert_slice

// CHECK:      scf.forall
// CHECK:        scf.for
// CHECK:          linalg.matmul
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK:          linalg.matmul
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            math.absf
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            arith.addf
// CHECK:        tensor.parallel_insert_slice

// CHECK:      scf.forall
// CHECK:        scf.for
// CHECK:          linalg.matmul
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK:          linalg.matmul
// CHECK:          scf.yield
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            math.absf
// CHECK:        scf.for
// CHECK:          scf.for
// CHECK:            arith.addf
// CHECK:        tensor.parallel_insert_slice
