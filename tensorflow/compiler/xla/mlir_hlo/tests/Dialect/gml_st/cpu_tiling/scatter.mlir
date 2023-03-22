// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --gml-st-cpu-tiling-pipeline=enable-fusion-clusters=false | \
// RUN: FileCheck %s

func.func @scatter_fusion(%indices: tensor<?x2xindex>,
  %updates: tensor<?x?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %updates, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %updates, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %updates, %c2 : tensor<?x?x?xf32>
  %init0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %abs = linalg.map { math.absf }
          ins(%updates:tensor<?x?x?xf32>)
          outs(%init0:tensor<?x?x?xf32>)

  %result = thlo.scatter
    ins (%indices: tensor<?x2xindex>, %abs: tensor<?x?x?xf32>)
    outs (%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      %0 = arith.addf %in, %out: f32
      thlo.yield %0: f32
    }
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @scatter_fusion

// CHECK:         scf.for
// CHECK:           scf.if
// CHECK:             scf.for
// CHECK:               math.absf
// CHECK:             scf.for
// CHECK:               math.absf
// CHECK:             linalg.reduce
// CHECK:             scf.yield {{.*}} : tensor<?x?xf32>

// -----

func.func @scatter_fusion_overwrite(%indices: tensor<?x2xindex>,
  %updates: tensor<?x?x?xf32>, %init: tensor<?x?xf32>) -> tensor<?x?xf32> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %updates, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %updates, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %updates, %c2 : tensor<?x?x?xf32>
  %init0 = tensor.empty(%dim0, %dim1, %dim2) : tensor<?x?x?xf32>
  %abs = linalg.map { math.absf }
          ins(%updates:tensor<?x?x?xf32>)
          outs(%init0:tensor<?x?x?xf32>)

  %result = thlo.scatter
    ins (%indices: tensor<?x2xindex>, %abs: tensor<?x?x?xf32>)
    outs (%init: tensor<?x?xf32>)
    (%in: f32, %out: f32) {
      thlo.yield %in: f32
    }
  return %result : tensor<?x?xf32>
}
// CHECK-LABEL: @scatter_fusion_overwrite

// CHECK:         scf.for
// CHECK:           scf.if
// CHECK:             scf.for
// CHECK:               math.absf
// CHECK:             scf.for
// CHECK:               math.absf
// CHECK:             scf.yield {{.*}} : tensor<?x?xf32>
