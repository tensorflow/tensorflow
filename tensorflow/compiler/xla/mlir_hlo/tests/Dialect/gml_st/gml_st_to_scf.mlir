// RUN: mlir-hlo-opt %s -gml-st-to-scf -split-input-file | FileCheck %s

func.func @for(%A: memref<192x192xf32>) {
   %c24 = arith.constant 24 : index
   %c16 = arith.constant 16 : index
   %c0 = arith.constant 0 : index
   %c192 = arith.constant 192 : index
   %cst = arith.constant 0.000000e+00 : f32

  scf.forall (%i, %j) = (%c0, %c0) to (%c192, %c192) step (%c24, %c16) {
    linalg.fill ins(%cst : f32) outs(%A : memref<192x192xf32>)
  }
  func.return
}

// CHECK-LABEL: @for
// CHECK-DAG:   %[[C24:.*]] = arith.constant 24 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C192:.*]] = arith.constant 192 : index
// CHECK:       scf.parallel
// CHECK-SAME:  %{{.*}} = (%[[C0]], %[[C0]]) to (%[[C192]], %[[C192]]) step (%[[C24]], %[[C16]])
// CHECK:         linalg.fill
