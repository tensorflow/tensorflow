// RUN: mlir-hlo-opt %s -lhlo-legalize-to-gpu -split-input-file | FileCheck %s

func @reduce(%arg: memref<100x10xf32>,
             %init: memref<f32>,
             %result: memref<100xf32>) {
  "lmhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[1]> : tensor<1xi64>}
      : (memref<100x10xf32>, memref<f32>, memref<100xf32>) -> ()
  return
}

// CHECK-DAG: #[[$MAP:.*]] = affine_map<()[s0] -> (s0)>

//     CHECK: func @reduce(%[[ARG0:.*]]: memref<100x10xf32>, %[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<100xf32>) {
// CHECK-DAG: %[[C100:.*]] = arith.constant 100 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//     CHECK: gpu.launch blocks({{.*}}, {{.*}}, {{.*}}) in ({{.*}} = %[[C1]], {{.*}} = %[[C1]], {{.*}} = %[[C1]]) threads(%[[IDX:.*]], {{.*}}, {{.*}}) in ({{.*}} = %[[C100]], {{.*}} = %[[C1]], {{.*}} = %[[C1]]) {
//     CHECK:   %[[ACC:.*]] = memref.load %[[ARG1]][] : memref<f32>
//     CHECK:   store %[[ACC]], %[[ARG2]][%[[IDX:.*]]] : memref<100xf32>
// CHECK-DAG:   %[[LB:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[UB:.*]] = arith.constant 10 : index
// CHECK-DAG:   %[[STEP:.*]] = arith.constant 1 : index
//     CHECK:   scf.for %[[IDX1:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
//     CHECK:     %[[LHS:.*]] = memref.subview %[[ARG2]][%[[IDX]]] [1] [1] : memref<100xf32> to memref<f32, #[[$MAP]]>
//     CHECK:     %[[RHS:.*]] = memref.subview %[[ARG0]][%[[IDX]], %[[IDX1]]] [1, 1] [1, 1] : memref<100x10xf32> to memref<f32, #[[$MAP]]>
//     CHECK:     "lmhlo.add"(%[[LHS]], %[[RHS]], %[[LHS]]) : (memref<f32, {{.*}}>, memref<f32, {{.*}}>, memref<f32, {{.*}}>) -> ()
//     CHECK:   }
//     CHECK:   gpu.terminator
//     CHECK: }
//     CHECK: return
