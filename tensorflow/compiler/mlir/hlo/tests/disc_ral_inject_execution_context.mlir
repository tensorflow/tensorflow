// RUN: mlir-hlo-opt -disc-ral-inject-execution-context=entry-func-name=test \
// RUN:  -canonicalize %s -o - | FileCheck %s

// CHECK-LABEL: func @test
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context) {
func @test(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
           %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>) -> (memref<?x?xf32>, memref<?x?xf32>) {
  // %[[T0:.*]] = "disc_ral.recv_input"(%[[CTX]], %c0) : (!disc_ral.context, index) -> memref<?x?xf32>
  // %[[T1:.*]] = "disc_ral.recv_input"(%[[CTX]], %c1) : (!disc_ral.context, index) -> memref<?x?xf32>
  // %[[T2:.*]] = "disc_ral.recv_input"(%[[CTX]], %c2) : (!disc_ral.context, index) -> memref<?x?xf32>
  // %[[T3:.*]] = "disc_ral.recv_input"(%[[CTX]], %c3) : (!disc_ral.context, index) -> memref<?x?xf32>
  // "lmhlo.abs"(%[[T0]], %[[T1]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  // "lmhlo.add"(%[[T1]], %[[T2]], %[[T3]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  // "disc_ral.send_output"(%[[CTX]], %c0, %[[T0]]) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  // "disc_ral.send_output"(%[[CTX]], %c1, %[[T3]]) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  "lmhlo.abs"(%arg0, %arg1) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  "lmhlo.add"(%arg1, %arg2, %arg3) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %arg1, %arg3 : memref<?x?xf32>, memref<?x?xf32>
}
