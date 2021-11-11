// RUN: mlir-hlo-opt -disc-ral-lower-to-library-call %s -o - | FileCheck %s

// CHECK-LABEL: func @test_recv_input_and_send_output
// CHECK-SAME: (%[[CTX:.*]]: !disc_ral.context) {
func @test_recv_input_and_send_output(%arg0: !disc_ral.context) {
  // CHECK: %[[T0:.*]] = "disc_ral.dispatch"(%[[CTX]], %c0)
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index) -> memref<?x?xf32>

  // CHECK: %[[T1:.*]] = "disc_ral.dispatch"(%[[CTX]], %c1)
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_recv_input", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index) -> memref<?x?xf32>

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %c0, %[[T0]])
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index, memref<?x?xf32>) -> ()

  // CHECK: "disc_ral.dispatch"(%[[CTX]], %c1, %[[T1]])
  // CHECK-SAME: {backend_config = "cpu", call_target_name = "ral_send_output", has_side_effect = false} :
  // CHECK-SAME: (!disc_ral.context, index, memref<?x?xf32>) -> ()
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = "disc_ral.recv_input"(%arg0, %c0) : (!disc_ral.context, index) -> memref<?x?xf32>
  %1 = "disc_ral.recv_input"(%arg0, %c1) : (!disc_ral.context, index) -> memref<?x?xf32>
  "disc_ral.send_output"(%arg0, %c0, %0) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  "disc_ral.send_output"(%arg0, %c1, %1) : (!disc_ral.context, index, memref<?x?xf32>) -> ()
  return
}
