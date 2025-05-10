// RUN: emitters_opt %s --split-input-file -debug
func.func @load(%arg0: !xla_cpu.call_frame) -> tensor<32x32xf32> {
  %0 = xla_cpu.load %arg0, 0 : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: @load(
// CHECK:   %[[ARG0:.+]]: !xla_cpu.call_frame
// CHECK: ) -> tensor<32x32xf32> {
// CHECK:   %[[LOAD:.+]] = xla_cpu.load %[[ARG0]], 0 : tensor<32x32xf32>
// CHECK:   return %[[LOAD]] : tensor<32x32xf32>
// CHECK: }

// -----

func.func @store(%arg0: !xla_cpu.call_frame, %arg1: tensor<32x32xf32>) {
  xla_cpu.store %arg1 into %arg0, 0 : tensor<32x32xf32>
  return
}

// CHECK-LABEL: @store(
// CHECK:   %[[ARG0:.+]]: !xla_cpu.call_frame,
// CHECK:   %[[ARG1:.+]]: tensor<32x32xf32>
// CHECK: ) {
// CHECK:   xla_cpu.store %[[ARG1]] into %[[ARG0]], 0 : tensor<32x32xf32>
// CHECK:   return
// CHECK: }

// -----

func.func @thread_id(%arg0: !xla_cpu.call_frame) -> (index, index, index) {
  %thread_id_x = xla_cpu.thread_id x in %arg0
  %thread_id_y = xla_cpu.thread_id y in %arg0
  %thread_id_z = xla_cpu.thread_id z in %arg0
  func.return %thread_id_x, %thread_id_y, %thread_id_z : index, index, index
}
