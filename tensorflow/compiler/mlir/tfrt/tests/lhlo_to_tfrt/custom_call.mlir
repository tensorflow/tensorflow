// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @custom_call(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @custom_call(%input0: memref<2x2xf32>, %input1: memref<2x2xf32>, %output0: memref<2x2xf32>, %output1: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN:%[0-9]+]] = xlir.custom_call
  // CHECK-SAME: %arg1, %arg2, %arg3, %arg4, %arg5, %arg0
  // CHECK-SAME: indices = [0 : i32, -1 : i32, -1 : i32, 1 : i32, -1 : i32, 2 : i32, 3 : i32],
  // CHECK-SAME: opaque = "my_config",
  // CHECK-SAME: symbol = "my_target"

  "lmhlo.custom_call"(%input0, %input1, %output0, %output1) {
      backend_config = "my_config",
      call_target_name = "my_target",
      has_side_effects = false,
      operand_segment_sizes = dense<2> : vector<2xi32>,
      target_arg_mapping = {
        num_args = 4 : i64,
        num_results = 3 : i64,
        args_to_target_args = [0,3],
        results_to_target_results = [1,2]
      }
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
