// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-rewrite-reductions | \
// RUN:   FileCheck %s

func.func @add(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

func.func @row_reduction(%arg0: tensor<128x1027xf32>)
    -> tensor<128xf32> attributes {
    xla_gpu.launch_grid = #xla_gpu.launch_grid<
      block_counts = [42, 1, 1],
      thread_counts = [128, 1, 1]
    >
  } {
  %c0 = arith.constant 0.0 : f32
  %0 = xla_gpu.reduce (%arg0) inits(%c0) dimensions=[1] combiner=@add
    : tensor<128x1027xf32>
  return %0 : tensor<128xf32>
}

// CHECK: #[[$PAD_AND_PROJECT:.*]] = #xla_gpu.indexing_map<(d0, d1, d2) -> (d0, d1 * 128 + d2),
// CHECK-SAME: domain: d0 in [0, 127], d1 in [0, 8], d2 in [0, 127], d1 * 128 + d2 in [0, 1026], d2 + d1 * 128 in [0, 1026]
// CHECK: #[[$TO_WARP:.*]] = #xla_gpu.indexing_map<(d0, d1) -> (d0, d1 floordiv 32, d1 mod 32)
// CHECK-LABEL: @row_reduction
// CHECK-SAME:    %[[IN:.*]]: tensor<128x1027xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0.00
// CHECK:         %[[PADDED:.*]] = xla_gpu.reindex %[[IN]] at #[[$PAD_AND_PROJECT]] default %[[C0]]
// CHECK:         %[[R1:.*]] = xla_gpu.reduce(%[[PADDED]]) inits(%[[C0]]) dimensions=[1] combiner=@add
// CHECK:         %[[RESHAPED:.*]] = xla_gpu.reindex %[[R1]] at #[[$TO_WARP]] default %[[C0]]
// CHECK:         %[[R2:.*]] = xla_gpu.reduce(%[[RESHAPED]]) inits(%[[C0]]) dimensions=[2] combiner=@add
// CHECK:         %[[R3:.*]] = xla_gpu.reduce(%[[R2]]) inits(%[[C0]]) dimensions=[1] combiner=@add
// CHECK:         return %[[R3]] : tensor<128xf32>
