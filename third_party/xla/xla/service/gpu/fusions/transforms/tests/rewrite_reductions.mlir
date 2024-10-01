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
    : tensor<128x1027xf32> to tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK: #[[$PAD_AND_RESHAPE:.*]] = #xla_gpu.indexing_map<"(d0, d1, d2, d3) -> (d0, d1 * 128 + d2 * 32 + d3),
// CHECK-SAME: domain: d0 in [0, 127], d1 in [0, 8], d2 in [0, 3], d3 in [0, 31], d1 * 128 + d2 * 32 + d3 in [0, 1026]
// CHECK-LABEL: @row_reduction
// CHECK-SAME:    %[[IN:.*]]: tensor<128x1027xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0.00
// CHECK:         %[[REINDEXED:.*]] = xla_gpu.reindex %[[IN]] at #[[$PAD_AND_RESHAPE]] default %[[C0]]
// CHECK:         %[[R1:.*]] = xla_gpu.reduce(%[[REINDEXED]]) inits(%[[C0]]) dimensions=[1] combiner=@add
// CHECK:         %[[R2:.*]] = xla_gpu.reduce(%[[R1]]) inits(%[[C0]]) dimensions=[2] combiner=@add
// CHECK:         %[[R3:.*]] = xla_gpu.reduce(%[[R2]]) inits(%[[C0]]) dimensions=[1] combiner=@add
// CHECK:         return %[[R3]] : tensor<128xf32>

// -----

func.func @add(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

func.func @row_reduction_with_major_reduced_dim(%arg0: tensor<2x42x128x32x8xf32>)
    -> tensor<2x128xf32> attributes {
    xla_gpu.launch_grid = #xla_gpu.launch_grid<
      block_counts = [42, 1, 1],
      thread_counts = [128, 1, 1]
    >
  } {
  %c0 = arith.constant 0.0 : f32
  %0 = xla_gpu.reduce (%arg0) inits(%c0) dimensions=[1, 3, 4] combiner=@add
    : tensor<2x42x128x32x8xf32> to tensor<2x128xf32>
  return %0 : tensor<2x128xf32>
}

// CHECK-LABEL: @row_reduction_with_major_reduced_dim
// CHECK:       %[[REINDEXED:.*]] = xla_gpu.reindex
// CHECK-SAME:    : tensor<2x42x128x32x8xf32> -> tensor<2x42x128x2x4x32xf32>
// CHECK:       xla_gpu.reduce(%[[REINDEXED]])
// CHECK-SAME:    dimensions=[1, 3]
// CHECK-SAME:    : tensor<2x42x128x2x4x32xf32>

// -----

func.func @add(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}

func.func @column(%arg0: tensor<2x32x32xf32>)
    -> tensor<2x32xf32> attributes {
    xla_gpu.launch_grid = #xla_gpu.launch_grid<
      block_counts = [42, 1, 1],
      thread_counts = [128, 1, 1]
    >
  } {
  %c0 = arith.constant 0.0 : f32
  %0 = xla_gpu.reduce (%arg0) inits(%c0) dimensions=[1] combiner=@add
    : tensor<2x32x32xf32> to tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK:       #[[$RESHAPE:.*]] = #xla_gpu.indexing_map<"(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)
// CHECK-SAME:    d1 * 4 + d2 in [0, 31]
// CHECK:       #[[$TRANSPOSE:.*]] = #xla_gpu.indexing_map<"(d0, d1, d2) -> (d0, d2, d1)
// CHECK-LABEL: @column
// CHECK-SAME:    %[[IN:.*]]: tensor<2x32x32xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0.00
// CHECK:         %[[REINDEXED:.*]] = xla_gpu.reindex %[[IN]] at #[[$RESHAPE]] default %[[C0]]
// CHECK-SAME:       -> tensor<2x8x4x32xf32>
// CHECK:         %[[R1:.*]] = xla_gpu.reduce(%[[REINDEXED]]) inits(%[[C0]]) dimensions=[1]
// CHECK-SAME:       to tensor<2x4x32xf32>
// CHECK:         %[[TRANSPOSED:.*]] = xla_gpu.reindex %[[R1]] at #[[$TRANSPOSE]]
// CHECK-SAME:       -> tensor<2x32x4xf32>
// CHECK:         %[[R2:.*]] = xla_gpu.reduce(%[[TRANSPOSED]]) inits(%[[C0]]) dimensions=[2]
// CHECK:         return %[[R2]] : tensor<2x32xf32>
