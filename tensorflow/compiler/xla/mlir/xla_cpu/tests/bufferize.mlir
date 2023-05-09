// RUN: xla-cpu-opt %s -split-input-file -empty-tensor-to-alloc-tensor \
// RUN:   -one-shot-bufferize | FileCheck %s

func.func @max_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = tensor.empty() : tensor<10xf32>
  %1 = "xla_cpu.all_reduce"(%arg0, %0) {
    channel_handle = 5 : i64,
    reduction_kind = 3 : i32,
    replica_groups = dense<[]> : tensor<0xi64>,
    use_global_device_ids = 0 : i32
  } : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK-LABEL: @max_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<10xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<10xf32>
//       CHECK: "xla_cpu.all_reduce"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   channel_handle = 5
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]

// -----

func.func @collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = tensor.empty() : tensor<16x8xf32>
  %1 = "xla_cpu.collective_permute"(%arg0, %0) {
    channel_handle = 1 : i64,
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<16x8xf32>, tensor<16x8xf32>) -> tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-LABEL: @collective_permute
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<16x8xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<16x8xf32>
//       CHECK: "xla_cpu.collective_permute"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   channel_handle = 1
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]

// -----

func.func @all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = tensor.empty() : tensor<16x4xf32>
  %1 = "xla_cpu.all_to_all"(%arg0, %0) {
    concat_dimension = 0 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_id_present = 0 : i32,
    op_id = 0 : i64,
    split_count = 4 : i64,
    split_dimension = 1 : i64
  } : (tensor<4x16xf32>, tensor<16x4xf32>) -> tensor<16x4xf32>
  return %1 : tensor<16x4xf32>
}

// CHECK-LABEL: @all_to_all
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<4x16xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} memref<16x4xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0_MEMREF]], %[[OUT]]) {
//  CHECK-SAME:   split_count = 4
//       CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[OUT]]
//       CHECK: return %[[RESULT]]


// -----

func.func @all_to_all_tuple(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>)
    -> (tensor<128x4xf32>, tensor<128x4xf32>) {
  %0 = tensor.empty() : tensor<128x4xf32>
  %1 = tensor.empty() : tensor<128x4xf32>
  %2:2 = "xla_cpu.all_to_all"(%arg0, %arg1, %0, %1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_id_present = 0 : i32,
    op_id = 0 : i64
  } : (tensor<128x4xf32>, tensor<128x4xf32>,
       tensor<128x4xf32>, tensor<128x4xf32>) ->
      (tensor<128x4xf32>, tensor<128x4xf32>)
  return %2#0, %2#1 : tensor<128x4xf32>, tensor<128x4xf32>
}

// CHECK-LABEL: @all_to_all_tuple
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<128x4xf32>,
//  CHECK-SAME:   %[[ARG1:.*]]: tensor<128x4xf32>
//   CHECK-DAG: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//   CHECK-DAG: %[[ARG1_MEMREF:.*]] = bufferization.to_memref %[[ARG1]]
//   CHECK-DAG: "xla_cpu.all_to_all"(%[[ARG0_MEMREF]], %[[ARG1_MEMREF]], %[[OUT0:.*]], %[[OUT1:.*]]) {
//   CHECK-DAG: %[[OUT0]] = memref.alloc() {{.*}} memref<128x4xf32>
//   CHECK-DAG: %[[OUT1]] = memref.alloc() {{.*}} memref<128x4xf32>
//   CHECK-DAG: %[[RESULT0:.*]] = bufferization.to_tensor %[[OUT0]] :
//   CHECK-DAG: %[[RESULT1:.*]] = bufferization.to_tensor %[[OUT1]] :
//       CHECK: return %[[RESULT0]], %[[RESULT1]]

// -----

func.func @fft(%arg0: tensor<3x5x4x8x256xf32>) -> tensor<3x5x4x8x129xcomplex<f32>> {
  %0 = tensor.empty() : tensor<3x5x4x8x129xcomplex<f32>>
  %1 = "xla_cpu.fft"(%arg0, %0) {
    fft_length = [4, 8, 256],
    fft_type = 2 : i32
   } : (tensor<3x5x4x8x256xf32>,tensor<3x5x4x8x129xcomplex<f32>>) -> tensor<3x5x4x8x129xcomplex<f32>>
  return %1 : tensor<3x5x4x8x129xcomplex<f32>>
}

// CHECK-LABEL: @fft
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<3x5x4x8x256xf32>
//       CHECK: %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0]]
//       CHECK: %[[OUT:.*]] = memref.alloc() {{.*}}
//       CHECK: "xla_cpu.fft"(%[[ARG0_MEMREF]], %[[OUT]])


// -----

func.func @rng_bit_generator(%state: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  %new_state_init = tensor.empty() : tensor<2xui64>
  %output_init = tensor.empty() : tensor<10x12xui32>
  %new_state, %output = "xla_cpu.rng_bit_generator"(%state, %new_state_init,
      %output_init) {
    rng_algorithm = #mhlo.rng_algorithm<DEFAULT>
  } : (tensor<2xui64>, tensor<2xui64>, tensor<10x12xui32>)
      -> (tensor<2xui64>, tensor<10x12xui32>)
  func.return %new_state, %output : tensor<2xui64>, tensor<10x12xui32>
}

// CHECK-LABEL: @rng_bit_generator
//  CHECK-SAME:   %[[STATE:.*]]: tensor
//       CHECK: %[[STATE_MEMREF:.*]] = bufferization.to_memref %[[STATE]]
//       CHECK: %[[STATE_OUT:.*]] = memref.alloc() {{.*}}<2xui64>
//       CHECK: %[[OUTPUT:.*]] = memref.alloc() {{.*}}<10x12xui32>
//       CHECK: "xla_cpu.rng_bit_generator"(%[[STATE_MEMREF]], %[[STATE_OUT]], %[[OUTPUT]])