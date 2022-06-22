// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @all_gather(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @all_gather(%operand0: memref<2x2xf32>, %operand1: memref<2x2xf32>, %result0: memref<2x2xf32>, %result1: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = xlir.ccl.create [[CONTEXT]]
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.ccl.all_gather [[HANDLE]],
  // CHECK-SAME: %arg2, %arg4, ncclFloat32, %arg0
  // CHECK: [[CHAIN2:%[0-9]+]] = tfrt_gpu.ccl.all_gather [[HANDLE]],
  // CHECK-SAME: %arg3, %arg5, ncclFloat32, [[CHAIN1]]
  // CHECK: [[CHAIN3:%[0-9]+]] = tfrt_gpu.ccl.execute %arg1, [[HANDLE]],
  // CHECK-SAME: [[CHAIN2]]

  "lmhlo.all_gather"(%operand0, %operand1, %result0, %result1) {
      all_gather_dimension = 0 : i64,
      replica_groups = dense<0> : tensor<1x1xi64>,
      channel_id = #mhlo.channel_handle<handle = 5, type = 2>,
      constrain_layout = false,
      use_global_device_ids = false
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN3]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @all_reduce(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @all_reduce(%operand0: memref<2x2xf32>, %operand1: memref<2x2xf32>, %result0: memref<2x2xf32>, %result1: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = xlir.ccl.create [[CONTEXT]]
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.ccl.all_reduce [[HANDLE]],
  // CHECK-SAME: %arg2, %arg4, ncclFloat32, ncclSum, %arg0
  // CHECK: [[CHAIN2:%[0-9]+]] = tfrt_gpu.ccl.all_reduce [[HANDLE]],
  // CHECK-SAME: %arg3, %arg5, ncclFloat32, ncclSum, [[CHAIN1]]
  // CHECK: [[CHAIN3:%[0-9]+]] = tfrt_gpu.ccl.execute %arg1, [[HANDLE]],
  // CHECK-SAME: [[CHAIN2]]

  "lmhlo.all_reduce"(%operand0, %operand1, %result0, %result1) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
          %0 = mhlo.add %lhs, %rhs : tensor<f32>
          "mhlo.return"(%0) : (tensor<f32>) -> ()
      }) {
          replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
          channel_id = #mhlo.channel_handle<handle = 5, type = 2>,
          constrain_layout = true,
          use_global_device_ids = true
      } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN3]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @reduce_scatter(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @reduce_scatter(%operand0: memref<2x2xf32>, %operand1: memref<2x2xf32>, %result0: memref<2x2xf32>, %result1: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = xlir.ccl.create [[CONTEXT]]
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.ccl.reduce_scatter [[HANDLE]],
  // CHECK-SAME: %arg2, %arg4, ncclFloat32, ncclSum, %arg0
  // CHECK: [[CHAIN2:%[0-9]+]] = tfrt_gpu.ccl.reduce_scatter [[HANDLE]],
  // CHECK-SAME: %arg3, %arg5, ncclFloat32, ncclSum, [[CHAIN1]]
  // CHECK: [[CHAIN3:%[0-9]+]] = tfrt_gpu.ccl.execute %arg1, [[HANDLE]],
  // CHECK-SAME: [[CHAIN2]]

  "lmhlo.reduce_scatter"(%operand0, %operand1, %result0, %result1) ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
          %0 = mhlo.add %lhs, %rhs : tensor<f32>
          "mhlo.return"(%0) : (tensor<f32>) -> ()
      }) {
          replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
          channel_id = #mhlo.channel_handle<handle = 5, type = 2>,
          scatter_dimension = 1 : i64
      } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN3]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @all_to_all(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @all_to_all(%operand0: memref<2x2xf32>, %operand1: memref<2x2xf32>, %result0: memref<2x2xf32>, %result1: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = xlir.ccl.create [[CONTEXT]]
  // CHECK: [[PEER0:%[0-9]+]] = tfrt.constant.i32 0
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.ccl.send [[HANDLE]],
  // CHECK-SAME: %arg2, [[PEER0]], ncclFloat32, %arg0
  // CHECK: [[CHAIN2:%[0-9]+]] = tfrt_gpu.ccl.recv [[HANDLE]],
  // CHECK-SAME: %arg4, [[PEER0]], ncclFloat32, [[CHAIN1]]
  // CHECK: [[PEER1:%[0-9]+]] = tfrt.constant.i32 1
  // CHECK: [[CHAIN3:%[0-9]+]] = tfrt_gpu.ccl.send [[HANDLE]],
  // CHECK-SAME: %arg3, [[PEER1]], ncclFloat32, [[CHAIN2]]
  // CHECK: [[CHAIN4:%[0-9]+]] = tfrt_gpu.ccl.recv [[HANDLE]],
  // CHECK-SAME: %arg5, [[PEER1]], ncclFloat32, [[CHAIN3]]
  // CHECK: [[CHAIN5:%[0-9]+]] = tfrt_gpu.ccl.execute %arg1, [[HANDLE]],
  // CHECK-SAME: [[CHAIN4]]

  "lmhlo.all_to_all"(%operand0, %operand1, %result0, %result1) {
      replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>,
      channel_id = #mhlo.channel_handle<handle = 1, type = 0>,
      constrain_layout = false,
      use_global_device_ids = false
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN5]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @all_to_all_split_dimension(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @all_to_all_split_dimension(%operand0: memref<2x2xf32>, %operand1: memref<2x2xf32>, %result0: memref<2x2xf32>, %result1: memref<2x2xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = xlir.ccl.create [[CONTEXT]]
  // CHECK: [[CHAIN1:%[0-9]+]] = tfrt_gpu.ccl.all_to_all [[HANDLE]],
  // CHECK-SAME: %arg2, %arg4, ncclFloat32, %arg0
  // CHECK: [[CHAIN2:%[0-9]+]] = tfrt_gpu.ccl.all_to_all [[HANDLE]],
  // CHECK-SAME: %arg3, %arg5, ncclFloat32, [[CHAIN1]]
  // CHECK: [[CHAIN3:%[0-9]+]] = tfrt_gpu.ccl.execute %arg1, [[HANDLE]],
  // CHECK-SAME: [[CHAIN2]]

  "lmhlo.all_to_all"(%operand0, %operand1, %result0, %result1) {
      replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>,
      channel_id = #mhlo.channel_handle<handle = 1, type = 0>,
      constrain_layout = false,
      use_global_device_ids = false,
      split_dimension = 0
  } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN3]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @collective_permute(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @collective_permute(%operand: memref<2x2xf32>, %result: memref<2x2xf32>)
  attributes {replica_count = 0 : i32, num_partitions = 0 : i32} {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CONTEXT:%[0-9]+]] = tfrt_gpu.stream.get_context %arg1
  // CHECK: [[HANDLE:%[0-9]+]] = xlir.ccl.create [[CONTEXT]]
  // CHECK: [[CHAIN1:%[0-9]+]] = xlir.ccl.collective_permute [[HANDLE]],
  // CHECK-SAME: %arg2, %arg3, ncclFloat32, 1, [0, 1, 2], [1, 2, 3], %arg0
  // CHECK: [[CHAIN2:%[0-9]+]] = tfrt_gpu.ccl.execute %arg1, [[HANDLE]],
  // CHECK-SAME: [[CHAIN1]]

  "lmhlo.collective_permute"(%operand, %result) {
      source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
      channel_id = #mhlo.channel_handle<handle = 5, type = 2>
  } : (memref<2x2xf32>, memref<2x2xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN2]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
