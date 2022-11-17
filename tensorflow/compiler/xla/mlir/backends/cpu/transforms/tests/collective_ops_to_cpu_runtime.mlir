// RUN: xla-cpu-opt %s -split-input-file -xla-lmhlo-to-cpu-runtime | FileCheck %s

func.func @partition_id() -> i32 {
  %0 = "xla_cpu.partition_id"() : () -> i32
  func.return %0 : i32
}

// CHECK-LABEL: @partition_id
// CHECK: call @xla.cpu.partition_id() : () -> i32

// CHECK: func private @xla.cpu.partition_id() -> i32 attributes {rt.custom_call = "xla.cpu.partition_id"}

// -----

func.func @replica_id() -> i32 {
  %0 = "xla_cpu.replica_id"() : () -> i32
  func.return %0 : i32
}

// CHECK-LABEL: @replica_id
// CHECK: call @xla.cpu.replica_id() : () -> i32

// CHECK: func private @xla.cpu.replica_id() -> i32 attributes {rt.custom_call = "xla.cpu.replica_id"}

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @all_reduce(%arg0: memref<32xf32, #map>, %arg1: memref<32xf32>) {
  "xla_cpu.all_reduce"(%arg0, %arg1) {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = 42 : i64,
    reduction_kind = 3 : i32
  } : (memref<32xf32, #map>, memref<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: @all_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: memref<32xf32,
//  CHECK-SAME:   %[[ARG1:.*]]: memref<32xf32>
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: memref.copy %[[ARG0]], %[[ALLOC]]
//       CHECK: call @xla.cpu.all_reduce(%[[ALLOC]], %[[ARG1]])
//  CHECK-SAME:   channel_handle = 42
//  CHECK-SAME:   op_id = 0
//  CHECK-SAME:   reduction_kind = 3
//  CHECK-SAME:   replica_groups = dense<
//       CHECK: func.func private @xla.cpu.all_reduce(
//  CHECK-SAME:     memref<32xf32>, memref<32xf32>)
//  CHECK-SAME:     attributes {rt.custom_call = "xla.cpu.all_reduce"}
