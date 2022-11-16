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
