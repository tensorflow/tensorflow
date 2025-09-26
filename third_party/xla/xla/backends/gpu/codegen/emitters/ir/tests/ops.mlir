// RUN: emitters_opt %s --split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: emitters_opt %s --split-input-file | emitters_opt --split-input-file | FileCheck %s
// Verify the generic form can be parsed.
// RUN: emitters_opt %s --split-input-file --mlir-print-op-generic | emitters_opt --split-input-file | FileCheck %s

func.func @shared_and_sync() -> (tensor<2xf32>, tensor<2xf32>) {
  %shared1 = xla_gpu.allocate_shared : tensor<2xf32>
  %shared2 = xla_gpu.allocate_shared : tensor<2xf32>
  %sync:2 = xla_gpu.sync_threads %shared1, %shared2
    : tensor<2xf32>, tensor<2xf32>
  return %sync#0, %sync#1 : tensor<2xf32>, tensor<2xf32>
}
// CHECK-LABEL: @shared_and_sync
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: allocate_shared
// CHECK-NEXT: sync_threads
// CHECK-NEXT: return

// -----

func.func @add(%a_acc: f32, %b_acc: i32, %a: f32, %b: i32)
    -> (f32, i32) {
  %0 = arith.addf %a_acc, %a : f32
  %1 = arith.addi %b_acc, %b : i32
  func.return %0, %1 : f32, i32
}
func.func @reduce(%in0: tensor<16x8x4xf32>, %init0: f32,
    %in1: tensor<16x8x4xi32>, %init1: i32) -> (tensor<8xf32>, tensor<8xi32>) {
  %sum:2 = xla_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
    combiner=@add {xla.range = [0 : index, 42 : index]}
    : tensor<16x8x4xf32>, tensor<16x8x4xi32> to tensor<8xf32>, tensor<8xi32>
  func.return %sum#0, %sum#1 : tensor<8xf32>, tensor<8xi32>
}
// CHECK-LABEL: func.func @reduce(
// CHECK-SAME:    %[[IN1:.*]]: tensor<16x8x4xf32>, %[[INIT1:.*]]: f32,
// CHECK-SAME:    %[[IN2:.*]]: tensor<16x8x4xi32>, %[[INIT2:.*]]: i32)

// CHECK:        xla_gpu.reduce(%[[IN1]], %[[IN2]])
// CHECK-SAME:    inits(%[[INIT1]], %[[INIT2]]) dimensions=[0, 2]
// CHECK-SAME:    combiner=@add {xla.range = [0 : index, 42 : index]}
// CHECK-SAME:    : tensor<16x8x4xf32>, tensor<16x8x4xi32> to tensor<8xf32>, tensor<8xi32>

// -----

func.func @add(%a_acc: f32, %a: f32) -> (f32) {
  %0 = arith.addf %a_acc, %a : f32
  func.return %0 : f32
}

func.func @reduce_middle_dim(%in: tensor<16x8x4xf32>, %init: f32)
    -> (tensor<16x4xf32>) {
  %sum = xla_gpu.reduce (%in) inits(%init) dimensions=[1]
    combiner=@add : tensor<16x8x4xf32> to tensor<16x4xf32>
  func.return %sum : tensor<16x4xf32>
}

// CHECK-LABEL: func.func @reduce_middle_dim(
// CHECK-SAME:    %[[IN:.*]]: tensor<16x8x4xf32>, %[[INIT:.*]]: f32)
// CHECK:        xla_gpu.reduce(%[[IN]])
// CHECK-SAME:    inits(%[[INIT]]) dimensions=[1]
// CHECK-SAME:    combiner=@add
// CHECK-SAME:    : tensor<16x8x4xf32> to tensor<16x4xf32>

// -----

func.func @do_nothing(%a: f32, %b: i32, %c: f32, %d: i32) -> (f32, i32) {
  return %a, %b : f32, i32
}
func.func @shuffler(%a: f32, %b: i32) -> (f32, i32) {
  %ret:2 = xla_gpu.shuffle_reduce(%a, %b) to 4 combiner=@do_nothing
    {xla.range = [0 : index, 42 : index]} : f32, i32
  return %ret#0, %ret#1 : f32, i32
}
// CHECK-LABEL: func.func @shuffler(
// CHECK-SAME:    %[[IN1:.*]]: f32, %[[IN2:.*]]: i32)

// CHECK:        xla_gpu.shuffle_reduce(%[[IN1]], %[[IN2]]) to 4
// CHECK-SAME:    combiner=@do_nothing {xla.range = [0 : index, 42 : index]}
// CHECK-SAME:    : f32, i32
