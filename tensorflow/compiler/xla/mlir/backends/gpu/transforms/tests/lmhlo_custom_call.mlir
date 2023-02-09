// RUN: xla-gpu-opt %s -split-input-file -xla-lmhlo-to-gpu-runtime \
// RUN:   | FileCheck %s

// CHECK: func @test
// CHECK:   %[[ARG0:.*]]: memref<f32>
// CHECK: )
func.func @test(%arg0: memref<f32>) {
  // CHECK: call @[[CUSTOM_CALL:.*]](%[[ARG0]])
  // CHECK-SAME:   api_version = 2 : i32
  // CHECK-SAME:   backend_config = ""
  // CHECK-SAME:   call_target_name = "target"
  // CHECK-SAME: : (memref<f32>) -> ()
  "lmhlo.custom_call"(%arg0) ({}) {
    api_version = 2 : i32,
    backend_config = "",
    call_target_name = "target",
    operand_segment_sizes = array<i32: 0, 1>
  } : (memref<f32>) -> ()
  return
}

// CHECK: func.func private @[[CUSTOM_CALL]](memref<f32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.custom_call"}

// -----

// CHECK: func @test_with_mapping
// CHECK:   %[[ARG0:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG1:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG2:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG3:[0-9a-z]*]]: memref<f32>,
// CHECK:   %[[ARG4:[0-9a-z]*]]: memref<f32>
// CHECK: )
func.func @test_with_mapping(
    %arg0: memref<f32>,
    %arg1: memref<f32>,
    %arg2: memref<f32>,
    %arg3: memref<f32>,
    %arg4: memref<f32>) {
  // CHECK: %[[HOLE:.*]] = arith.constant -1 : i64

  // CHECK: call @[[CUSTOM_CALL:.*]](%[[ARG0]], %[[HOLE]], %[[ARG1]], %[[HOLE]],
  // CHECK-SAME:  %[[ARG2]], %[[ARG3]], %[[HOLE]], %[[ARG4]])
  // CHECK-SAME:   api_version = 1 : i32
  // CHECK-SAME:   backend_config = ""
  // CHECK-SAME:   call_target_name = "target"
  "lmhlo.custom_call"(%arg0, %arg1, %arg2, %arg3, %arg4) ({}) {
    api_version = 1 : i32,
    backend_config = "",
    call_target_name = "target",
    operand_segment_sizes = array<i32: 2, 3>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 4,
      args_to_target_args = [0, 2],
      results_to_target_results = [0, 1, 3]>
    } : (memref<f32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> ()

  return
}

// CHECK: func.func private @[[CUSTOM_CALL]](memref<f32>, i64, memref<f32>, i64,
// CHECK-SAME: memref<f32>, memref<f32>, i64, memref<f32>)
// CHECK-SAME: attributes {rt.custom_call = "xla.gpu.custom_call"}
