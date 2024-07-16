// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// CHECK:  HloModule
// CHECK: all-reduce{{.*}} channel_id=1
// CHECK-NEXT: all-reduce{{.*}} channel_id=2
func.func @main(%arg0: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %0 = "mhlo.all_reduce"(%arg0) ({
  // Duplicate channel ID must be de-duped
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<10xf32>) -> tensor<10xf32>
  %1 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0, %1 : tensor<10xf32>, tensor<10xf32>
}
