// RUN: not tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s 2>&1 | FileCheck %s

// CHECK: Only dense elements attr are supported
func.func @main() {
  %0 = "mhlo.constant"() {value = opaque<"mhlo", "0x0123456789ABCDEF"> : tensor<4xf32>} : () -> tensor<4xf32>
  func.return
}

// -----

// Tests dynamic result shape

// CHECK: 'mhlo.all_gather' op can't be translated to XLA HLO
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x?xf32> {
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x32xf32>) -> tensor<128x?xf32>
  func.return %0 : tensor<128x?xf32>
}

// -----

// Tests dynamic operand shape

// CHECK: 'mhlo.all_gather' op can't be translated to XLA HLO
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x?xf32> {
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x32xf32>) -> tensor<128x?xf32>
  func.return %0 : tensor<128x?xf32>
}
