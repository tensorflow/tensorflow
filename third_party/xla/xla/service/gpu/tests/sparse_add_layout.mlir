// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:80 num-warps=4' | FileCheck %s

// CHECK-COUNT-4: #triton_gpu.blocked
module {
  tt.func @sparse_dot() {
    %A = arith.constant dense<1.00e+00> : tensor<64x32xf16>
    %meta = arith.constant dense<0x3333> : tensor<64x4xi16>
    %B = arith.constant dense<2.00e+00> : tensor<64x64xf16>
    %C = arith.constant dense<0.00e+00> : tensor<64x64xf32>
    // CHECK-COUNT-4: triton_gpu.convert_layout
    // CHECK: triton_gpu.sparse_dot {{.+}} #triton_gpu.sparse_dot_meta
    %D = triton_gpu.sparse_dot %A, %B, %C, %meta : tensor<64x32xf16> meta tensor<64x4xi16> * tensor<64x64xf16> -> tensor<64x64xf32>
    tt.return
  }
}
