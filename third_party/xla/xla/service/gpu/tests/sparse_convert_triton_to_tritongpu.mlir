// RUN: xla-opt %s -convert-triton-to-tritongpu='target=cuda:80' | FileCheck %s

module attributes {} {
  tt.func @gemm_fusion_dot_1_impl() {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %a = arith.constant dense<0.000000e+00> : tensor<32x16xbf16>
    // CHECK: %[[A:.+]] = triton_gpu.convert_layout {{.+}} : tensor<32x16xbf16, {{.+}}> -> tensor<32x16xbf16>
    %b = arith.constant dense<0.000000e+00> : tensor<32x32xbf16>
    // CHECK: %[[B:.+]] = triton_gpu.convert_layout {{.+}} : tensor<32x32xbf16, {{.+}}> -> tensor<32x32xbf16>
    %meta = arith.constant dense<0> : tensor<32x2xi16>
    // CHECK: %[[META:.+]] = triton_gpu.convert_layout {{.+}} : tensor<32x2xi16, {{.+}}> -> tensor<32x2xi16>
    %35:1 = scf.for %arg4 = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%arg8 = %acc) -> (tensor<32x32xf32>)  : i32 {
      // CHECK: %[[ACC:.+]] = triton_gpu.convert_layout {{.+}} : tensor<32x32xf32, {{.+}}> -> tensor<32x32xf32>
      // CHECK-NEXT: %[[D:.*]] = triton_xla.sparse_dot %[[A]], %[[B]], %[[ACC]], %[[META]]
      // CHECK-SAME:   : tensor<32x16xbf16> meta tensor<32x2xi16>
      // CHECK-SAME:     * tensor<32x32xbf16> -> tensor<32x32xf32>
      %74 = triton_xla.sparse_dot %a, %b, %arg8, %meta : tensor<32x16xbf16> meta tensor<32x2xi16> * tensor<32x32xbf16> -> tensor<32x32xf32>
      // CHECK: %[[ACC:.+]] = triton_gpu.convert_layout {{.+}} : tensor<32x32xf32> -> tensor<32x32xf32, {{.+}}>
      scf.yield %74 : tensor<32x32xf32>
    }
    tt.return
  }
}