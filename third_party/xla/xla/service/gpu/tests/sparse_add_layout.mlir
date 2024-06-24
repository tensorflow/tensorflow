// RUN: sparse-opt %s \
// RUN:   -convert-triton-to-tritongpu='target=cuda:80' \
// RUN:   -add-sparse-encoding -canonicalize \
// RUN: | FileCheck %s

// Note: 'canonicalize' folds redundant (back-and-forth) convert_layout ops.

// CHECK-DAG: #[[BLOCKED4x4:.*]] = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}> 
// CHECK-DAG: #[[BLOCKED1x1:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

module {
  // CHECK: @sparse_dot
  tt.func @sparse_dot() {
    // CHECK-NEXT: %[[A:.*]] = arith.constant dense<1.000000e+00>
    // CHECK-SAME:   : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[BLOCKED4x4]]}>> 
    // CHECK-NEXT: %[[B:.*]] = arith.constant dense<2.000000e+00>
    // CHECK-SAME:   : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[BLOCKED4x4]]}>> 
    // CHECK-NEXT: %[[C:.*]] = arith.constant dense<0.000000e+00>
    // CHECK-SAME:   : tensor<64x64xf32, #[[BLOCKED4x4]]> 
    // CHECK-NEXT: %[[META:.*]] = arith.constant dense<13107>
    // CHECK-SAME:   : tensor<64x4xi16, #triton_gpu.sparse_dot_meta<{parent = #[[BLOCKED4x4]]}>> 
    %a = arith.constant dense<1.00e+00> : tensor<64x32xf16>
    %b = arith.constant dense<2.00e+00> : tensor<64x64xf16>
    %c = arith.constant dense<0.00e+00> : tensor<64x64xf32>
    %meta = arith.constant dense<0x3333> : tensor<64x4xi16>

    // CHECK-NEXT: %[[D:.*]] = triton_gpu.sparse_dot %[[A]], %[[B]], %[[C]], %[[META]]
    // CHECK-SAME:   : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #[[BLOCKED4x4]]}>>
    // CHECK-SAME:     meta tensor<64x4xi16, #triton_gpu.sparse_dot_meta<{parent = #[[BLOCKED4x4]]}>>
    // CHECK-SAME:     * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #[[BLOCKED4x4]]}>>
    // CHECK-SAME:     -> tensor<64x64xf32, #[[BLOCKED4x4]]>
    %d = triton_gpu.sparse_dot %a, %b, %c, %meta
      : tensor<64x32xf16> meta tensor<64x4xi16> * tensor<64x64xf16> -> tensor<64x64xf32>

    // CHECK-NEXT: %[[CVT:.*]] = triton_gpu.convert_layout %[[D]]
    // CHECK-SAME:   : tensor<64x64xf32, #[[BLOCKED4x4]]>
    // CHECK-SAME:     -> tensor<64x64xf32, #[[BLOCKED1x1]]>
    // CHECK-NEXT: tt.print "" {hex = false} : %[[CVT]]
    // CHECK-SAME:   : tensor<64x64xf32, #[[BLOCKED1x1]]>
    // A use with side effects so we don't DCE the whole function.
    tt.print "" { hex = false } : %d : tensor<64x64xf32>

    // CHECK-NEXT: tt.return 
    tt.return
  }
}
