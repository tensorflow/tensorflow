// RUN: xla-opt %s -split-input-file -sparse-blocked-to-mma | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
// CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_blocked_to_mma_ampere
  tt.func @sparse_blocked_to_mma_ampere(%A: tensor<64x32xf16, #lhs>, %B: tensor<64x64xf16, #rhs>, %meta: tensor<64x4xi16, #blocked>) -> tensor<64x64xf32, #blocked> {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK-DAG: %[[LHS:.+]] = ttg.convert_layout {{.+}} : {{.+}} -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$MMA]], kWidth = 2}>>
    // CHECK-DAG: %[[RHS:.+]] = ttg.convert_layout {{.+}} : {{.+}} -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$MMA]], kWidth = 2}>>
    // CHECK-DAG: %[[ACC:.+]] = ttg.convert_layout {{.+}} : {{.+}} -> tensor<64x64xf32, #[[$MMA]]>
    // CHECK-DAG: %[[META:.+]] = ttg.convert_layout {{.+}} : {{.+}} -> tensor<64x4xi16, #triton_xla.sparse_dot_meta<{parent = #[[$MMA]]}>>
    // CHECK: %[[OUT:.+]] = triton_xla.sparse_dot %[[LHS]], %[[RHS]], %[[ACC]], %[[META]] : {{.+}} -> tensor<64x64xf32, #[[$MMA]]>
    %D = triton_xla.sparse_dot %A, %B, %C, %meta : tensor<64x32xf16, #lhs> meta tensor<64x4xi16, #blocked> * tensor<64x64xf16, #rhs> -> tensor<64x64xf32, #blocked>
    // CHECK: ttg.convert_layout %[[OUT]] : tensor<64x64xf32, #[[$MMA]]> -> tensor<64x64xf32, #blocked>
    tt.return %D : tensor<64x64xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
// CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: sparse_blocked_to_mma_hopper
  tt.func @sparse_blocked_to_mma_hopper(%A: tensor<64x32xf16, #lhs>, %B: tensor<64x64xf16, #rhs>, %meta: tensor<64x4xi16, #blocked>) -> tensor<64x64xf32, #blocked> {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK-DAG: %[[LHS_TEMP:.+]] = ttg.convert_layout {{.+}} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<64x32xf16, #blocked>
    // CHECK-DAG: %[[LHS:.+]] = ttg.local_alloc %[[LHS_TEMP]] : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #{{.+}}>
    // CHECK-DAG: %[[RHS_TEMP:.+]] = ttg.convert_layout {{.+}} : tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf16, #blocked>
    // CHECK-DAG: %[[RHS:.+]] = ttg.local_alloc %[[RHS_TEMP]] : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #{{.+}}>
    // CHECK-DAG: %[[ACC:.+]] = ttg.convert_layout {{.+}} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #[[$MMA]]>
    // CHECK-DAG: %[[META:.+]] = ttg.convert_layout {{.+}} : tensor<64x4xi16, #blocked> -> tensor<64x4xi16, #triton_xla.sparse_dot_meta<{parent = #[[$MMA]]}>>
    // CHECK: %[[OUT:.+]] = triton_xla.sparse_dot %[[LHS]], %[[RHS]], %[[ACC]], %[[META]] : {{.+}} -> tensor<64x64xf32, #[[$MMA]]>
    %D = triton_xla.sparse_dot %A, %B, %C, %meta : tensor<64x32xf16, #lhs> meta tensor<64x4xi16, #blocked> * tensor<64x64xf16, #rhs> -> tensor<64x64xf32, #blocked>
    // CHECK: ttg.convert_layout %[[OUT]] : tensor<64x64xf32, #[[$MMA]]> -> tensor<64x64xf32, #blocked>
    tt.return %D : tensor<64x64xf32, #blocked>
  }
}