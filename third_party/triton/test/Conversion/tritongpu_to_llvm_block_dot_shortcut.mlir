// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// CHECK-LABEL: blocked_to_dot_op_shortcut_warp32
#blocked = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot_op_shortcut_warp32(%arg0: tensor<32x32xf16, #blocked>, %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>) {
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_warp64
#blocked = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [2, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot_op_shortcut_warp64(%arg0: tensor<32x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot3d_op_shortcut_warp32
#blocked = #ttg.blocked<{sizePerThread = [2, 32, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [1, 2, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot3d_op_shortcut_warp32(%arg0: tensor<8x32x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<8x32x32xf16, #blocked> -> tensor<8x32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot3d_op_shortcut_warp64
#blocked = #ttg.blocked<{sizePerThread = [1, 32, 1], threadsPerWarp = [1, 2, 32], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot3d_op_shortcut_warp64(%arg0: tensor<8x32x32xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<8x32x32xf16, #blocked> -> tensor<8x32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}
