// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

//       CHECK:   #[[$SHARED:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1]}
//       CHECK-LABEL: apply_swizzle
//       CHECK:   %{{.*}} = ttg.local_alloc %{{.*}} : (tensor<16x256xf16, #{{.*}}>) -> !ttg.memdesc<16x256xf16, #[[$SHARED]], #smem>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @apply_swizzle(%arg0: tensor<16x256xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<16x256xf16, #blocked> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

//       CHECK-LABEL:   conversion_shortcut_blocked_dotop_warp32
//       CHECK-NOT:  ttg.local_alloc
//       CHECK: ttg.convert_layout
//       CHECK-NOT:  ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @conversion_shortcut_blocked_dotop_warp32(%arg0: tensor<64x64xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

//       CHECK-LABEL:   conversion_shortcut_blocked_dotop_warp64
//       CHECK-NOT:  ttg.local_alloc
//       CHECK: ttg.convert_layout
//       CHECK-NOT:  ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @conversion_shortcut_blocked_dotop_warp64(%arg0: tensor<64x64xf16, #blocked>) {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_gfx1130
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1130", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot_op_shortcut_gfx1130(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttg.local_alloc
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot_op_shortcut_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.local_alloc
    // CHECK: ttg.convert_layout
    // CHECK-NOT: ttg.local_alloc
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: neg_blocked_to_dot_op_incompatible_elems_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @neg_blocked_to_dot_op_incompatible_elems_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: neg_blocked_to_dot_op_incompatible_threads_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @neg_blocked_to_dot_op_incompatible_threads_gfx940(%arg0: tensor<32x32xf16, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    tt.return
  }
}

// -----

// CHECK-LABEL: neg_blocked_to_dot_op_incompatible_warp_gfx940
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx940", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @neg_blocked_to_dot_op_incompatible_warp_gfx940(%arg0: tensor<128x128xf16, #blocked>) {
    // CHECK-NOT: ttg.convert_layout
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %0 = ttg.convert_layout %arg0 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    tt.return
  }
}
