// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-tensor-memory-allocation | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 256, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: ttg.tensor_memory_size = 512
  // CHECK: alloc_tensor_memory
  tt.func public @alloc_tensor_memory(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %cst3 = arith.constant dense<0> : tensor<64x4xi8, #linear>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc %cst0 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc %cst1 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 320 : i32, tensor_memory_row_offset = 0 : i32}
    %3 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst, %0, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst0, %1, %true : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst1, %2, %true : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %3, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %4 = ttng.tmem_alloc %cst2 : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 16 : i32}
    %5 = ttng.tmem_alloc %cst2 : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %6 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst2, %4, %true : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %5, %true : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x128xf16, #tmem2, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %6, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %7 = ttng.tmem_alloc : () -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %8 = ttng.tmem_alloc : () -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc  {tensor_memory_col_offset = 4 : i32, tensor_memory_row_offset = 0 : i32}

    ttng.tmem_store %cst3, %7, %true : tensor<64x4xi8, #linear> -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst3, %8, %true : tensor<64x4xi8, #linear> -> !ttg.memdesc<64x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>


    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: ttg.tensor_memory_size = 512
  // CHECK: alloc_tensor_memory
  tt.func public @alloc_tensor_memory_re_use(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %a = ttng.tmem_alloc %cst0 : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %0 = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %1, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %2, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    // Test that the 2 allocations above are re-used.
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %3 = ttng.tmem_alloc %cst0 : (tensor<128x256xf32, #blocked>) -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %4 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32}
    %5 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %4, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %6 = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %s = ttg.memdesc_subview %6[%c1, %c0, %c0] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %7 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc %{{.+}} {tensor_memory_col_offset = 384 : i32, tensor_memory_row_offset = 0 : i32}
    %8 = ttng.tmem_alloc %cst2 : (tensor<128x64xf32, #blocked>) -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst, %s, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %7, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst2, %5, %true : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true, CTASplitM = 2>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true, CTASplitN = 2>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, ttg.shared = 65536 : i32} {
  // CHECK-LABEL: multi_ctas
  tt.func public @multi_ctas() {
    %true = arith.constant true
    %cst0 = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blocked>

    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
    %0 = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>

    ttng.tmem_store %cst0, %0, %true : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst0, %1, %true : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst0, %2, %true : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#layout = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem = #ttng.tensor_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @alloc_warp_specialize
tt.func @alloc_warp_specialize() {
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
  %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
  ttg.warp_specialize()
  default {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 384 : i32, tensor_memory_row_offset = 0 : i32}
    %2 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    "use"(%1) : (!ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) -> ()
    ttg.warp_return
  } : () -> ()
  "use"(%0) : (!ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @alloc_warp_specialize_explicit_capture
tt.func @alloc_warp_specialize_explicit_capture() {
  // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32}
  %0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
  ttg.warp_specialize(%0)
  default {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) num_warps(1) {
    // CHECK: ttng.tmem_alloc {tensor_memory_col_offset = 256 : i32, tensor_memory_row_offset = 0 : i32}
    %1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #layout, #tmem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<128x128xf32, #layout, #tmem, mutable>) -> ()
  tt.return
}

}
