// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering | FileCheck %s
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_load
// CHECK: ttg.local_alloc : ()
// CHECK: ttg.local_alloc : ()
// CHECK: ttng.init_barrier
// CHECK: ttng.tensor_desc_to_tma_ptr
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.wait_barrier
// CHECK: ttng.inval_barrier
// CHECK: ttg.local_load
  tt.func public @tma_load(%arg0: !tt.tensordesc<tensor<128x64xf16, #nvmma_128>>, %arg1: i32) -> tensor<128x64xf16, #blocked> {
    %l = tt.descriptor_load %arg0[%arg1, %arg1] : !tt.tensordesc<tensor<128x64xf16, #nvmma_128>> -> tensor<128x64xf16, #blocked>
    tt.return %l : tensor<128x64xf16, #blocked>
  }
}

// -----
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: ttg.local_alloc
//       CHECK: ttng.fence_async_shared {bCluster = false}
//       CHECK: ttng.tensor_desc_to_tma_ptr
//       CHECK: ttng.async_tma_copy_local_to_global
  tt.func public @tma_store(%arg0: !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<tensor<128x256xf32, #nvmma_128>>, tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----
#nvmma_32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: make_tensor_descriptor
  // CHECK: %0 = arith.extsi %arg2 : i32 to i64
  // CHECK: %1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
  // CHECK: tt.experimental_tensormap_create %1, %arg0, [%c32_i32, %c8_i32], [%arg2, %arg1], [%0], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 1 : i32} : (!tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32, i32, i64, i32, i32) -> ()
  // CHECK: tt.experimental_tensormap_fenceproxy_acquire %1 : !tt.ptr<i8>
  // CHECK: tt.reinterpret_tensor_descriptor %1 : !tt.ptr<i8> to !tt.tensordesc<tensor<8x32xi8, #shared>>
  tt.func public @make_tensor_descriptor(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32} ) -> !tt.tensordesc<tensor<8x32xi8, #nvmma_32>> {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<32> : tensor<8x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<tensor<8x32xi8, #nvmma_32>>
    tt.return %1 : !tt.tensordesc<tensor<8x32xi8, #nvmma_32>>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @tma_gather
tt.func @tma_gather(%arg0: !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, %arg1: tensor<32xi32, #blocked>, %arg2: i32) -> tensor<32x128xbf16, #blocked1> {
  // CHECK: [[RESULT:%.*]] = ttg.local_alloc
  // CHECK: [[BARRIER:%.*]] = ttg.local_alloc
  // CHECK: ttng.init_barrier [[BARRIER]]
  // CHECK: [[DESC_PTR:%.*]] = ttng.tensor_desc_to_tma_ptr %arg0
  // CHECK: ttng.async_tma_gather [[DESC_PTR]][%arg1, %arg2] [[RESULT]], [[BARRIER]], %true
  // CHECK: ttng.wait_barrier [[BARRIER]]
  // CHECK: ttng.inval_barrier [[BARRIER]]
  // CHECK: [[OUT:%.*]] = ttg.local_load [[RESULT]]
  %0 = tt.descriptor_gather %arg0[%arg1, %arg2] : (!tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, tensor<32xi32, #blocked>, i32) -> tensor<32x128xbf16, #blocked1>
  // CHECK: return [[OUT]]
  tt.return %0 : tensor<32x128xbf16, #blocked1>
}

// CHECK-LABEL: @tma_scatter
tt.func @tma_scatter(%arg0: !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, %arg1: tensor<32xi32, #blocked>, %arg2: i32, %arg3: tensor<32x128xbf16, #blocked1>) {
  // CHECK-NEXT: [[SRC:%.*]] = ttg.local_alloc %arg3
  // CHECK-NEXT: ttng.fence_async_shared {bCluster = false}
  // CHECK-NEXT: [[PTR:%.*]] = ttng.tensor_desc_to_tma_ptr %arg0
  // CHECK-NEXT: ttng.async_tma_scatter [[PTR]][%arg1, %arg2] [[SRC]]
  // CHECK-NEXT: ttng.async_tma_store_wait
  tt.descriptor_scatter %arg0[%arg1, %arg2], %arg3 : !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, tensor<32xi32, #blocked>, i32, tensor<32x128xbf16, #blocked1>
  tt.return
}

}
