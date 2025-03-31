// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering --allocate-shared-memory -test-print-membar | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
  tt.func @init_barrier() {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: inval_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: inval_barrier
  tt.func @inval_barrier() {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
		ttng.inval_barrier %alloc : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: barrier_expect
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: barrier_expect
  tt.func @barrier_expect(%pred : i1) {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.barrier_expect %alloc, 16384, %pred : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
	// CHECK: local_alloc
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: init_barrier
	// CHECK-NEXT: gpu.barrier
	// CHECK-NEXT: wait_barrier
  tt.func @wait_barrier(%phase : i32) {
  	%cst = arith.constant dense<0> : tensor<1xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<1xi64, #blocked0>) -> !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----


#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_load(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared0>>, %arg1: i32) -> tensor<128x64xf16, #blocked0> {
		// CHECK-LABEL: tma_load
		// CHECK: local_dealloc
		// CHECK-NEXT: local_alloc
		// CHECK-NEXT: local_alloc
		// CHECK-NEXT: gpu.barrier
		// CHECK-NEXT: init_barrier
  	%cst = arith.constant dense<0> : tensor<128x64xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<128x64xi64, #blocked0>) -> !ttg.memdesc<128x64xi64, #shared0, #smem, mutable>
  	ttg.local_dealloc %alloc : !ttg.memdesc<128x64xi64, #shared0, #smem, mutable>
    %l = tt.descriptor_load %arg0[%arg1, %arg1] : !tt.tensordesc<tensor<128x64xf16, #shared0>> -> tensor<128x64xf16, #blocked0>
    tt.return %l : tensor<128x64xf16, #blocked0>
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: ttg.local_alloc
//       CHECK-NEXT: ttg.local_dealloc
//       CHECK-NEXT: gpu.barrier
//       CHECK-NEXT: ttg.local_alloc
  tt.func public @tma_store(%arg0: !tt.tensordesc<tensor<128x256xf32, #shared0>>, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked0>) {
  	%cst = arith.constant dense<0> : tensor<128x64xi64, #blocked0>
  	%alloc = ttg.local_alloc %cst : (tensor<128x64xi64, #blocked0>) -> !ttg.memdesc<128x64xi64, #shared0, #smem, mutable>
  	ttg.local_dealloc %alloc : !ttg.memdesc<128x64xi64, #shared0, #smem, mutable>
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<tensor<128x256xf32, #shared0>>, tensor<128x256xf32, #blocked0>
    tt.return
  }
}
