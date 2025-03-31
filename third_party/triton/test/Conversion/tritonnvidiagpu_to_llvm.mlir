// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32) {
    // CHECK: waitLoop:
    // CHECK: mbarrier.try_wait.parity.shared.b64
    // CHECK: @!P1 bra.uni waitLoop
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK-NEXT: [[TID:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK-NEXT: [[IS_ZERO:%.*]] = llvm.icmp "eq" [[TID]], [[C0]]
    // CHECK-NEXT: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 2;", "b,r" [[IS_ZERO]], %arg0
    ttng.arrive_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: arrive_barrier_pred
  tt.func @arrive_barrier_pred(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    // CHECK-NEXT: [[TID:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK-NEXT: [[C0:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK-NEXT: [[IS_ZERO:%.*]] = llvm.icmp "eq" [[TID]], [[C0]]
    // CHECK-NEXT: [[PRED:%.*]] = llvm.and [[IS_ZERO]], %arg1
    // CHECK-NEXT: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 2;", "b,r" [[PRED]], %arg0
    ttng.arrive_barrier %alloc, 2, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}


// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tma_copy_global_to_local
  // CHECK: elect.sync
  // CHECK: "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" {{.*}} : (i1, !llvm.ptr<3>, !llvm.ptr<1>, i32, i32, !llvm.ptr<3>) -> !llvm.void
  // CHECK-NOT: cp.async.bulk.tensor.2d.shared
  // CHECK: return
  tt.func @tma_copy_global_to_local(%tma: !tt.ptr<i64>, %alloc: !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>, %x: i32, %barrier: !ttg.memdesc<1xi64, #shared0, #smem>, %pred: i1) {
    ttng.async_tma_copy_global_to_local %tma[%x, %x] %alloc, %barrier, %pred : !tt.ptr<i64>, !ttg.memdesc<1xi64, #shared0, #smem> -> !ttg.memdesc<128x128xf32, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tma_copy_local_to_global
  // CHECK: elect.sync
  // CHECK: "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" {{.*}} : (i1, !llvm.ptr<1>, i32, i32, !llvm.ptr<3>) -> !llvm.void
  // CHECK-NOT: cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
  // CHECK: nvvm.cp.async.bulk.commit.group
  tt.func @tma_copy_local_to_global(%tma: !tt.ptr<i64>, %alloc: !ttg.memdesc<128x128xf32, #shared1, #smem>, %x: i32) {
    ttng.async_tma_copy_local_to_global %tma[%x, %x] %alloc : !tt.ptr<i64>, !ttg.memdesc<128x128xf32, #shared1, #smem>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: async_tma_store_wait
  // CHECK: nvvm.cp.async.bulk.wait_group 0 {read}
  tt.func @async_tma_store_wait() {
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: expect_barrier
  // CHECK: @$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 16384;
  tt.func @expect_barrier(%barrier: !ttg.memdesc<1xi64, #shared0, #smem, mutable>, %pred: i1) {
    ttng.barrier_expect %barrier, 16384, %pred : !ttg.memdesc<1xi64, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: byval_tma_desc
  // CHECK: llvm.align = 64
  // CHECK: llvm.byval = !llvm.array<128 x i8>
  // CHECK: nvvm.grid_constant
  tt.func @byval_tma_desc(%desc: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}) {
    tt.return
  }
}

// -----

// CHECK-LABEL: device_tensormap_create1d
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @device_tensormap_create1d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: st.shared.b32
    // CHECK: bar.warp.sync
    // CHECK: tensormap.replace.tile.global_address.shared::cta.b1024.b64 [ $0 + 0 ], $1;
    // CHECK: tensormap.replace.tile.rank.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;
    // CHECK: tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [ $0 + 0 ], 0x3;
    // CHECK: tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;
    // CHECK: tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x2;
    // CHECK: tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x1;
    // CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [ $0 + 0 ], [ $1 + 0 ], 0x80;
    tt.experimental_tensormap_create %arg1, %arg0, [%c256_i32], [%arg2], [], [%c1_i32] {elem_type = 3 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32, allocation.offset = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: device_tensormap_create2d
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @device_tensormap_create2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i64 = arith.constant 1024 : i64
    // CHECK: st.shared.b32
    // CHECK: bar.warp.sync
    // CHECK: tensormap.replace.tile.global_address.shared::cta.b1024.b64 [ $0 + 0 ], $1;
    // CHECK: tensormap.replace.tile.rank.shared::cta.b1024.b32 [ $0 + 0 ], 0x1;
    // CHECK: tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;
    // CHECK: tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;
    // CHECK: tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;
    // CHECK: tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;
    // CHECK: tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [ $0 + 0 ], 0x3;
    // CHECK: tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;
    // CHECK: tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x2;
    // CHECK: tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x1;
    // CHECK: tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [ $0 + 0 ], [ $1 + 0 ], 0x80;
    tt.experimental_tensormap_create %arg1, %arg0, [%c256_i32, %c256_i32], [%arg2, %arg2], [%c1024_i64], [%c1_i32, %c1_i32] {elem_type = 3 : i32, fill_mode = 1 : i32, interleave_layout = 0 : i32, swizzle_mode = 2 : i32, allocation.offset = 0 : i32} : (!tt.ptr<i8>, !tt.ptr<i16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    tt.return
  }
}

// -----

// CHECK-LABEL: tensormap_fenceproxy_acquire
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tensormap_fenceproxy_acquire(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK: fence.proxy.tensormap::generic.acquire.gpu [ $0 + 0 ], 0x80;
    tt.experimental_tensormap_fenceproxy_acquire %arg0 : !tt.ptr<i8>
    tt.return
  }
}
