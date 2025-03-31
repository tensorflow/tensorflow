// RUN: triton-opt %s --convert-nv-gpu-to-llvm -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: @nvvm_syncs
llvm.func @nvvm_syncs() {
  // CHECK: wgmma.fence.sync.aligned;
  nvgpu.wgmma_fence

  // CHECK: wgmma.commit_group.sync.aligned;
  nvgpu.wgmma_commit_group

  // CHECK: barrier.cluster.wait.aligned;
  nvgpu.cluster_wait

  // CHECK: fence.proxy.async.shared::cta;
  nvgpu.fence_async_shared {bCluster = false}
  // CHECK: fence.proxy.async.shared::cluster;
  nvgpu.fence_async_shared {bCluster = true}

  // CHECK: barrier.cluster.arrive.aligned;
  nvgpu.cluster_arrive {relaxed = false}
  // CHECK: barrier.cluster.arrive.relaxed.aligned;
  nvgpu.cluster_arrive {relaxed = true}

  llvm.return
}

// CHECK-LABEL: @cluster_id
llvm.func @cluster_id() -> i32 {
  // CHECK:      %cluster_ctaid.x;
  // CHECK-SAME: %cluster_ctaid.y;
  // CHECK-SAME: %cluster_ctaid.z;
  // CHECK-SAME: %cluster_nctaid.x;
  // CHECK-SAME: %cluster_nctaid.y;
  %id = nvgpu.cluster_id
  llvm.return %id : i32
}

// -----

// CHECK-LABEL: @stmatrix
llvm.func @stmatrix(%i: i32, %ptr: !llvm.ptr<3>) {
  // CHECK: stmatrix.sync.aligned.m8n8.x4.shared.b16 [$0], {$1, $2, $3, $4};
  nvgpu.stmatrix %ptr, %i, %i, %i, %i : !llvm.ptr<3>, i32, i32, i32, i32
  // CHECK: stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};
  nvgpu.stmatrix %ptr, %i, %i, %i, %i {trans} : !llvm.ptr<3>, i32, i32, i32, i32
  llvm.return
}

// -----

// CHECK-LABEL: @ldmatrix
llvm.func @ldmatrix(%ptr: !llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];
  %0 = nvgpu.ldmatrix %ptr : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
  // CHECK: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];
  %1 = nvgpu.ldmatrix %ptr {trans} : (!llvm.ptr<3>) -> !llvm.struct<(i32, i32, i32, i32)>
  %2 = llvm.extractvalue %1[0] : !llvm.struct<(i32, i32, i32, i32)>
  %3 = llvm.insertvalue %2, %0[0] : !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %3 : !llvm.struct<(i32, i32, i32, i32)>
}

// -----

!struct_128xf32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
)>

!struct_64xf32 = !llvm.struct<(
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
)>

// CHECK-LABEL: @wgmma
llvm.func @wgmma(%desc: i64, %in: !struct_64xf32) {
// CHECK: wgmma.mma_async.sync.aligned.m64n256k32.f32.e5m2.e5m2
%false = llvm.mlir.constant(false) : i1
%acc0 = nvgpu.wgmma %desc, %desc, %false {
  eltTypeA = 3 : i32,
  eltTypeB = 3 : i32,
  eltTypeC = 7 : i32,
  layoutA = 0 : i32,
  layoutB = 1 : i32,
  m = 64 : i32,
  n = 256 : i32,
  k = 32 : i32
} : (i64, i64, i1) -> !struct_128xf32

  // CHECK: // wait for regs: $0,$1,$2,{{.*}},$127
  // CHECK: wgmma.wait_group.sync.aligned 0;
  %out = nvgpu.wgmma_wait_group %in {pendings = 0 : i32} : !struct_64xf32
  llvm.return
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tensor_memory_base_lowering
  //      CHECK:    %[[TID:.+]] = nvvm.read.ptx.sreg.tid.x : i32
  //      CHECK:    %[[C32:.+]] = llvm.mlir.constant(32 : i32) : i32
  //      CHECK:    %[[PRED:.+]] = llvm.icmp "ult" %[[TID]], %[[C32]] : i32
  //      CHECK:    %[[SHMEM:.+]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  //      CHECK:    %[[A:.+]] = llvm.inline_asm has_side_effects
  // CHECK-SAME:    "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128;", "b,r" %[[PRED]], %[[SHMEM]] : (i1, !llvm.ptr<3>) -> !llvm.void
  //      CHECK:    %[[AR:.+]] = llvm.load %[[SHMEM]] : !llvm.ptr<3> -> i32
  //      CHECK:    nvvm.barrier0
  //      CHECK:    "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %[[PRED]]  : (i1) -> !llvm.void
  //      CHECK:    llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 128;", "b,r" %[[PRED]], %{{.+}} : (i1, !llvm.ptr<6>) -> !llvm.void
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @tensor_memory_base_lowering() -> i32 attributes {nvvm.kernel = 1 : ui1, nvvm.maxntid = array<i32: 128>} {
    %263 = nvgpu.tensor_memory_base
    %264 = llvm.ptrtoint %263 : !llvm.ptr<6> to i32
    llvm.return %264 : i32
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @warpid_warp_specialize
llvm.func @warpid_warp_specialize() {
  // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
  // CHECK: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
  // CHECK: [[ID:%.*]] = llvm.udiv [[TIDX]], [[C32]]
  // CHECK: [[UNIFORM:%.*]] = nvvm.shfl.sync idx {{%[0-9]+}}, [[ID]]
  %0 = nvgpu.warp_id
  // CHECK: "use"([[UNIFORM]])
  "use"(%0) : (i32) -> ()

  // CHECK: ttg.warp_specialize
  ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 6, 4>}
  // CHECK: default
  default {
    // CHECK: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK: [[ID:%.*]] = llvm.udiv [[TIDX]], [[C32]]
    // CHECK: [[UNIFORM:%.*]] = nvvm.shfl.sync idx {{%[0-9]+}}, [[ID]]
    %1 = nvgpu.warp_id
    // CHECK: "use"([[UNIFORM]])
    "use"(%1) : (i32) -> ()
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0() num_warps(4) {
    // 6*32 = 196

    // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
    // CHECK: [[C192:%.*]] = llvm.mlir.constant(192 : i32)
    // CHECK: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK: [[REL_TIDX:%.*]] = llvm.sub [[TIDX]], [[C192]]
    // CHECK: [[ID:%.*]] = llvm.udiv [[REL_TIDX]], [[C32]]
    // CHECK: [[UNIFORM:%.*]] = nvvm.shfl.sync idx {{%[0-9]+}}, [[ID]]
    %1 = nvgpu.warp_id
    // CHECK: "use"([[UNIFORM]])
    "use"(%1) : (i32) -> ()
    ttg.warp_return
  }
  partition1() num_warps(2) {
    // 4*32 = 128

    // CHECK: [[C32:%.*]] = llvm.mlir.constant(32 : i32)
    // CHECK: [[C128:%.*]] = llvm.mlir.constant(128 : i32)
    // CHECK: [[TIDX:%.*]] = nvvm.read.ptx.sreg.tid.x
    // CHECK: [[REL_TIDX:%.*]] = llvm.sub [[TIDX]], [[C128]]
    // CHECK: [[ID:%.*]] = llvm.udiv [[REL_TIDX]], [[C32]]
    // CHECK: [[UNIFORM:%.*]] = nvvm.shfl.sync idx {{%[0-9]+}}, [[ID]]
    %1 = nvgpu.warp_id
    // CHECK: "use"([[UNIFORM]])
    "use"(%1) : (i32) -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
