// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f32_nomask(%dest_ptrs: tensor<256x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f32_nomask
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f32_withmask(%dest_ptrs: tensor<256x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf32, #blocked>, %mask: tensor<256xi1, #blocked> {tt.constancy = 2 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f32_withmask
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    // CHECK: atom.global.gpu.acq_rel.add.f32
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f16_withmask(%dest_ptrs: tensor<256x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf16, #blocked>, %mask: tensor<256xi1, #blocked> {tt.constancy = 4 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f16_withmask
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    // CHECK: atom.global.gpu.acq_rel.add.noftz.f16x2
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
    tt.return
  }
}
