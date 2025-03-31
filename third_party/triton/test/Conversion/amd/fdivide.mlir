// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f32(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f32
    // CHECK: llvm.amdgcn.div.scale.f32
    // CHECK: llvm.amdgcn.div.scale.f32
    // CHECK: llvm.amdgcn.rcp.f32
    // CHECK: llvm.fmul
    // CHECK: llvm.amdgcn.div.fmas.f32
    // CHECK: llvm.amdgcn.div.fixup.f32
    // CHECK-NOT: llvm.fdiv
    %0 = arith.divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fdiv_f64(%arg0: tensor<64xf64, #blocked>, %arg1: tensor<64xf64, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_fdiv_f64
    // CHECK: llvm.fdiv
    %0 = arith.divf %arg0, %arg1 : tensor<64xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_div_rn(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: test_div_rn
    // CHECK: llvm.fdiv
    %0 = tt.precise_divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}
