// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=True" | FileCheck %s --check-prefixes=COMMON,LLVM_FTZ
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=False" | FileCheck %s --check-prefixes=COMMON,LLVM_NO_FTZ


#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp2(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp2 %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_rsqrt(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.rsq.f32
    // LLVM_NO_FTZ: _ocml_rsqrt_f32
    %0 = math.rsqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_f32(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ-LABEL: test_sqrt_f32
    // LLVM_FTZ-NOT: llvm.fcmp "ogt"
    // LLVM_FTZ: llvm.amdgcn.sqrt.f32
    // LLVM_FTZ-NOT: llvm.fmul
    // LLVM_FTZ-NOT: llvm.select
    //
    // LLVM_NO_FTZ-LABEL: test_sqrt_f32
    // LLVM_NO_FTZ: llvm.fcmp "ogt"
    // LLVM_NO_FTZ: llvm.fmul
    // LLVM_NO_FTZ-NEXT: llvm.select
    // LLVM_NO_FTZ-NEXT: llvm.amdgcn.sqrt.f32
    // LLVM_NO_FTZ: llvm.fmul
    // LLVM_NO_FTZ-NEXT: llvm.select
    %0 = math.sqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_rn_f32(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ-LABEL: test_sqrt_rn_f32
    // LLVM_FTZ: llvm.amdgcn.rsq.f32
    // LLVM_FTZ: llvm.fmul
    // LLVM_FTZ: llvm.fmul
    // LLVM_FTZ: llvm.fneg
    // LLVM_FTZ: llvm.intr.fma
    // LLVM_FTZ-NEXT: llvm.intr.fma
    // LLVM_FTZ-NEXT: llvm.intr.fma
    // LLVM_FTZ-NEXT: llvm.fneg
    // LLVM_FTZ-NEXT: llvm.intr.fma
    // LLVM_FTZ-NEXT: llvm.intr.fma
    // LLVM_FTZ-NEXT: llvm.intr.is.fpclass
    // LLVM_FTZ-NEXT: llvm.select
    //
    // LLVM_NO_FTZ-LABEL: test_sqrt_rn_f32
    // LLVM_NO_FTZ: llvm.intr.sqrt
    %0 = tt.precise_sqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_rn_f64(%arg0: tensor<64xf64, #blocked>) attributes {noinline = false} {
    // COMMON-LABEL: test_sqrt_rn_f64
    // COMMON: llvm.intr.sqrt
    %0 = tt.precise_sqrt %arg0 : tensor<64xf64, #blocked>
    tt.return
  }
}
