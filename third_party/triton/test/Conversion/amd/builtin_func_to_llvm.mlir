// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_expf(%arg0: tensor<64xf32, #blocked>) attributes {noinline = false} {
    // LLVM_FTZ: llvm.amdgcn.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_expf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return
  }
}
