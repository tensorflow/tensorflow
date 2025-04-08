// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM90 --dump-input-context=20 %s
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM80 --dump-input-context=20 %s


#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @add_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: add_bf16
    // SM80-COUNT-4: fma.rn.bf16x2
    // SM90-COUNT-4: add.rn.bf16x2
    %0 = arith.addf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @sub_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: sub_bf16
    // SM80-COUNT-4: fma.rn.bf16x2
    // SM90-COUNT-4: sub.rn.bf16x2
    %0 = arith.subf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @mul_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: mul_bf16
    // SM80-COUNT-4: fma.rn.bf16x2
    // SM90-COUNT-4: mul.rn.bf16x2
    %0 = arith.mulf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @extf_bf16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: extf_bf16
    // CHECK-COUNT-8: cvt.f32.bf16
    %0 = arith.extf %arg0 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @truncf_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: truncf_bf16
    // CHECK-COUNT-4: cvt.rn.bf16x2.f32
    %0 = arith.truncf %arg0 : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @extf_f16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf16, #blocked>) {
    // CHECK-LABEL: extf_f16
    // CHECK-COUNT-8: cvt.f32.f16
    %0 = arith.extf %arg0 : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @truncf_f16(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: truncf_f16
    // CHECK-COUNT-4: cvt.rn.f16x2.f32
    %0 = arith.truncf %arg0 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
