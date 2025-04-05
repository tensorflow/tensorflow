// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm='arch=gfx942' | FileCheck %s

#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @basic_insert_slice(%arg0: tensor<256x128xi32, #blocked1> {tt.divisibility = 16 : i32}) {
    // CHECK: llvm.func @basic_insert_slice
    // CHECK-COUNT-64: %{{[0-9]*}} = llvm.extractvalue  %arg0[{{[0-9]*}}] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK: %64 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-COUNT-8:  %{{[0-9]*}} = llvm.insertvalue %{{[0-9]*}}, %{{[0-9]*}}[{{[0-9]*}}] : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    %72 = amdgpu.extract_slice %arg0 [0,0] : tensor<256x128xi32, #blocked1> to tensor<256x16xi32, #blocked1>
    tt.return
  }
}
