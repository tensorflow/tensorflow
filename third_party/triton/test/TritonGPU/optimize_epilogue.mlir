// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-epilogue | FileCheck --check-prefixes=GCN %s

#mfma = #ttg.amd_mfma<{warpsPerCTA=[1,1], instrShape=[32,32], isTranspose=false}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GCN-LABEL: mfma_epilogue_simple
  // CHECK-LABEL: mfma_epilogue_simple
  tt.func public @mfma_epilogue_simple(%data: tensor<64x64xf16, #mfma>, %ptr: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    // GCN: [[PTR:%[a-z0-9]+]] = ttg.convert_layout {{.*}} : tensor<{{.*}}, #blocked> -> tensor<{{.*}}, #mma>
    // GCN: tt.store [[PTR]], {{.*}} : tensor<{{.*}}, #mma>
    %converted_data = ttg.convert_layout %data : tensor<64x64xf16, #mfma> -> tensor<64x64xf16, #blocked>
    tt.store %ptr, %converted_data : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{warpsPerCTA=[1,1], instrShape=[32,32], isTranspose=false}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GCN-LABEL: mfma_epilogue_chained_elementwise
  // CHECK-LABEL: mfma_epilogue_chained_elementwise
  tt.func public @mfma_epilogue_chained_elementwise(%data: tensor<64x64xf32, #mfma>, %ptr: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    // GCN: [[PTR:%[a-z0-9]+]] = ttg.convert_layout {{.*}} : tensor<{{.*}}, #blocked> -> tensor<{{.*}}, #mma>
    // GCN: tt.store [[PTR]], {{.*}} : tensor<{{.*}}, #mma>
    %converted_data = ttg.convert_layout %data : tensor<64x64xf32, #mfma> -> tensor<64x64xf32, #blocked>
    %trunked = arith.truncf %converted_data : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    tt.store %ptr, %trunked : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
