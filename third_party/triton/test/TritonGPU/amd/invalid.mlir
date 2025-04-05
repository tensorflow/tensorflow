// RUN: triton-opt --split-input-file %s --verify-diagnostics

// expected-error @+1 {{Transposed WMMA is supported only for version 2}}
#wmma = #ttg.amd_wmma<{version = 1, isTranspose = true, warpsPerCTA = [2, 2]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}

// -----

// expected-error @+1 {{WMMA version must be in the [1, 2] range}}
#wmma = #ttg.amd_wmma<{version = 0, isTranspose = false, warpsPerCTA = [2, 2]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_output_encoding(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error-re @+15 {{Expect output layout to be transposed per thread:{{.*}}- register=1 -> (1, 0){{.*}}register=2 -> (2, 0){{.*}}register=4 -> (0, 1){{.*}}register=8 -> (0, 2)}}
// Full expected layout is following:
// - register=1 -> (1, 0)
//   register=2 -> (2, 0)
//   register=4 -> (0, 1)
//   register=8 -> (0, 2)}}
// - lane=1 -> (0, 4)
//   lane=2 -> (0, 8)
//   lane=4 -> (0, 16)
//   lane=8 -> (4, 0)
//   lane=16 -> (8, 0)
//   lane=32 -> (16, 0)
// - warp is a size 1 dimension
// - block is a size 1 dimension
// where out dims are: [dim0 (size 32), dim1 (size 32)]
    %0 = amdgpu.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_input_encoding(%arg0: tensor<32x32xf16, #mfma>) {
// expected-error @+1 {{Expect input tensor in Blocked encoding}}
    %0 = amdgpu.in_thread_transpose %arg0 : tensor<32x32xf16, #mfma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_shape(%arg0: tensor<64x64xf16, #blocked>) {
// expected-error @+1 {{Expect equal input and output shapes}}
    %0 = amdgpu.in_thread_transpose %arg0 : tensor<64x64xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_dtype(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error @+1 {{Expect input and output tensor to have same dtype}}
    %0 = amdgpu.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf32, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 4], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_3d_shape(%arg0: tensor<2x32x32xf16, #blocked>) {
// expected-error @+1 {{Expect 2d tensor}}
    %0 = amdgpu.in_thread_transpose %arg0 : tensor<2x32x32xf16, #blocked> -> tensor<2x32x32xf16, #linear>
    tt.return
  }
}
