// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' 2>&1 | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_high_precision_acc
  tt.func @dot_high_precision_acc(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #smem>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    %m = ttng.warp_group_dot %a, %b, %c
      {maxNumImpreciseAcc = 32 : i32, inputPrecision = 0 : i32} :
      !ttg.memdesc<128x128xf8E5M2, #shared, #smem> * !ttg.memdesc<128x256xf8E5M2, #shared1, #smem> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_low_precision_acc
  tt.func @dot_low_precision_acc(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #smem>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: llvm.return
    %m = ttng.warp_group_dot %a, %b, %c
      {maxNumImpreciseAcc = 129 : i32, inputPrecision = 0 : i32} :
      !ttg.memdesc<128x128xf8E5M2, #shared, #smem> * !ttg.memdesc<128x256xf8E5M2, #shared1, #smem> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_mix_precision_acc
  tt.func @dot_mix_precision_acc(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #smem>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: llvm.return
    %m = ttng.warp_group_dot %a, %b, %c
      {maxNumImpreciseAcc = 64 : i32, inputPrecision = 0 : i32} :
      !ttg.memdesc<128x128xf8E5M2, #shared, #smem> * !ttg.memdesc<128x256xf8E5M2, #shared1, #smem> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_zero_acc
  // Generate a wgmma with 2 sources.
  // CHECK: nvgpu.wgmma %{{.*}}, %{{.*}} {
  tt.func @dot_zero_acc(%a: !ttg.memdesc<128x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x64xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %m = ttng.warp_group_dot %a, %b, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
    tt.return
  }

  // CHECK-LABEL: @wgmma_on_subtile
  // CHECK: nvgpu.wgmma %{{.*}}, %{{.*}}
  tt.func @wgmma_on_subtile(%a: tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %b:  !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 3x64x256>){
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %m = ttng.warp_group_dot %a, %b, %cst {inputPrecision = 0 : i32, isAsync = true} : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 3x64x256> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_reg_operand_A
  // Generate a wgmma where the first operand is a struct.
  // CHECK: nvgpu.wgmma {{.*}} : (!llvm.struct<(i32, i32, i32, i32)>, i64, i1) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: nvgpu.wgmma_wait_group %{{.*}} {pendings = 0 : i32} : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  tt.func @dot_reg_operand_A(%a: tensor<128x64xf16, #mma>, %b: !ttg.memdesc<64x64xf16, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %opA = ttg.convert_layout %a : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %m = ttng.warp_group_dot %opA, %b, %cst { inputPrecision = 0 : i32 }:
      tensor<128x64xf16,  #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>
#mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_reg_operand_A_fp8
  // Generate a wgmma where the first operand is a struct.
  // CHECK: nvgpu.wgmma {{.*}} : (!llvm.struct<(i32, i32, i32, i32)>, i64, i1) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: nvgpu.wgmma_wait_group %{{.*}} {pendings = 0 : i32}
  tt.func @dot_reg_operand_A_fp8(%a: tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %b: !ttg.memdesc<128x256xf8E5M2, #shared, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma1>
    %m = ttng.warp_group_dot %a, %b, %cst { maxNumImpreciseAcc = 1073741824 : i32, inputPrecision = 0 : i32 } :
      tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * !ttg.memdesc<128x256xf8E5M2, #shared, #smem> -> tensor<128x256xf32, #mma1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: dot_reg_operand_upcast
  tt.func @dot_reg_operand_upcast(%a_desc: !ttg.memdesc<128x64xi8, #shared, #smem>, %b: !ttg.memdesc<64x64xf16, #shared, #smem>, %acc: tensor<128x64xf32, #mma>) {
    %a_dotop = ttg.local_load %a_desc : !ttg.memdesc<128x64xi8, #shared, #smem> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %a_casted = arith.sitofp %a_dotop : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> to tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %res = ttng.warp_group_dot %a_casted, %b, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_fp8_to_f16_conversion
  tt.func @test_fp8_to_f16_conversion(
    %in0: tensor<128xf8E5M2, #blocked>, %in1: tensor<128xf8E4M3FN, #blocked>,
    %in2: tensor<128xf16, #blocked>, %in3: tensor<128xf32, #blocked>) {
    // CHECK-COUNT-2: cvt.rn.f16x2.e5m2x2 {{.*}} "=r,h" %{{.*}} : (i16) -> vector<2xf16>
    %out0 = tt.fp_to_fp %in0 : tensor<128xf8E5M2, #blocked> -> tensor<128xf16, #blocked>
    // CHECK-COUNT-2: cvt.rn.f16x2.e4m3x2 {{.*}} "=r,h" %{{.*}} : (i16) -> vector<2xf16>
    %out1 = tt.fp_to_fp %in1 : tensor<128xf8E4M3FN, #blocked> -> tensor<128xf16, #blocked>
    // CHECK-COUNT-2: mul.rn.bf16x2
    %out2 = tt.fp_to_fp %in0 : tensor<128xf8E5M2, #blocked> -> tensor<128xbf16, #blocked>

    // CHECK-COUNT-2: cvt.rn.satfinite.e5m2x2.f16x2 {{.*}} "=h,r" %{{.*}} : (i32) -> vector<2xi8>
    %out3 = tt.fp_to_fp %in2, rounding = rtne : tensor<128xf16, #blocked> -> tensor<128xf8E5M2, #blocked>
    // CHECK-COUNT-2: cvt.rn.satfinite.e4m3x2.f16x2 {{.*}} "=h,r" %{{.*}} : (i32) -> vector<2xi8>
    %out4 = tt.fp_to_fp %in2, rounding = rtne : tensor<128xf16, #blocked> -> tensor<128xf8E4M3FN, #blocked>

    // CHECK-COUNT-2: cvt.rn.satfinite.e5m2x2.f32 {{.*}} "=h,r,r" %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi8>
    %out5 = tt.fp_to_fp %in3, rounding = rtne : tensor<128xf32, #blocked> -> tensor<128xf8E5M2, #blocked>
    // CHECK-COUNT-2: cvt.rn.satfinite.e4m3x2.f32 {{.*}} "=h,r,r" %{{.*}}, %{{.*}} : (i32, i32) -> vector<2xi8>
    %out6 = tt.fp_to_fp %in3, rounding = rtne : tensor<128xf32, #blocked> -> tensor<128xf8E4M3FN, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// CHECK-LABEL: clamp
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clamp(%x : tensor<1024xf32, #blocked>, %limit : tensor<1024xf32, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %neg_limit = arith.subf %cst, %limit : tensor<1024xf32, #blocked>

    // CHECK-COUNT-8: nvvm.fmin.xorsign.abs.f
    %12 = tt.clampf %x, %neg_limit, %limit, propagateNan = none : tensor<1024xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 16]}>
// CHECK-LABEL: convert_mma_to_blocked
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @convert_mma_to_blocked(%a: tensor<128x256xf16, #mma>) {
    // CHECK-COUNT-16: nvgpu.stmatrix
    //          CHECK: nvvm.barrier0
    %c = ttg.convert_layout %a : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: cvt_mma_to_dot_fp8
// CHECK: nvvm.prmt
// CHECK: nvvm.prmt
// CHECK: nvvm.shfl.sync
// CHECK: nvvm.shfl.sync
// CHECK: nvvm.prmt
// CHECK: nvvm.prmt
  tt.func @cvt_mma_to_dot_fp8(%a: tensor<128x64xf8E5M2, #mma>) {
    %opA = ttg.convert_layout %a : tensor<128x64xf8E5M2, #mma> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: dot_zero_acc_operand
// CHECK-COUNT-128: llvm.fadd
  tt.func @dot_zero_acc_operand(%a: !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, %b: !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %m = ttng.warp_group_dot %a, %b, %cst {maxNumImpreciseAcc = 64 : i32, inputPrecision = 0 : i32} :
      !ttg.memdesc<128x128xf8E5M2, #shared, #smem> * !ttg.memdesc<128x128xf8E5M2, #shared1, #smem> -> tensor<128x128xf32, #mma>
    tt.return
  }
}


// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: distribute_to_shared_st_matrix
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @distribute_to_shared_st_matrix(%a: tensor<128x128xf16, #mma>) {
    // CHECK-COUNT-16: nvgpu.stmatrix
    //          CHECK: llvm.return
    %b = ttg.local_alloc %a {allocation.offset = 0 : i32} : (tensor<128x128xf16, #mma>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: distribute_to_shared_st_matrix_local_store
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @distribute_to_shared_st_matrix_local_store(%a: tensor<128x128xf16, #mma>) {
    // CHECK-COUNT-16: nvgpu.stmatrix
    //          CHECK: llvm.return
    %b = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.local_store %a, %b : tensor<128x128xf16, #mma> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fp8_const(%arg0: tensor<1024xi1, #blocked>, %arg1: tensor<1024xf8E4M3FNUZ, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: @fp8_const
    // CHECK: llvm.mlir.constant(0.000000e+00 : f8E4M3FNUZ) : i8
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf8E4M3FNUZ, #blocked>
    %a = arith.select %arg0, %arg1, %cst : tensor<1024xi1, #blocked>, tensor<1024xf8E4M3FNUZ, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f32_nomask(%dest_ptrs: tensor<256x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf32, #blocked>) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f32_nomask
    // CHECK: atom.global.gpu.acq_rel.add.v4.f32
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f32_withmask(%dest_ptrs: tensor<256x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf32, #blocked>, %mask: tensor<256xi1, #blocked> {tt.constancy = 2 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f32_withmask
    // CHECK: atom.global.gpu.acq_rel.add.v2.f32
    // CHECK: atom.global.gpu.acq_rel.add.v2.f32
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xf32, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @atomic_add_f16_withmask(%dest_ptrs: tensor<256x!tt.ptr<f16>, #blocked> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32}, %data: tensor<256xf16, #blocked>, %mask: tensor<256xi1, #blocked> {tt.constancy = 4 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: atomic_add_f16_withmask
    // CHECK: atom.global.gpu.acq_rel.add.noftz.v4.f16
    // CHECK: atom.global.gpu.acq_rel.add.noftz.v4.f16
    %0 = tt.atomic_rmw fadd, acq_rel, gpu, %dest_ptrs, %data, %mask : (tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xf16, #blocked>, tensor<256xi1, #blocked>) -> tensor<256xf16, #blocked>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_fp8_to_fp16_dot_operand
  // CHECK-COUNT-16: cvt.rn.f16x2.e5m2x2
  tt.func @test_fp8_to_fp16_dot_operand(%arg: tensor<128x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
    %r = tt.fp_to_fp %arg : tensor<128x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    tt.return
  }
}
