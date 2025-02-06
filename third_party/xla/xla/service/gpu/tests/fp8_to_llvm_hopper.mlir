// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=90 \
// RUN: | FileCheck %s

// Check that Triton uses incorrect type to map to NVIDIA .e4m3 type.
// When this test fails, change the mapping in ir_emitter_triton.cc.
// See b/345700241.
#mma = #ttg.nvidia_mma<{
  versionMajor = 2,
  versionMinor = 0,
  warpsPerCTA = [1, 1],
  instrShape = [16, 8]
}>

module attributes {
  "ttg.compute-capability" = 90 : i32,
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32
} {

// CHECK-LABEL: e4m3_mapping
tt.func @e4m3_mapping(
    %arg0: tensor<16x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>,
    %arg1: tensor<256x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  ) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
  // CHECK: mma.{{.*}}.e4m3.e4m3.f32
  %res = tt.dot %arg0, %arg1, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32}
      : tensor<16x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> *
        tensor<256x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        -> tensor<16x16xf32, #mma>
  tt.return
}

}
