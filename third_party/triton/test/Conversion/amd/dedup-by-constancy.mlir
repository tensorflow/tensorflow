// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: dedup_by_constancy_mfma
// CHECK-COUNT-4: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// Here is why we expect exactly 4 icmp:
// For a 32x32 tensor A with mfma layout, each thread holds 16 elements, which are divided
// into 4 groups. E.g. thread 0 holds elements A[0:3,0], A[8:11,0], A[16:19,0], and A[24:27,0].
// In this example, constancy of the tensor is 16 for dim 0, meaning A[0:15,0] have same values
// and A[16:31,0] have same values. Therefore, for thread 0, the first 8 elements are duplicated
// and the last 8 elements are duplicated. Ideally, thread 0 only needs two icmp, one for the
// first 8 elements and the other for the last 8 elements. In practice, the dedup analysis
// only allows duplication within each group of 4 elemnets. Therefore, we expect 4 icmp, one
// for each group of 4 elements.
// In the future, we can reduce the icmp to 2 in such case.
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @dedup_by_constancy_mfma(%arg0: i32 {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %1 = tt.splat %arg0 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %2 = arith.cmpi slt, %0, %1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<32x1xi1, #mma>
    %4 = tt.broadcast %3 : tensor<32x1xi1, #mma> -> tensor<32x32xi1, #mma>
    %cst = arith.constant dense<0.100000e+00> : tensor<32x32xf16, #mma>
    %5 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #mma>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f16>, #mma> -> tensor<32x32x!tt.ptr<f16>, #mma>
    tt.store %6, %cst, %4 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}
