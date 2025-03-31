// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=70 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// CHECK-LABEL: clamp
module attributes {"ttg.target" = "cuda:70", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clamp(%x : tensor<1024xf32, #blocked>, %limit : tensor<1024xf32, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %neg_limit = arith.subf %cst, %limit : tensor<1024xf32, #blocked>

    // CHECK:      llvm.fcmp "une" %[[REG:[a-zA-Z0-9]+]], %[[REG]]
    // CHECK-NEXT: llvm.intr.maxnum
    // CHECK-NEXT: llvm.intr.minnum
    // CHECK-NEXT: llvm.mlir.constant
    // CHECK-NEXT: llvm.select
    %12 = tt.clampf %x, %neg_limit, %limit, propagateNan = all : tensor<1024xf32, #blocked>
    tt.return
  }
}
