// RUN: triton-opt --verify-diagnostics --dump-pass-pipeline --run-reproducer %s 2>&1 | FileCheck %s

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton__() attributes {noinline = false} {
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(any(convert-scf-to-cf,convert-index-to-llvm{index-bitwidth=0},convert-triton-gpu-to-llvm{compute-capability=90},convert-nv-gpu-to-llvm,convert-arith-to-llvm{index-bitwidth=0},canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse,symbol-dce,enable-line-info))",
      disable_threading: false,
      verify_each: false
    }
  }
#-}

// CHECK: Pass Manager with
// CHECK-NEXT: convert-triton-gpu-to-llvm
