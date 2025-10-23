// RUN: xla-opt %s --triton-xla-pipeline='target=9.0' \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=CUDA
//
// RUN: xla-opt %s --triton-xla-pipeline='target=gfx950' \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=ROCM

// CHECK: module attributes
// CUDA: ttg.target = "cuda:90"
// ROCM: ttg.target = "hip:gfx950"

// CHECK: llvm.func @func
tt.func @func() {
  // CHECK: llvm.return
  tt.return
}

