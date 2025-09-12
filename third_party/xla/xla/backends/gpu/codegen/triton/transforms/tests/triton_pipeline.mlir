// RUN: xla-opt %s --triton-xla-pipeline='cuda=8.0' \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=CUDA
//
// RUN: xla-opt %s --triton-xla-pipeline='rocm=gfx950' \
// RUN:   | FileCheck %s --check-prefix=CHECK --check-prefix=ROCM

// CHECK: module attributes
// CUDA: ttg.target = "cuda:80"
// ROCM: ttg.target = "hip:gfx950"

// CHECK: llvm.func @func
tt.func @func() {
  // CHECK: llvm.return
  tt.return
}

