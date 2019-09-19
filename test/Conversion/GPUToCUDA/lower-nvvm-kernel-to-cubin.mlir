// RUN: mlir-opt %s --test-kernel-to-cubin -split-input-file | FileCheck %s

module attributes {gpu.kernel_module} {
  func @kernel(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {gpu.kernel, nvvm.cubin = "CUBIN"}
    attributes  { gpu.kernel } {
    // CHECK-NOT: llvm.return
    llvm.return
  }
}

// -----

module attributes {gpu.kernel_module} {
  // CHECK: func @kernel_a
  func @kernel_a()
    attributes  { gpu.kernel } {
    llvm.return
  }

  // CHECK: func @kernel_b
  func @kernel_b()
    attributes  { gpu.kernel } {
    llvm.return
  }
}
