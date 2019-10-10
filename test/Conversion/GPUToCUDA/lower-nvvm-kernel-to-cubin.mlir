// RUN: mlir-opt %s --test-kernel-to-cubin -split-input-file | FileCheck %s

// CHECK: attributes {gpu.kernel_module, nvvm.cubin = "CUBIN"}
module @foo attributes {gpu.kernel_module} {
  llvm.func @kernel(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
    // CHECK: attributes  {gpu.kernel}
    attributes  { gpu.kernel } {
    llvm.return
  }
}

// -----

module @bar attributes {gpu.kernel_module} {
  // CHECK: func @kernel_a
  llvm.func @kernel_a()
    attributes  { gpu.kernel } {
    llvm.return
  }

  // CHECK: func @kernel_b
  llvm.func @kernel_b()
    attributes  { gpu.kernel } {
    llvm.return
  }
}
