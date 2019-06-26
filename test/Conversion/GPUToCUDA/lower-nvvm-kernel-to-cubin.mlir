// RUN: mlir-opt %s --test-kernel-to-cubin | FileCheck %s

func @kernel(%arg0 : !llvm.float, %arg1 : !llvm<"float*">)
// CHECK: attributes  {gpu.kernel, nvvm.cubin = "CUBIN"}
  attributes  { gpu.kernel } {
// CHECK-NOT: llvm.return
  llvm.return
}