// RUN: mlir-opt %s --generate-cubin-accessors | FileCheck %s

module attributes {gpu.container_module} {

// CHECK: llvm.mlir.global constant @[[global:.*]]("CUBIN")

  module attributes {gpu.kernel_module} {
    // CHECK-LABEL: func @kernel
    func @kernel(!llvm.float, !llvm<"float*">)
    // CHECK: attributes  {nvvm.cubingetter = @[[getter:.*]]}
    attributes  {nvvm.cubin = "CUBIN"}
  }

// CHECK: func @[[getter]]() -> !llvm<"i8*">
// CHECK: %[[addressof:.*]] = llvm.mlir.addressof @[[global]]
// CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index)
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[addressof]][%[[c0]], %[[c0]]]
// CHECK-SAME: -> !llvm<"i8*">
// CHECK: llvm.return %[[gep]] : !llvm<"i8*">
}
