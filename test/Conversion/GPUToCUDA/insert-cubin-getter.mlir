// RUN: mlir-opt %s --generate-cubin-accessors | FileCheck %s

// CHECK: llvm.mlir.global constant @[[global:.*]]("CUBIN")

module attributes {gpu.kernel_module} {
  func @kernel(!llvm.float, !llvm<"float*">)
  attributes  {nvvm.cubin = "CUBIN"}
}

func @kernel(!llvm.float, !llvm<"float*">)
// CHECK: attributes  {gpu.kernel, nvvm.cubingetter = @[[getter:.*]]}
  attributes  {gpu.kernel}

// CHECK: func @[[getter]]() -> !llvm<"i8*">
// CHECK: %[[addressof:.*]] = llvm.mlir.addressof @[[global]]
// CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index)
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[addressof]][%[[c0]], %[[c0]]]
// CHECK-SAME: -> !llvm<"i8*">
// CHECK: llvm.return %[[gep]] : !llvm<"i8*">
