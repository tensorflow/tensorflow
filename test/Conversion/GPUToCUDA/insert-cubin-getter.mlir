// RUN: mlir-opt %s --generate-cubin-accessors | FileCheck %s

// CHECK: llvm.global constant @[[global:.*]]("CUBIN")

func @kernel(!llvm.float, !llvm<"float*">)
// CHECK: attributes  {gpu.kernel, nvvm.cubin = "CUBIN", nvvm.cubingetter = @[[getter:.*]]}
  attributes  {gpu.kernel, nvvm.cubin = "CUBIN"}

// CHECK: func @[[getter]]() -> !llvm<"i8*">
// CHECK: %[[addressof:.*]] = llvm.addressof @[[global]]
// CHECK: %[[c0:.*]] = llvm.constant(0 : index)
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[addressof]][%[[c0]], %[[c0]]]
// CHECK-SAME: -> !llvm<"i8*">
// CHECK: llvm.return %[[gep]] : !llvm<"i8*">
