// RUN: mlir-opt -lower-to-llvm %s | FileCheck %s


// CHECK-LABEL: func @check_attributes(%arg0: !llvm<"float*"> {dialect.a: true, dialect.b: 4}) {
func @check_attributes(%static: memref<10x20xf32> {dialect.a: true, dialect.b: 4 }) {
  return
}

