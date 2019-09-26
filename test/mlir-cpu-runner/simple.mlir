// RUN: echo TODO: re-enable
// DISABLED: mlir-cpu-runner %s | FileCheck %s
// DISABLED: mlir-cpu-runner %s -e foo | FileCheck -check-prefix=NOMAIN %s
// DISABLED: mlir-cpu-runner %s -O3 | FileCheck %s

// DISABLED: cp %s %t
// DISABLED: mlir-cpu-runner %t -dump-object-file | FileCheck %t
// DISABLED: ls %t.o
// DISABLED: rm %t.o

// DISABLED: mlir-cpu-runner %s -dump-object-file -object-filename=%T/test.o | FileCheck %s
// DISABLED: ls %T/test.o
// DISABLED: rm %T/test.o

// Declarations of C library functions.
func @fabsf(!llvm.float) -> !llvm.float
func @malloc(!llvm.i64) -> !llvm<"i8*">
func @free(!llvm<"i8*">)

// Check that a simple function with a nested call works.
func @main() -> !llvm.float {
  %0 = llvm.mlir.constant(-4.200000e+02 : f32) : !llvm.float
  %1 = llvm.call @fabsf(%0) : (!llvm.float) -> !llvm.float
  llvm.return %1 : !llvm.float
}
// CHECK: 4.200000e+02

// Helper typed functions wrapping calls to "malloc" and "free".
func @allocation() -> !llvm<"float*"> {
  %0 = llvm.mlir.constant(4 : index) : !llvm.i64
  %1 = llvm.call @malloc(%0) : (!llvm.i64) -> !llvm<"i8*">
  %2 = llvm.bitcast %1 : !llvm<"i8*"> to !llvm<"float*">
  llvm.return %2 : !llvm<"float*">
}
func @deallocation(%arg0: !llvm<"float*">) {
  %0 = llvm.bitcast %arg0 : !llvm<"float*"> to !llvm<"i8*">
  llvm.call @free(%0) : (!llvm<"i8*">) -> ()
  llvm.return
}

// Check that allocation and deallocation works, and that a custom entry point
// works.
func @foo() -> !llvm.float {
  %0 = llvm.call @allocation() : () -> !llvm<"float*">
  %1 = llvm.mlir.constant(0 : index) : !llvm.i64
  %2 = llvm.mlir.constant(1.234000e+03 : f32) : !llvm.float
  %3 = llvm.getelementptr %0[%1] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  llvm.store %2, %3 : !llvm<"float*">
  %4 = llvm.getelementptr %0[%1] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
  %5 = llvm.load %4 : !llvm<"float*">
  llvm.call @deallocation(%0) : (!llvm<"float*">) -> ()
  llvm.return %5 : !llvm.float
}
// NOMAIN: 1.234000e+03
