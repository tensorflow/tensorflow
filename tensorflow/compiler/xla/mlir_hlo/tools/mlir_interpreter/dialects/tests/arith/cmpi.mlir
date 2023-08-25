// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @eq() -> (i1, i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %e1 = arith.cmpi eq, %c1, %c2 : index
  %e2 = arith.cmpi eq, %c1, %c1 : index
  return %e1, %e2 : i1, i1
}

// CHECK-LABEL: @eq
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true

func.func @ne() -> (i1, i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %e1 = arith.cmpi ne, %c1, %c2 : index
  %e2 = arith.cmpi ne, %c1, %c1 : index
  return %e1, %e2 : i1, i1
}

// CHECK-LABEL: @ne
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false

func.func @slt() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi slt, %c-1, %c1 : index
  %e2 = arith.cmpi slt, %c1, %c-1 : index
  %e3 = arith.cmpi slt, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @slt
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: false

func.func @sle() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi sle, %c-1, %c1 : index
  %e2 = arith.cmpi sle, %c1, %c-1 : index
  %e3 = arith.cmpi sle, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @sle
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true

func.func @sgt() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi sgt, %c-1, %c1 : index
  %e2 = arith.cmpi sgt, %c1, %c-1 : index
  %e3 = arith.cmpi sgt, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @sgt
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false

func.func @sge() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi sge, %c-1, %c1 : index
  %e2 = arith.cmpi sge, %c1, %c-1 : index
  %e3 = arith.cmpi sge, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @sge
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: true

func.func @ult() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi ult, %c-1, %c1 : index
  %e2 = arith.cmpi ult, %c1, %c-1 : index
  %e3 = arith.cmpi ult, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @ult
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false

func.func @ule() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi ule, %c-1, %c1 : index
  %e2 = arith.cmpi ule, %c1, %c-1 : index
  %e3 = arith.cmpi ule, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @ule
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: true

func.func @ugt() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi ugt, %c-1, %c1 : index
  %e2 = arith.cmpi ugt, %c1, %c-1 : index
  %e3 = arith.cmpi ugt, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @ugt
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: false

func.func @uge() -> (i1, i1, i1) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %e1 = arith.cmpi uge, %c-1, %c1 : index
  %e2 = arith.cmpi uge, %c1, %c-1 : index
  %e3 = arith.cmpi uge, %c1, %c1 : index
  return %e1, %e2, %e3 : i1, i1, i1
}

// CHECK-LABEL: @uge
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: true
// CHECK-NEXT{LITERAL}: false
// CHECK-NEXT{LITERAL}: true
