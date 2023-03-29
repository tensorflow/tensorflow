// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @addi() -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %ret = arith.addi %c1, %c2 : i32
  return %ret : i32
}

// CHECK-LABEL: @addi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 3

func.func @muli() -> i32 {
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %ret = arith.muli %c3, %c5 : i32
  return %ret : i32
}

// CHECK-LABEL: @muli
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 15

func.func @divsi() -> i32 {
  %c10 = arith.constant 10 : i32
  %c-2 = arith.constant -2 : i32
  %ret = arith.divsi %c10, %c-2 : i32
  return %ret : i32
}

// CHECK-LABEL: @divsi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: -5

func.func @subi() -> i32 {
  %c10 = arith.constant 10 : i32
  %c3 = arith.constant 3 : i32
  %ret = arith.subi %c10, %c3 : i32
  return %ret : i32
}

// CHECK-LABEL: @subi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 7

func.func @andi() -> i32 {
  %c63 = arith.constant 63 : i32
  %c131 = arith.constant 131 : i32
  %ret = arith.andi %c63, %c131 : i32
  return %ret : i32
}

// CHECK-LABEL: @andi
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 3

func.func @ori() -> i32 {
  %c3 = arith.constant 3 : i32
  %c10 = arith.constant 10 : i32
  %ret = arith.ori %c3, %c10 : i32
  return %ret : i32
}

// CHECK-LABEL: @ori
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 11

func.func @shrui() -> i32 {
  %c3 = arith.constant 20 : i32
  %c-1 = arith.constant -1 : i32
  %ret = arith.shrui %c-1, %c3 : i32
  return %ret : i32
}

// CHECK-LABEL: @shrui
// CHECK-NEXT: Results
// CHECK-NEXT: 4095

func.func @shli() -> i32 {
  %c2 = arith.constant 2 : i32
  %c42 = arith.constant 42 : i32
  %ret = arith.shli %c42, %c2 : i32
  return %ret : i32
}

// CHECK-LABEL: @shli
// CHECK-NEXT: Results
// CHECK-NEXT: 168

