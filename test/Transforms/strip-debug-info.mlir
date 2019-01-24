// RUN: mlir-opt %s -mlir-print-debuginfo -strip-debug-info | FileCheck %s
// This test verifies that debug locations are stripped.

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: inline_notation
// CHECK: () -> i32 loc(unknown) {
func @inline_notation() -> i32 loc("mysource.cc":10:8) {
  // CHECK: "foo"() : () -> i32 loc(unknown)
  %1 = "foo"() : () -> i32 loc("foo")

  // CHECK: for %i0 = 0 to 8 loc(unknown)
  for %i0 = 0 to 8 loc(fused["foo", "mysource.cc":10:8]) {
  }

  // CHECK: if #set0(%c4) loc(unknown)
  %2 = constant 4 : index
  if #set0(%2) loc(fused<"myPass">["foo", "foo2"]) {
  }

  // CHECK: return %0 : i32 loc(unknown)
  return %1 : i32 loc("bar")
}
