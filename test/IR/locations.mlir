// RUN: mlir-opt %s -mlir-print-debuginfo | FileCheck %s
// This test verifies that debug locations are round-trippable.

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: inline_notation
// CHECK () -> i32 loc("mysource.cc":10:8)
func @inline_notation() -> i32 loc("mysource.cc":10:8) {
  // CHECK: -> i32 loc("foo")
  %1 = "foo"() : () -> i32 loc("foo")

  // CHECK: constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))
  %2 = constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))

  // CHECK: } loc(fused["foo", "mysource.cc":10:8])
  affine.for %i0 = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8])

  // CHECK: } loc(fused<"myPass">["foo", "foo2"])
  if #set0(%2) {
  } loc(fused<"myPass">["foo", "foo2"])

  // CHECK: return %0 : i32 loc(unknown)
  return %1 : i32 loc(unknown)
}
