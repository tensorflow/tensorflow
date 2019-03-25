// RUN: mlir-opt %s -mlir-print-debuginfo -mlir-pretty-debuginfo | FileCheck %s

#set0 = (d0) : (1 == 0)

// CHECK-LABEL: inline_notation
// CHECK () -> i32 mysource.cc:10:8
func @inline_notation() -> i32 loc("mysource.cc":10:8) {
  // CHECK: -> i32 "foo"
  %1 = "foo"() : () -> i32 loc("foo")

  // CHECK: constant 4 : index "foo" at mysource.cc:10:8
  %2 = constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))

  // CHECK:      constant 4 : index "foo"
  // CHECK-NEXT:  at mysource1.cc:10:8
  // CHECK-NEXT:  at mysource2.cc:13:8
  // CHECK-NEXT:  at mysource3.cc:100:10
  %3 = constant 4 : index loc(callsite("foo" at callsite("mysource1.cc":10:8 at callsite("mysource2.cc":13:8 at "mysource3.cc":100:10))))

  // CHECK: } ["foo", mysource.cc:10:8]
  for %i0 = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8])

  // CHECK: } <"myPass">["foo", "foo2"]
  affine.if #set0(%2) {
  } loc(fused<"myPass">["foo", "foo2"])

  // CHECK: return %0 : i32 [unknown]
  return %1 : i32 loc(unknown)
}
