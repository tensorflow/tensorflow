// RUN: xla-runtime-opt %s --xla-rt-convert-asserts | FileCheck %s

// CHECK: func @exported(
// CHECK:  %[[CTX:.*]]: !rt.execution_context,
// CHECK:  %[[PRED:.*]]: i1
// CHECK: )
func.func @exported(%arg0: !rt.execution_context, %arg1: i1)
    attributes {rt.exported = 0 : i32} {
  // CHECK:  cf.cond_br %[[PRED]], ^[[OK:.*]], ^[[ERR:.*]]
  // CHECK:  ^[[OK]]:
  // CHECK:  return
  // CHECK:  ^[[ERR]]:
  // CHECK:  rt.set_error %[[CTX]], "Oops"
  // CHECK:  return
  cf.assert %arg1, "Oops"
  return
}
