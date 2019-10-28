// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt -mlir-print-op-generic -mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-GENERIC

// CHECK-LABEL: func @wrapping_op
// CHECK-GENERIC-LABEL: func @wrapping_op
func @wrapping_op(%arg0 : i32, %arg1 : f32) -> (i3, i2, i1) {
// CHECK: %0:3 = test.wrapping_region wraps "some.op"(%arg1, %arg0) {test.attr = "attr"} : (f32, i32) -> (i1, i2, i3)
// CHECK-GENERIC: "test.wrapping_region"() ( {
// CHECK-GENERIC:   %[[NESTED_RES:.*]]:3 = "some.op"(%arg1, %arg0) {test.attr = "attr"} : (f32, i32) -> (i1, i2, i3) loc("some_NameLoc")
// CHECK-GENERIC:   "test.return"(%[[NESTED_RES]]#0, %[[NESTED_RES]]#1, %[[NESTED_RES]]#2) : (i1, i2, i3) -> () loc("some_NameLoc")
// CHECK-GENERIC: }) : () -> (i1, i2, i3) loc("some_NameLoc")
  %res:3 = test.wrapping_region wraps "some.op"(%arg1, %arg0) { test.attr = "attr" } : (f32, i32) -> (i1, i2, i3) loc("some_NameLoc")
  return %res#2, %res#1, %res#0 : i3, i2, i1
}
