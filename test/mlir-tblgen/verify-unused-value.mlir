// RUN: mlir-opt -test-patterns %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 'verifyUnusedValue'
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @match_success_on_unused_first_result
func @match_success_on_unused_first_result(%arg0 : i32) -> i32 {
  // CHECK-NEXT: return {{.*}} : i32
  %result:2 = "test.vuv_two_result_op"(%arg0) : (i32) -> (i32, i32)
  return %result#1 : i32
}

// CHECK-LABEL: @match_fail_on_used_first_result
func @match_fail_on_used_first_result(%arg0 : i32) -> i32 {
  // CHECK-NEXT: "test.vuv_two_result_op"(%arg0) : (i32) -> (i32, i32)
  %result:2 = "test.vuv_two_result_op"(%arg0) : (i32) -> (i32, i32)
  "foo.unknown_op"(%result#0) : (i32) -> ()
  return %result#1 : i32
}
