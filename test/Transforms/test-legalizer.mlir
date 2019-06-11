// RUN: mlir-test-opt -test-legalize-patterns %s | FileCheck %s

// CHECK-LABEL: verifyDirectPattern
func @verifyDirectPattern() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() {status: "Success"}
  %result = "test.illegal_op_a"() : () -> (i32)
  return %result : i32
}

// CHECK-LABEL: verifyLargerBenefit
func @verifyLargerBenefit() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() {status: "Success"}
  %result = "test.illegal_op_c"() : () -> (i32)
  return %result : i32
}
