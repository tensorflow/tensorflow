// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: succeededSameOperandAndResultElementType
func @succeededSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>) {
  %0 = "test.same_operand_and_result_type"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "test.same_operand_and_result_type"(%t1, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<1xf32>
  %2 = "test.same_operand_and_result_type"(%t10x10, %v1) : (tensor<10x10xf32>, vector<1xf32>) -> tensor<1xf32>
  %3 = "test.same_operand_and_result_type"(%v1, %t1) : (vector<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %4 = "test.same_operand_and_result_type"(%v1, %t1) : (vector<1xf32>, tensor<1xf32>) -> tensor<121xf32>
  return
}

// -----

func @failedSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "test.same_operand_and_result_type"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi32>
}

// -----

func @failedSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "test.same_operand_and_result_type"(%t1, %t1i) : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>
}

// -----

// CHECK: succeededSameOperandAndResultShape
func @succeededSameOperandAndResultShape(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %tr: tensor<*xf32>) {
  %0 = "test.same_operand_and_result_shape"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "test.same_operand_and_result_shape"(%t10x10, %t10x10) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "test.same_operand_and_result_shape"(%t1, %tr) : (tensor<1xf32>, tensor<*xf32>) -> tensor<1xf32>
  return
}

// -----

func @succeededSameOperandAndResultShape(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>) {
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %0 = "test.same_operand_and_result_shape"(%t1, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func @hasParent() {
  "some.op"() ({
   // expected-error@+1 {{'test.child' op expects parent op 'test.parent'}}
    "test.child"() : () -> ()
  }) : () -> ()
}

// -----

func @singleBlockImplicitTerminator() {
   // expected-error@+1 {{'test.SingleBlockImplicitTerminator' op expects a non-empty block}}
  "test.SingleBlockImplicitTerminator"() ({
  ^entry:
  }) : () -> ()
}

// -----

func @singleBlockImplicitTerminator() {
   // expected-error@+1 {{'test.SingleBlockImplicitTerminator' op expects region #0 to have 0 or 1 block}}
  "test.SingleBlockImplicitTerminator"() ({
  ^entry:
    "test.finish" () : () -> ()
  ^other:
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

func @singleBlockImplicitTerminator() {
   // expected-error@+2 {{'test.SingleBlockImplicitTerminator' op expects regions to end with 'test.finish'}}
   // expected-note@+1 {{in custom textual format, the absence of terminator implies 'test.finish'}}
  "test.SingleBlockImplicitTerminator"() ({
  ^entry:
    "test.non_existent_op"() : () -> ()
  }) : () -> ()
}

