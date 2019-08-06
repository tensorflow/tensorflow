// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @remove_op_with_inner_ops
func @remove_op_with_inner_ops() {
  // CHECK-NEXT: return
  "test.op_with_region"() ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}
