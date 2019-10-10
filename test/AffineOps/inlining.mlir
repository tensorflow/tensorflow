// RUN: mlir-opt %s -inline -mlir-disable-inline-simplify | FileCheck %s

// Basic test that functions within affine operations are inlined.
func @func_with_affine_ops(%N: index) {
  %c = constant 200 : index
  affine.for %i = 1 to 10 {
    affine.if (i)[N] : (i - 2 >= 0, 4 - i >= 0)(%i)[%c]  {
      %w = affine.apply (d0,d1)[s0] -> (d0+d1+s0) (%i, %i) [%N]
    }
  }
  return
}

// CHECK-LABEL: func @inline_with_affine_ops
func @inline_with_affine_ops() {
  %c = constant 1 : index

  // CHECK: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: affine.apply
  // CHECK-NOT: call
  call @func_with_affine_ops(%c) : (index) -> ()
  return
}

// CHECK-LABEL: func @not_inline_in_affine_op
func @not_inline_in_affine_op() {
  %c = constant 1 : index

  // CHECK-NOT: affine.if
  // CHECK: call
  affine.for %i = 1 to 10 {
    call @func_with_affine_ops(%c) : (index) -> ()
  }
  return
}

// -----

// Test when an invalid operation is nested in an affine op.
func @func_with_invalid_nested_op() {
  affine.for %i = 1 to 10 {
    "foo.opaque"() : () -> ()
  }
  return
}

// CHECK-LABEL: func @not_inline_invalid_nest_op
func @not_inline_invalid_nest_op() {
  // CHECK: call @func_with_invalid_nested_op
  call @func_with_invalid_nested_op() : () -> ()
  return
}

// -----

// Test that calls are not inlined into affine structures.
func @func_noop() {
  return
}

// CHECK-LABEL: func @not_inline_into_affine_ops
func @not_inline_into_affine_ops() {
  // CHECK: call @func_noop
  affine.for %i = 1 to 10 {
    call @func_noop() : () -> ()
  }
  return
}
