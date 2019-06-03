// RUN: mlir-opt %s -split-input-file -verify

// -----

func @func_op() {
  // expected-error@+1 {{expected non-function type}}
  func missingsigil() -> (i1, index, f32)
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{expected type instead of SSA identifier}}
  func @mixed_named_arguments(f32, %a : i32) {
    return
  }
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{expected SSA identifier}}
  func @mixed_named_arguments(%a : i32, f32) -> () {
    return
  }
  return
}

// -----

func @func_op() {
  // expected-error@+2 {{optional region with explicit entry arguments must be defined}}
  func @mixed_named_arguments(%a : i32)
  return
}
