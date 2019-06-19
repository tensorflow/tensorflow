// RUN: mlir-opt %s -split-input-file -verify-diagnostics

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

// -----

func @func_op() {
  // expected-error@+1 {{entry block must have 1 arguments to match function signature}}
  func @mixed_named_arguments(f32) {
  ^entry:
    return
  }
  return
}

// -----

func @func_op() {
  // expected-error@+1 {{type of entry block argument #0('i32') must match the type of the corresponding argument in function signature('f32')}}
  func @mixed_named_arguments(f32) {
  ^entry(%arg : i32):
    return
  }
  return
}
