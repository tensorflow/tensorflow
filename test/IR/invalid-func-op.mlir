// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @func_op() {
  // expected-error@+1 {{expected valid '@'-identifier for symbol name}}
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

// -----

// expected-error@+1 {{expected non-function type}}
func @f() -> (foo

// -----

// expected-error@+1 {{expected attribute name}}
func @f() -> (i1 {)

// -----

// expected-error@+1 {{invalid to use 'test.invalid_attr'}}
func @f(%arg0: i64 {test.invalid_attr}) {
  return
}

// -----

// expected-error@+1 {{invalid to use 'test.invalid_attr'}}
func @f(%arg0: i64) -> (i64 {test.invalid_attr}) {
  return %arg0 : i64
}
