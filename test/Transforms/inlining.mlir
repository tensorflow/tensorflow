// RUN: mlir-opt %s -inline -mlir-disable-inline-simplify | FileCheck %s
// RUN: mlir-opt %s -inline -mlir-disable-inline-simplify -mlir-print-debuginfo | FileCheck %s --check-prefix INLINE-LOC
// RUN: mlir-opt %s -inline -mlir-disable-inline-simplify=false | FileCheck %s --check-prefix INLINE_SIMPLIFY

// Inline a function that takes an argument.
func @func_with_arg(%c : i32) -> i32 {
  %b = addi %c, %c : i32
  return %b : i32
}

// CHECK-LABEL: func @inline_with_arg
func @inline_with_arg(%arg0 : i32) -> i32 {
  // CHECK-NEXT: addi
  // CHECK-NEXT: return

  %0 = call @func_with_arg(%arg0) : (i32) -> i32
  return %0 : i32
}

// Inline a function that has multiple return operations.
func @func_with_multi_return(%a : i1) -> (i32) {
  cond_br %a, ^bb1, ^bb2

^bb1:
  %const_0 = constant 0 : i32
  return %const_0 : i32

^bb2:
  %const_55 = constant 55 : i32
  return %const_55 : i32
}

// CHECK-LABEL: func @inline_with_multi_return() -> i32
func @inline_with_multi_return() -> i32 {
// CHECK-NEXT:    [[VAL_7:%.*]] = constant 0 : i1
// CHECK-NEXT:    cond_br [[VAL_7]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[VAL_8:%.*]] = constant 0 : i32
// CHECK-NEXT:    br ^bb3([[VAL_8]] : i32)
// CHECK:       ^bb2:
// CHECK-NEXT:    [[VAL_9:%.*]] = constant 55 : i32
// CHECK-NEXT:    br ^bb3([[VAL_9]] : i32)
// CHECK:       ^bb3([[VAL_10:%.*]]: i32):
// CHECK-NEXT:    return [[VAL_10]] : i32

  %false = constant 0 : i1
  %x = call @func_with_multi_return(%false) : (i1) -> i32
  return %x : i32
}

// Check that location information is updated for inlined instructions.
func @func_with_locations(%c : i32) -> i32 {
  %b = addi %c, %c : i32 loc("mysource.cc":10:8)
  return %b : i32 loc("mysource.cc":11:2)
}

// INLINE-LOC-LABEL: func @inline_with_locations
func @inline_with_locations(%arg0 : i32) -> i32 {
  // INLINE-LOC-NEXT: addi %{{.*}}, %{{.*}} : i32 loc(callsite("mysource.cc":10:8 at "mysource.cc":55:14))
  // INLINE-LOC-NEXT: return

  %0 = call @func_with_locations(%arg0) : (i32) -> i32 loc("mysource.cc":55:14)
  return %0 : i32
}


// Check that external functions are not inlined.
func @func_external()

// CHECK-LABEL: func @no_inline_external
func @no_inline_external() {
  // CHECK-NEXT: call @func_external()
  call @func_external() : () -> ()
  return
}

// Check that multiple levels of calls will be inlined.
func @multilevel_func_a() {
  return
}
func @multilevel_func_b() {
  call @multilevel_func_a() : () -> ()
  return
}

// CHECK-LABEL: func @inline_multilevel
func @inline_multilevel() {
  // CHECK-NOT: call
  %fn = "test.functional_region_op"() ({
    call @multilevel_func_b() : () -> ()
    "test.return"() : () -> ()
  }) : () -> (() -> ())

  call_indirect %fn() : () -> ()
  return
}

// Check that recursive calls are not inlined.
// CHECK-LABEL: func @no_inline_recursive
func @no_inline_recursive() {
  // CHECK: test.functional_region_op
  // CHECK-NOT: test.functional_region_op
  %fn = "test.functional_region_op"() ({
    call @no_inline_recursive() : () -> ()
    "test.return"() : () -> ()
  }) : () -> (() -> ())
  return
}

// Check that we can convert types for inputs and results as necessary.
func @convert_callee_fn(%arg : i32) -> i32 {
  return %arg : i32
}
func @convert_callee_fn_multi_arg(%a : i32, %b : i32) -> () {
  return
}
func @convert_callee_fn_multi_res() -> (i32, i32) {
  %res = constant 0 : i32
  return %res, %res : i32, i32
}

// CHECK-LABEL: func @inline_convert_call
func @inline_convert_call() -> i16 {
  // CHECK: %[[INPUT:.*]] = constant
  %test_input = constant 0 : i16

  // CHECK: %[[CAST_INPUT:.*]] = "test.cast"(%[[INPUT]]) : (i16) -> i32
  // CHECK: %[[CAST_RESULT:.*]] = "test.cast"(%[[CAST_INPUT]]) : (i32) -> i16
  // CHECK-NEXT: return %[[CAST_RESULT]]
  %res = "test.conversion_call_op"(%test_input) { callee=@convert_callee_fn } : (i16) -> (i16)
  return %res : i16
}

// CHECK-LABEL: func @no_inline_convert_call
func @no_inline_convert_call() {
  // CHECK: "test.conversion_call_op"
  %test_input_i16 = constant 0 : i16
  %test_input_i64 = constant 0 : i64
  "test.conversion_call_op"(%test_input_i16, %test_input_i64) { callee=@convert_callee_fn_multi_arg } : (i16, i64) -> ()

  // CHECK: "test.conversion_call_op"
  %res_2:2 = "test.conversion_call_op"() { callee=@convert_callee_fn_multi_res } : () -> (i16, i64)
  return
}

// Check that we properly simplify when inlining.
func @simplify_return_constant() -> i32 {
  %res = constant 0 : i32
  return %res : i32
}

func @simplify_return_reference() -> (() -> i32) {
  %res = constant @simplify_return_constant : () -> i32
  return %res : () -> i32
}

// INLINE_SIMPLIFY-LABEL: func @inline_simplify
func @inline_simplify() -> i32 {
  // INLINE_SIMPLIFY-NEXT: %[[CST:.*]] = constant 0 : i32
  // INLINE_SIMPLIFY-NEXT: return %[[CST]]
  %fn = call @simplify_return_reference() : () -> (() -> i32)
  %res = call_indirect %fn() : () -> i32
  return %res : i32
}
