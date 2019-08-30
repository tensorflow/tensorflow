// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Branch
//===----------------------------------------------------------------------===//

func @branch() -> () {
  // CHECK: spv.Branch ^bb1
  spv.Branch ^next
^next:
  spv.Return
}

// -----

func @missing_accessor() -> () {
  spv.Branch
  // expected-error @+1 {{expected block name}}
}

// -----

func @wrong_accessor_count() -> () {
  %true = spv.constant true
  // expected-error @+1 {{must have exactly one successor}}
  "spv.Branch"()[^one, ^two] : () -> ()
^one:
  spv.Return
^two:
  spv.Return
}

// -----

func @accessor_argument_disallowed() -> () {
  %zero = spv.constant 0 : i32
  // expected-error @+1 {{requires zero operands}}
  "spv.Branch"()[^next(%zero : i32)] : () -> ()
^next(%arg: i32):
  spv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spv.BranchConditional
//===----------------------------------------------------------------------===//

func @cond_branch() -> () {
  %true = spv.constant true
  // CHECK: spv.BranchConditional %{{.*}}, ^bb1, ^bb2
  spv.BranchConditional %true, ^one, ^two
// CHECK: ^bb1
^one:
  spv.Return
// CHECK: ^bb2
^two:
  spv.Return
}

// -----

func @cond_branch_with_weights() -> () {
  %true = spv.constant true
  // CHECK: spv.BranchConditional %{{.*}} [5, 10]
  spv.BranchConditional %true [5, 10], ^one, ^two
^one:
  spv.Return
^two:
  spv.Return
}

// -----

func @missing_condition() -> () {
  // expected-error @+1 {{expected SSA operand}}
  spv.BranchConditional ^one, ^two
^one:
  spv.Return
^two:
  spv.Return
}

// -----

func @wrong_condition_type() -> () {
  // expected-note @+1 {{prior use here}}
  %zero = spv.constant 0 : i32
  // expected-error @+1 {{use of value '%zero' expects different type than prior uses: 'i1' vs 'i32'}}
  spv.BranchConditional %zero, ^one, ^two
^one:
  spv.Return
^two:
  spv.Return
}

// -----

func @wrong_accessor_count() -> () {
  %true = spv.constant true
  // expected-error @+1 {{must have exactly two successors}}
  "spv.BranchConditional"(%true)[^one] : (i1) -> ()
^one:
  spv.Return
^two:
  spv.Return
}

// -----

func @accessor_argment_disallowed() -> () {
  %true = spv.constant true
  // expected-error @+1 {{requires a single operand}}
  "spv.BranchConditional"(%true)[^one(%true : i1), ^two] : (i1) -> ()
^one(%arg : i1):
  spv.Return
^two:
  spv.Return
}

// -----

func @wrong_number_of_weights() -> () {
  %true = spv.constant true
  // expected-error @+1 {{must have exactly two branch weights}}
  "spv.BranchConditional"(%true)[^one, ^two] {branch_weights = [1 : i32, 2 : i32, 3 : i32]} : (i1) -> ()
^one:
  spv.Return
^two:
  spv.Return
}

// -----

func @weights_cannot_both_be_zero() -> () {
  %true = spv.constant true
  // expected-error @+1 {{branch weights cannot both be zero}}
  spv.BranchConditional %true [0, 0], ^one, ^two
^one:
  spv.Return
^two:
  spv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spv.Return
//===----------------------------------------------------------------------===//

"foo.function"() ({
  // expected-error @+1 {{op must appear in a 'func' block}}
  spv.Return
})  : () -> ()

// -----

// Return mismatches function signature
spv.module "Logical" "VulkanKHR" {
  func @work() -> (i32) {
    // expected-error @+1 {{cannot be used in functions returning value}}
    spv.Return
  }
}

// -----

//===----------------------------------------------------------------------===//
// spv.ReturnValue
//===----------------------------------------------------------------------===//

func @ret_val() -> (i32) {
  %0 = spv.constant 42 : i32
  // CHECK: spv.ReturnValue %{{.*}} : i32
  spv.ReturnValue %0 : i32
}

// -----

"foo.function"() ({
  %0 = spv.constant true
  // expected-error @+1 {{op must appear in a 'func' block}}
  spv.ReturnValue %0 : i1
})  : () -> ()

// -----

func @value_count_mismatch() -> () {
  %0 = spv.constant 42 : i32
  // expected-error @+1 {{op returns 1 value but enclosing function requires 0 results}}
  spv.ReturnValue %0 : i32
}

// -----

func @value_type_mismatch() -> (f32) {
  %0 = spv.constant 42 : i32
  // expected-error @+1 {{return value's type ('i32') mismatch with function's result type ('f32')}}
  spv.ReturnValue %0 : i32
}
