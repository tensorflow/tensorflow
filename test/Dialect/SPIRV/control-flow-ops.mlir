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
// spv.loop
//===----------------------------------------------------------------------===//

// for (int i = 0; i < count; ++i) {}
func @loop(%count : i32) -> () {
  %zero = spv.constant 0: i32
  %one = spv.constant 1: i32
  %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  // CHECK: spv.loop {
  spv.loop {
    // CHECK-NEXT: spv.Branch ^bb1
    spv.Branch ^header

  // CHECK-NEXT: ^bb1:
  ^header:
    %val0 = spv.Load "Function" %var : i32
    %cmp = spv.SLessThan %val0, %count : i32
    // CHECK: spv.BranchConditional %{{.*}}, ^bb2, ^bb4
    spv.BranchConditional %cmp, ^body, ^merge

  // CHECK-NEXT: ^bb2:
  ^body:
    // Do nothing
    // CHECK-NEXT: spv.Branch ^bb3
    spv.Branch ^continue

  // CHECK-NEXT: ^bb3:
  ^continue:
    %val1 = spv.Load "Function" %var : i32
    %add = spv.IAdd %val1, %one : i32
    spv.Store "Function" %var, %add : i32
    // CHECK: spv.Branch ^bb1
    spv.Branch ^header

  // CHECK-NEXT: ^bb4:
  ^merge:
    spv._merge
  }
  return
}

// -----

// CHECK-LABEL: @empty_region
func @empty_region() -> () {
  // CHECK: spv.loop
  spv.loop {
  }
  return
}

// -----

func @wrong_merge_block() -> () {
  // expected-error @+1 {{last block must be the merge block with only one 'spv._merge' op}}
  spv.loop {
    spv.Return
  }
  return
}

// -----

func @missing_entry_block() -> () {
  // expected-error @+1 {{must have an entry block branching to the loop header block}}
  spv.loop {
    spv._merge
  }
  return
}

// -----

func @missing_header_block() -> () {
  // expected-error @+1 {{must have a loop header block branched from the entry block}}
  spv.loop {
  ^entry:
    spv.Branch ^merge
  ^merge:
    spv._merge
  }
  return
}

// -----

func @entry_should_branch_to_header() -> () {
  // expected-error @+1 {{entry block must only have one 'spv.Branch' op to the second block}}
  spv.loop {
  ^entry:
    spv.Branch ^merge
  ^header:
    spv.Branch ^merge
  ^merge:
    spv._merge
  }
  return
}

// -----

func @missing_continue_block() -> () {
  // expected-error @+1 {{requires a loop continue block branching to the loop header block}}
  spv.loop {
  ^entry:
    spv.Branch ^header
  ^header:
    spv.Branch ^merge
  ^merge:
    spv._merge
  }
  return
}

// -----

func @continue_should_branch_to_header() -> () {
  // expected-error @+1 {{second to last block must be the loop continue block that branches to the loop header block}}
  spv.loop {
  ^entry:
    spv.Branch ^header
  ^header:
    spv.Branch ^continue
  ^continue:
    spv.Branch ^merge
  ^merge:
    spv._merge
  }
  return
}

// -----

func @only_entry_and_continue_branch_to_header() -> () {
  // expected-error @+1 {{can only have the entry and loop continue block branching to the loop header block}}
  spv.loop {
  ^entry:
    spv.Branch ^header
  ^header:
    spv.Branch ^cont1
  ^cont1:
    spv.Branch ^header
  ^cont2:
    spv.Branch ^header
  ^merge:
    spv._merge
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.merge
//===----------------------------------------------------------------------===//

func @merge() -> () {
  // expected-error @+1 {{expects parent op 'spv.loop'}}
  spv._merge
}

// -----

func @only_allowed_in_last_block() -> () {
  %true = spv.constant true
  spv.loop {
    spv.Branch ^header
  ^header:
    spv.BranchConditional %true, ^body, ^merge
  ^body:
    // expected-error @+1 {{can only be used in the last block of 'spv.loop'}}
    spv._merge
  ^continue:
    spv.Branch ^header
  ^merge:
    spv._merge
  }
  return
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
spv.module "Logical" "GLSL450" {
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
