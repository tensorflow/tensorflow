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

func @branch_argument() -> () {
  %zero = spv.constant 0 : i32
  // CHECK: spv.Branch ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  spv.Branch ^next(%zero, %zero: i32, i32)
^next(%arg0: i32, %arg1: i32):
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

func @cond_branch_argument() -> () {
  %true = spv.constant true
  %zero = spv.constant 0 : i32
  // CHECK: spv.BranchConditional %{{.*}}, ^bb1(%{{.*}}, %{{.*}} : i32, i32), ^bb2
  spv.BranchConditional %true, ^true1(%zero, %zero: i32, i32), ^false1
^true1(%arg0: i32, %arg1: i32):
  // CHECK: spv.BranchConditional %{{.*}}, ^bb3, ^bb4(%{{.*}}, %{{.*}} : i32, i32)
  spv.BranchConditional %true, ^true2, ^false2(%zero, %zero: i32, i32)
^false1:
  spv.Return
^true2:
  spv.Return
^false2(%arg3: i32, %arg4: i32):
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
// spv.FunctionCall
//===----------------------------------------------------------------------===//

spv.module "Logical" "GLSL450" {
  func @fmain(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>, %arg2 : i32) -> i32 {
    // CHECK: {{%.*}} = spv.FunctionCall @f_0({{%.*}}, {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    %0 = spv.FunctionCall @f_0(%arg0, %arg1) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
    // CHECK: spv.FunctionCall @f_1({{%.*}}, {{%.*}}) : (vector<4xf32>, vector<4xf32>) -> ()
    spv.FunctionCall @f_1(%0, %arg1) : (vector<4xf32>, vector<4xf32>) ->  ()
    // CHECK: spv.FunctionCall @f_2() : () -> ()
    spv.FunctionCall @f_2() : () -> ()
    // CHECK: {{%.*}} = spv.FunctionCall @f_3({{%.*}}) : (i32) -> i32
    %1 = spv.FunctionCall @f_3(%arg2) : (i32) -> i32
    spv.ReturnValue %1 : i32
  }

  func @f_0(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> (vector<4xf32>) {
    spv.ReturnValue %arg0 : vector<4xf32>
  }

  func @f_1(%arg0 : vector<4xf32>, %arg1 : vector<4xf32>) -> () {
    spv.Return
  }

  func @f_2() -> () {
    spv.Return
  }

  func @f_3(%arg0 : i32) -> (i32) {
    spv.ReturnValue %arg0 : i32
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @f_invalid_result_type(%arg0 : i32, %arg1 : i32) -> () {
    // expected-error @+1 {{expected callee function to have 0 or 1 result, but provided 2}}
    %0 = spv.FunctionCall @f_invalid_result_type(%arg0, %arg1) : (i32, i32) -> (i32, i32)
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @f_result_type_mismatch(%arg0 : i32, %arg1 : i32) -> () {
    // expected-error @+1 {{has incorrect number of results has for callee: expected 0, but provided 1}}
    %1 = spv.FunctionCall @f_result_type_mismatch(%arg0, %arg0) : (i32, i32) -> (i32)
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @f_type_mismatch(%arg0 : i32, %arg1 : i32) -> () {
    // expected-error @+1 {{has incorrect number of operands for callee: expected 2, but provided 1}}
    spv.FunctionCall @f_type_mismatch(%arg0) : (i32) -> ()
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @f_type_mismatch(%arg0 : i32, %arg1 : i32) -> () {
    %0 = spv.constant 2.0 : f32
    // expected-error @+1 {{operand type mismatch: expected operand type 'i32', but provided 'f32' for operand number 1}}
    spv.FunctionCall @f_type_mismatch(%arg0, %0) : (i32, f32) -> ()
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @f_type_mismatch(%arg0 : i32, %arg1 : i32) -> i32 {
    // expected-error @+1 {{result type mismatch: expected 'i32', but provided 'f32'}}
    %0 = spv.FunctionCall @f_type_mismatch(%arg0, %arg0) : (i32, i32) -> f32
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @f_foo(%arg0 : i32, %arg1 : i32) -> i32 {
    // expected-error @+1 {{op callee function 'f_undefined' not found in 'spv.module'}}
    %0 = spv.FunctionCall @f_undefined(%arg0, %arg0) : (i32, i32) -> i32
    spv.Return
  }
}

// -----

func @f_foo(%arg0 : i32, %arg1 : i32) -> i32 {
    // expected-error @+1 {{must appear in a function inside 'spv.module'}}
    %0 = spv.FunctionCall @f_foo(%arg0, %arg0) : (i32, i32) -> i32
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
// spv._merge
//===----------------------------------------------------------------------===//

func @merge() -> () {
  // expected-error @+1 {{expected parent op to be 'spv.selection' or 'spv.loop'}}
  spv._merge
}

// -----

func @only_allowed_in_last_block(%cond : i1) -> () {
  %zero = spv.constant 0: i32
  %one = spv.constant 1: i32
  %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  spv.selection {
    spv.BranchConditional %cond, ^then, ^merge

  ^then:
    spv.Store "Function" %var, %one : i32
    // expected-error @+1 {{can only be used in the last block of 'spv.selection' or 'spv.loop'}}
    spv._merge

  ^merge:
    spv._merge
  }

  spv.Return
}

// -----

func @only_allowed_in_last_block() -> () {
  %true = spv.constant true
  spv.loop {
    spv.Branch ^header
  ^header:
    spv.BranchConditional %true, ^body, ^merge
  ^body:
    // expected-error @+1 {{can only be used in the last block of 'spv.selection' or 'spv.loop'}}
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

// CHECK-LABEL: func @in_selection
func @in_selection(%cond : i1) -> () {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^merge
  ^then:
    // CHECK: spv.Return
    spv.Return
  ^merge:
    spv._merge
  }
  spv.Return
}

// CHECK-LABEL: func @in_loop
func @in_loop(%cond : i1) -> () {
  spv.loop {
    spv.Branch ^header
  ^header:
    spv.BranchConditional %cond, ^body, ^merge
  ^body:
    // CHECK: spv.Return
    spv.Return
  ^continue:
    spv.Branch ^header
  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

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

// CHECK-LABEL: func @in_selection
func @in_selection(%cond : i1) -> (i32) {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^merge
  ^then:
    %zero = spv.constant 0 : i32
    // CHECK: spv.ReturnValue
    spv.ReturnValue %zero : i32
  ^merge:
    spv._merge
  }
  %one = spv.constant 1 : i32
  spv.ReturnValue %one : i32
}

// CHECK-LABEL: func @in_loop
func @in_loop(%cond : i1) -> (i32) {
  spv.loop {
    spv.Branch ^header
  ^header:
    spv.BranchConditional %cond, ^body, ^merge
  ^body:
    %zero = spv.constant 0 : i32
    // CHECK: spv.ReturnValue
    spv.ReturnValue %zero : i32
  ^continue:
    spv.Branch ^header
  ^merge:
    spv._merge
  }
  %one = spv.constant 1 : i32
  spv.ReturnValue %one : i32
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

// -----

//===----------------------------------------------------------------------===//
// spv.selection
//===----------------------------------------------------------------------===//

func @selection(%cond: i1) -> () {
  %zero = spv.constant 0: i32
  %one = spv.constant 1: i32
  %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  // CHECK: spv.selection {
  spv.selection {
    // CHECK-NEXT: spv.BranchConditional %{{.*}}, ^bb1, ^bb2
    spv.BranchConditional %cond, ^then, ^merge

  // CHECK: ^bb1
  ^then:
    spv.Store "Function" %var, %one : i32
    // CHECK: spv.Branch ^bb2
    spv.Branch ^merge

  // CHECK: ^bb2
  ^merge:
    // CHECK-NEXT: spv._merge
    spv._merge
  }

  spv.Return
}

// -----

func @selection(%cond: i1) -> () {
  %zero = spv.constant 0: i32
  %one = spv.constant 1: i32
  %two = spv.constant 2: i32
  %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  // CHECK: spv.selection {
  spv.selection {
    // CHECK-NEXT: spv.BranchConditional %{{.*}}, ^bb1, ^bb2
    spv.BranchConditional %cond, ^then, ^else

  // CHECK: ^bb1
  ^then:
    spv.Store "Function" %var, %one : i32
    // CHECK: spv.Branch ^bb3
    spv.Branch ^merge

  // CHECK: ^bb2
  ^else:
    spv.Store "Function" %var, %two : i32
    // CHECK: spv.Branch ^bb3
    spv.Branch ^merge

  // CHECK: ^bb3
  ^merge:
    // CHECK-NEXT: spv._merge
    spv._merge
  }

  spv.Return
}

// -----

// CHECK-LABEL: @empty_region
func @empty_region() -> () {
  // CHECK: spv.selection
  spv.selection {
  }
  return
}

// -----

func @wrong_merge_block() -> () {
  // expected-error @+1 {{last block must be the merge block with only one 'spv._merge' op}}
  spv.selection {
    spv.Return
  }
  return
}

// -----

func @missing_entry_block() -> () {
  // expected-error @+1 {{must have a selection header block}}
  spv.selection {
    spv._merge
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// spv.Unreachable
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @unreachable_no_pred
func @unreachable_no_pred() {
    spv.Return

  ^next:
    // CHECK: spv.Unreachable
    spv.Unreachable
}

// CHECK-LABEL: func @unreachable_with_pred
func @unreachable_with_pred() {
    spv.Return

  ^parent:
    spv.Branch ^unreachable

  ^unreachable:
    // CHECK: spv.Unreachable
    spv.Unreachable
}

// -----

func @unreachable() {
  // expected-error @+1 {{cannot be used in reachable block}}
  spv.Unreachable
}
