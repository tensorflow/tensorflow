// RUN: mlir-opt %s -split-input-file -pass-pipeline='spv.module(inline)' -mlir-disable-inline-simplify | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @callee() {
    spv.Return
  }

  // CHECK-LABEL: func @calling_single_block_ret_func
  func @calling_single_block_ret_func() {
    // CHECK-NEXT: spv.Return
    spv.FunctionCall @callee() : () -> ()
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @callee() -> i32 {
    %0 = spv.constant 42 : i32
    spv.ReturnValue %0 : i32
  }

  // CHECK-LABEL: func @calling_single_block_retval_func
  func @calling_single_block_retval_func() -> i32 {
    // CHECK-NEXT: %[[CST:.*]] = spv.constant 42
    %0 = spv.FunctionCall @callee() : () -> (i32)
    // CHECK-NEXT: spv.ReturnValue %[[CST]]
    spv.ReturnValue %0 : i32
  }
}

// -----

spv.module "Logical" "GLSL450" {
  spv.globalVariable @data bind(0, 0) : !spv.ptr<!spv.struct<!spv.rtarray<i32> [0]>, StorageBuffer>
  func @callee() {
    %0 = spv._address_of @data : !spv.ptr<!spv.struct<!spv.rtarray<i32> [0]>, StorageBuffer>
    %1 = spv.constant 0: i32
    %2 = spv.AccessChain %0[%1, %1] : !spv.ptr<!spv.struct<!spv.rtarray<i32> [0]>, StorageBuffer>
    spv.Branch ^next

  ^next:
    %3 = spv.constant 42: i32
    spv.Store "StorageBuffer" %2, %3 : i32
    spv.Return
  }

  // CHECK-LABEL: func @calling_multi_block_ret_func
  func @calling_multi_block_ret_func() {
    // CHECK-NEXT:   spv._address_of
    // CHECK-NEXT:   spv.constant 0
    // CHECK-NEXT:   spv.AccessChain
    // CHECK-NEXT:   spv.Branch ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spv.constant
    // CHECK-NEXT:   spv.Store
    // CHECK-NEXT:   spv.Branch ^bb2
    spv.FunctionCall @callee() : () -> ()
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spv.Return
    spv.Return
  }
}

// TODO: calling_multi_block_retval_func

// -----

spv.module "Logical" "GLSL450" {
  func @callee(%cond : i1) -> () {
    spv.selection {
      spv.BranchConditional %cond, ^then, ^merge
    ^then:
      spv.Return
    ^merge:
      spv._merge
    }
    spv.Return
  }

  // CHECK-LABEL: calling_selection_ret_func
  func @calling_selection_ret_func() {
    %0 = spv.constant true
    // CHECK: spv.FunctionCall
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @callee(%cond : i1) -> () {
    spv.selection {
      spv.BranchConditional %cond, ^then, ^merge
    ^then:
      spv.Branch ^merge
    ^merge:
      spv._merge
    }
    spv.Return
  }

  // CHECK-LABEL: calling_selection_no_ret_func
  func @calling_selection_no_ret_func() {
    // CHECK-NEXT: %[[TRUE:.*]] = spv.constant true
    %0 = spv.constant true
    // CHECK-NEXT: spv.selection
    // CHECK-NEXT:   spv.BranchConditional %[[TRUE]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spv.Branch ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spv._merge
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @callee(%cond : i1) -> () {
    spv.loop {
      spv.Branch ^header
    ^header:
      spv.BranchConditional %cond, ^body, ^merge
    ^body:
      spv.Return
    ^continue:
      spv.Branch ^header
    ^merge:
      spv._merge
    }
    spv.Return
  }

  // CHECK-LABEL: calling_loop_ret_func
  func @calling_loop_ret_func() {
    %0 = spv.constant true
    // CHECK: spv.FunctionCall
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
  func @callee(%cond : i1) -> () {
    spv.loop {
      spv.Branch ^header
    ^header:
      spv.BranchConditional %cond, ^body, ^merge
    ^body:
      spv.Branch ^continue
    ^continue:
      spv.Branch ^header
    ^merge:
      spv._merge
    }
    spv.Return
  }

  // CHECK-LABEL: calling_loop_no_ret_func
  func @calling_loop_no_ret_func() {
    // CHECK-NEXT: %[[TRUE:.*]] = spv.constant true
    %0 = spv.constant true
    // CHECK-NEXT: spv.loop
    // CHECK-NEXT:   spv.Branch ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   spv.BranchConditional %[[TRUE]], ^bb2, ^bb4
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT:   spv.Branch ^bb3
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   spv.Branch ^bb1
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:   spv._merge
    spv.FunctionCall @callee(%0) : (i1) -> ()
    spv.Return
  }
}
