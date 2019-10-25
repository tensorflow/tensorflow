// RUN: mlir-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.AccsessChain
//===----------------------------------------------------------------------===//

func @combine_full_access_chain() -> f32 {
  // CHECK: %[[INDEX:.*]] = spv.constant 0
  // CHECK-NEXT: %[[VAR:.*]] = spv.Variable
  // CHECK-NEXT: %[[PTR:.*]] = spv.AccessChain %[[VAR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]]
  // CHECK-NEXT: spv.Load "Function" %[[PTR]]
  %c0 = spv.constant 0: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %1[%c0, %c0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %3 = spv.Load "Function" %2 : f32
  spv.ReturnValue %3 : f32
}

// -----

func @combine_access_chain_multi_use() -> !spv.array<4xf32> {
  // CHECK: %[[INDEX:.*]] = spv.constant 0
  // CHECK-NEXT: %[[VAR:.*]] = spv.Variable
  // CHECK-NEXT: %[[PTR_0:.*]] = spv.AccessChain %[[VAR]][%[[INDEX]], %[[INDEX]]]
  // CHECK-NEXT: %[[PTR_1:.*]] = spv.AccessChain %[[VAR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]]
  // CHECK-NEXT: spv.Load "Function" %[[PTR_0]]
  // CHECK-NEXT: spv.Load "Function" %[[PTR_1]]
  %c0 = spv.constant 0: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.AccessChain %0[%c0] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %1[%c0] : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
  %3 = spv.AccessChain %2[%c0] : !spv.ptr<!spv.array<4xf32>, Function>
  %4 = spv.Load "Function" %2 : !spv.array<4xf32>
  %5 = spv.Load "Function" %3 : f32
  spv.ReturnValue %4: !spv.array<4xf32>
}

// -----

func @dont_combine_access_chain_without_common_base() -> !spv.array<4xi32> {
  // CHECK: %[[INDEX:.*]] = spv.constant 1
  // CHECK-NEXT: %[[VAR_0:.*]] = spv.Variable
  // CHECK-NEXT: %[[VAR_1:.*]] = spv.Variable
  // CHECK-NEXT: %[[VAR_0_PTR:.*]] = spv.AccessChain %[[VAR_0]][%[[INDEX]]]
  // CHECK-NEXT: %[[VAR_1_PTR:.*]] = spv.AccessChain %[[VAR_1]][%[[INDEX]]]
  // CHECK-NEXT: spv.Load "Function" %[[VAR_0_PTR]]
  // CHECK-NEXT: spv.Load "Function" %[[VAR_1_PTR]]
  %c1 = spv.constant 1: i32
  %0 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %1 = spv.Variable : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %2 = spv.AccessChain %0[%c1] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %3 = spv.AccessChain %1[%c1] : !spv.ptr<!spv.struct<!spv.array<4x!spv.array<4xf32>>, !spv.array<4xi32>>, Function>
  %4 = spv.Load "Function" %2 : !spv.array<4xi32>
  %5 = spv.Load "Function" %3 : !spv.array<4xi32>
  spv.ReturnValue %4 : !spv.array<4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.CompositeExtract
//===----------------------------------------------------------------------===//

// CHECK-LABEL: extract_vector
func @extract_vector() -> (i32, i32, i32) {
  // CHECK: spv.constant 42 : i32
  // CHECK: spv.constant -33 : i32
  // CHECK: spv.constant 6 : i32
  %0 = spv.constant dense<[42, -33, 6]> : vector<3xi32>
  %1 = spv.CompositeExtract %0[0 : i32] : vector<3xi32>
  %2 = spv.CompositeExtract %0[1 : i32] : vector<3xi32>
  %3 = spv.CompositeExtract %0[2 : i32] : vector<3xi32>
  return %1, %2, %3 : i32, i32, i32
}

// -----

// CHECK-LABEL: extract_array_final
func @extract_array_final() -> (i32, i32) {
  // CHECK: spv.constant 4 : i32
  // CHECK: spv.constant -5 : i32
  %0 = spv.constant [dense<[4, -5]> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.CompositeExtract %0[0 : i32, 0 : i32] : !spv.array<1 x vector<2 x i32>>
  %2 = spv.CompositeExtract %0[0 : i32, 1 : i32] : !spv.array<1 x vector<2 x i32>>
  return %1, %2 : i32, i32
}

// -----

// CHECK-LABEL: extract_array_interm
func @extract_array_interm() -> (vector<2xi32>) {
  // CHECK: spv.constant dense<[4, -5]> : vector<2xi32>
  %0 = spv.constant [dense<[4, -5]> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.CompositeExtract %0[0 : i32] : !spv.array<1 x vector<2 x i32>>
  return %1 : vector<2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

// TODO(antiagainst): test constants in different blocks

func @deduplicate_scalar_constant() -> (i32, i32) {
  // CHECK: %[[CST:.*]] = spv.constant 42 : i32
  %0 = spv.constant 42 : i32
  %1 = spv.constant 42 : i32
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : i32, i32
}

// -----

func @deduplicate_vector_constant() -> (vector<3xi32>, vector<3xi32>) {
  // CHECK: %[[CST:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %0 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : vector<3xi32>, vector<3xi32>
}

// -----

func @deduplicate_composite_constant() -> (!spv.array<1 x vector<2xi32>>, !spv.array<1 x vector<2xi32>>) {
  // CHECK: %[[CST:.*]] = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %0 = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  %1 = spv.constant [dense<5> : vector<2xi32>] : !spv.array<1 x vector<2xi32>>
  // CHECK-NEXT: return %[[CST]], %[[CST]]
  return %0, %1 : !spv.array<1 x vector<2xi32>>, !spv.array<1 x vector<2xi32>>
}

// -----

//===----------------------------------------------------------------------===//
// spv.selection
//===----------------------------------------------------------------------===//

func @canonicalize_selection_op_scalar_type(%cond: i1) -> () {
  %0 = spv.constant 0: i32
  // CHECK: %[[TRUE_VALUE:.*]] = spv.constant 1 : i32
  %1 = spv.constant 1: i32
  // CHECK: %[[FALSE_VALUE:.*]] = spv.constant 2 : i32
  %2 = spv.constant 2: i32
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<i32, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<i32, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, i32
  // CHECK-NEXT: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE]] ["Aligned", 4] : i32
  // CHECK-NEXT: spv.Return
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^else:
    spv.Store "Function" %3, %2 ["Aligned", 4]: i32
    spv.Branch ^merge

  ^then:
    spv.Store "Function" %3, %1 ["Aligned", 4]: i32
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

func @canonicalize_selection_op_vector_type(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[TRUE_VALUE:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[FALSE_VALUE:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: %[[SRC_VALUE:.*]] = spv.Select {{%.*}}, %[[TRUE_VALUE]], %[[FALSE_VALUE]] : i1, vector<3xi32>
  // CHECK-NEXT: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE]] ["Aligned", 8] : vector<3xi32>
  // CHECK-NEXT: spv.Return
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// Store to a different variables.
func @cannot_canonicalize_selection_op_0(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR_0:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>
  // CHECK: %[[DST_VAR_1:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %4 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR_0]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %4, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// A conditional block consists of more than 2 operations.
func @cannot_canonicalize_selection_op_1(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR_0:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>
  // CHECK: %[[DST_VAR_1:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %4 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR_0]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8] : vector<3xi32>
    // CHECK: spv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %4, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR_1]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %4, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// A control-flow goes into `^then` block from `^else` block.
func @cannot_canonicalize_selection_op_2(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^then

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// `spv.Return` as a block terminator.
func @cannot_canonicalize_selection_op_3(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 8]:  vector<3xi32>
    spv.Return

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}

// -----

// Different memory access attributes.
func @cannot_canonicalize_selection_op_4(%cond: i1) -> () {
  %0 = spv.constant dense<[0, 1, 2]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_0:.*]] = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  %1 = spv.constant dense<[1, 2, 3]> : vector<3xi32>
  // CHECK: %[[SRC_VALUE_1:.*]] = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xi32>
  // CHECK: %[[DST_VAR:.*]] = spv.Variable init({{%.*}}) : !spv.ptr<vector<3xi32>, Function>
  %3 = spv.Variable init(%0) : !spv.ptr<vector<3xi32>, Function>

  // CHECK: spv.selection {
  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_0]] ["Aligned", 4] : vector<3xi32>
    spv.Store "Function" %3, %1 ["Aligned", 4]:  vector<3xi32>
    spv.Branch ^merge

  ^else:
    // CHECK: spv.Store "Function" %[[DST_VAR]], %[[SRC_VALUE_1]] ["Aligned", 8] : vector<3xi32>
    spv.Store "Function" %3, %2 ["Aligned", 8] : vector<3xi32>
    spv.Branch ^merge

  ^merge:
    spv._merge
  }
  spv.Return
}
