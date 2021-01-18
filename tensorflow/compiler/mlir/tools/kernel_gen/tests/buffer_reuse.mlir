// RUN: kernel-gen-opt %s --buffer-reuse | FileCheck %s

// CHECK-LABEL: @unique_reuse_output
func @unique_reuse_output() -> (index, memref<2x3xi64>) attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_output = 1 : i32
  %result_0 = constant 1 : index
  %result_1 = alloc() : memref<2x3xi64>
  return %result_0, %result_1 : index, memref<2x3xi64>
}

// CHECK-LABEL: @ambiguous_reuse_output
func @ambiguous_reuse_output(%pred : i1)
    -> (memref<2x3xi64>, memref<2x3xi64>) attributes {tf_entry} {
  // CHECK: alloc
  // CHECK: reuse_output = -1 : i32
  %mem = alloc() : memref<2x3xi64>
  %other_mem = alloc() : memref<2x3xi64>
  cond_br %pred, ^bb0, ^bb1
^bb0:
  return %mem, %other_mem : memref<2x3xi64>, memref<2x3xi64>
^bb1:
  return %other_mem, %mem : memref<2x3xi64>, memref<2x3xi64>
}

// CHECK-LABEL: @direct_reuse
func @direct_reuse(%not_a_memref : index,
                   %smaller : memref<5xi64>,
                   %greater : memref<7xi64>,
                   %different_element_type : memref<2x3xf32>,
                   %reusable_0 : memref<2x3xi64>,
                   %reusable_1 : memref<6xi64>) -> memref<2x3xi64>
                   attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [4 : i32, 5 : i32]
  %result = alloc() : memref<2x3xi64>
  return %result : memref<2x3xi64>
}

// CHECK-LABEL: @local_reuse_with_memref_maps
func @local_reuse_with_memref_maps(
    %arg : memref<?xi64, offset: 2, strides: [3]>, %n : index)
    -> memref<?xi64, offset: 2, strides: [3]> attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32]
  %result = alloc(%n) : memref<?xi64, offset: 2, strides: [3]>
  linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } ins(%arg : memref<?xi64, offset: 2, strides: [3]>)
    outs(%result : memref<?xi64, offset: 2, strides: [3]>) {
  ^bb0(%a : i64, %b : i64):
    linalg.yield %a : i64
  }
  return %result : memref<?xi64, offset: 2, strides: [3]>
}

// CHECK-LABEL: @memref_reinterpret_cast_alias
func @memref_reinterpret_cast_alias(%arg : memref<f32>, %n : index)
    -> memref<?xf32> attributes {tf_entry} {
  %c0 = constant 0 : index
  %reinterpreted = memref_reinterpret_cast %arg to
      offset: [0],
      sizes: [%n],
      strides: [%c0]: memref<f32> to memref<?xf32>

  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32]
  %result = alloc(%n) : memref<?xf32>

  // reinterpreted (arg) and result are of same size.
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%reinterpreted : memref<?xf32>) outs(%result : memref<?xf32>) {
  ^bb0(%a : f32, %b : f32):
    linalg.yield %a : f32
  }

  return %result : memref<?xf32>
}

// CHECK-LABEL: @memref_cast_alias
func @memref_cast_alias(%arg : memref<*xf32>, %n : index)
    -> memref<?xf32> attributes {tf_entry} {
  %casted = memref_cast %arg : memref<*xf32> to memref<?xf32>

  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32]
  %result = alloc(%n) : memref<?xf32>

  // reinterpreted (arg) and result are of same size.
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%casted : memref<?xf32>) outs(%result : memref<?xf32>) {
  ^bb0(%a : f32, %b : f32):
    linalg.yield %a : f32
  }

  return %result : memref<?xf32>
}

// CHECK-LABEL: @indirect_size_equality
func @indirect_size_equality(%arg0 : memref<?xi64>,
                             %arg1 : memref<?xi64>,
                             %n : index) -> memref<?xi64>
                             attributes {tf_entry} {
  // arg0 and arg1 are equal in size.
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : memref<?xi64>) outs(%arg1 : memref<?xi64>) {
  ^bb0(%a : i64, %b : i64):
    linalg.yield %a : i64
  }

  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32, 1 : i32]
  %result = alloc(%n) : memref<?xi64>

  // arg0 and result are equal in size.
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : memref<?xi64>) outs(%result : memref<?xi64>) {
  ^bb0(%a : i64, %b : i64):
    linalg.yield %a : i64
  }

  return %result : memref<?xi64>
}

// CHECK-LABEL: @livetimes_incompatible
func @livetimes_incompatible(%arg0 : memref<3xi64>)
    -> memref<3xi64> attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %result = alloc() : memref<3xi64>

  // Use newly allocated buffer.
  %c0 = constant 0 : index
  %0 = load %result[%c0] : memref<3xi64>

  // Use argument buffer again.
  %1 = load %arg0[%c0] : memref<3xi64>

  return %result : memref<3xi64>
}

// CHECK-LABEL: @never_used
func @never_used(%arg0 : memref<3xi64>) -> memref<3xi64> attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32]
  %result = alloc() : memref<3xi64>
  %c0 = constant 0 : index
  %0 = load %arg0[%c0] : memref<3xi64>
  return %result : memref<3xi64>
}

// CHECK-LABEL: @branching_reuse
func @branching_reuse(%pred : i1, %arg : memref<6xi64>) -> memref<6xi64>
    attributes {tf_entry} {
  cond_br %pred, ^bb0, ^bb1
^bb0:
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [1 : i32]
  %mem0 = alloc() : memref<6xi64>

  // Keep buffer argument live in this branch. Reuse is still possible because
  // the newly allocated buffer was not used yet.
  %c0 = constant 0 : index
  load %arg[%c0] : memref<6xi64>

  br ^bb2(%mem0 : memref<6xi64>)
^bb1:
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [1 : i32]
  %mem1 = alloc() : memref<6xi64>
  br ^bb2(%mem1 : memref<6xi64>)
^bb2(%result : memref<6xi64>):
  return %result : memref<6xi64>
}

// CHECK-LABEL: @branching_no_reuse
func @branching_no_reuse(%pred : i1, %arg : memref<6xi64>) -> memref<6xi64>
    attributes {tf_entry} {
  cond_br %pred, ^bb0, ^bb1
^bb0:
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %mem0 = alloc() : memref<6xi64>

  // Use newly allocated memory immediately.
  %c0 = constant 0 : index
  load %mem0[%c0] : memref<6xi64>

  // Keep buffer argument live in this branch and prevent reuse.
  load %arg[%c0] : memref<6xi64>

  br ^bb2(%mem0 : memref<6xi64>)
^bb1:
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [1 : i32]
  %mem1 = alloc() : memref<6xi64>
  br ^bb2(%mem1 : memref<6xi64>)
^bb2(%result : memref<6xi64>):
  return %result : memref<6xi64>
}

// CHECK-LABEL: @branching_reuse_if
func @branching_reuse_if(%pred : i1, %arg : memref<6xi64>)
    -> memref<6xi64> attributes {tf_entry} {
  %result = scf.if %pred -> (memref<6xi64>) {
    // CHECK: alloc
    // CHECK-SAME: reuse_input_candidates = [1 : i32]
    %mem0 = alloc() : memref<6xi64>

    // Keep buffer argument live in this branch. Reuse is still possible because
    // the newly allocated buffer was not used yet.
    %c0 = constant 0 : index
    load %arg[%c0] : memref<6xi64>

    scf.yield %mem0 : memref<6xi64>
  } else {
    // CHECK: alloc
    // CHECK-SAME: reuse_input_candidates = [1 : i32]
    %mem1 = alloc() : memref<6xi64>
    scf.yield %mem1 : memref<6xi64>
  }
  return %result : memref<6xi64>
}

// CHECK-LABEL: @branching_no_reuse_if
func @branching_no_reuse_if(%pred : i1, %arg : memref<6xi64>) -> memref<6xi64>
    attributes {tf_entry} {
  %result = scf.if %pred -> (memref<6xi64>) {
    // CHECK: alloc
    // CHECK-SAME: reuse_input_candidates = []
    %mem0 = alloc() : memref<6xi64>

    // Use newly allocated memory immediately.
    %c0 = constant 0 : index
    load %mem0[%c0] : memref<6xi64>

    // Keep buffer argument live in this branch and prevent reuse.
    load %arg[%c0] : memref<6xi64>

    scf.yield %mem0 : memref<6xi64>
  } else {
    // CHECK: alloc
    // CHECK-SAME: reuse_input_candidates = [1 : i32]
    %mem1 = alloc() : memref<6xi64>
    scf.yield %mem1 : memref<6xi64>
  }
  return %result : memref<6xi64>
}

// CHECK-LABEL: @alloc_before_branching
// New buffer is first used in the blocks succeeding its allocation block. In
// both/all cases the newly allocated buffer is used after the buffer argument
// is no longer live. Because these first uses are not block-local the analysis
// does not detect this case (yet). It is correct but incomplete.
func @alloc_before_branching(%pred : i1, %arg : memref<6xi64>) -> memref<6xi64>
    attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %mem = alloc() : memref<6xi64>
  %c0 = constant 0 : index
  cond_br %pred, ^bb0, ^bb1
^bb0:
  // Last use of `arg` before first use of `mem` (can reuse).
  load %arg[%c0] : memref<6xi64>
  load %mem[%c0] : memref<6xi64>
  return %mem : memref<6xi64>
^bb1:
  // Last use of `arg` before first use of `mem` (can reuse).
  load %arg[%c0] : memref<6xi64>
  load %mem[%c0] : memref<6xi64>
  return %mem : memref<6xi64>
}

// CHECK-LABEL: @alloc_before_branching_2
func @alloc_before_branching_2(%pred : i1, %arg : memref<6xi64>)
    -> memref<6xi64> attributes {tf_entry} {
  // CHECK: alloc()
  // CHECK-SAME: reuse_input_candidates = []
  %mem = alloc() : memref<6xi64>
  %c0 = constant 0 : index
  cond_br %pred, ^bb0, ^bb1
^bb0:
  // Last use of `arg` after first use of `mem` (cannot reuse).
  load %mem[%c0] : memref<6xi64>
  load %arg[%c0] : memref<6xi64>
  return %mem : memref<6xi64>
^bb1:
  // Last use of `arg` before first use of `mem` (can reuse).
  load %arg[%c0] : memref<6xi64>
  load %mem[%c0] : memref<6xi64>
  return %mem : memref<6xi64>
}

// CHECK-LABEL: @alloc_before_branching_if
// New buffer is first used in the blocks succeeding its allocation block. In
// both/all cases the newly allocated buffer is used after the buffer argument
// is no longer live. Because these first uses are not block-local the analysis
// does not detect this case (yet). It is correct but incomplete.
func @alloc_before_branching_if(%pred : i1, %arg : memref<6xi64>) -> memref<6xi64>
    attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %mem = alloc() : memref<6xi64>
  %result = scf.if %pred -> (memref<6xi64>) {
    // Last use of `arg` before first use of `mem` (can reuse).
    %c0 = constant 0 : index
    load %arg[%c0] : memref<6xi64>
    load %mem[%c0] : memref<6xi64>
    scf.yield %mem : memref<6xi64>
  } else {
    // Last use of `arg` before first use of `mem` (can reuse).
    %c0 = constant 0 : index
    load %arg[%c0] : memref<6xi64>
    load %mem[%c0] : memref<6xi64>
    scf.yield %mem : memref<6xi64>
  }
  return %result : memref<6xi64>
}

// CHECK-LABEL: @alloc_before_branching_2_if
func @alloc_before_branching_2_if(%pred : i1, %arg : memref<6xi64>)
    -> memref<6xi64> attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %mem = alloc() : memref<6xi64>
  %result = scf.if %pred -> (memref<6xi64>) {
    // Last use of `arg` after first use of `mem` (cannot reuse).
    %c0 = constant 0 : index
    load %mem[%c0] : memref<6xi64>
    load %arg[%c0] : memref<6xi64>
    scf.yield %mem : memref<6xi64>
  } else {
    // Last use of `arg` before first use of `mem` (can reuse).
    %c0 = constant 0 : index
    load %arg[%c0] : memref<6xi64>
    load %mem[%c0] : memref<6xi64>
    scf.yield %mem : memref<6xi64>
  }
  return %result : memref<6xi64>
}

// CHECK-LABEL: @abs_unranked_i64
func @abs_unranked_i64(%arg : memref<*xi64>,
                       %arg_shape : memref<?xindex>,
                       %flat_shape : memref<1xindex>,
                       %arg_size : index) -> memref<*xi64>
                       attributes {tf_entry} {
  %flat_arg = memref_reshape %arg(%flat_shape)
      : (memref<*xi64>, memref<1xindex>) -> memref<?xi64>
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32], reuse_output = 0 : i32
  %flat_result = alloc(%arg_size) : memref<?xi64>
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%flat_arg : memref<?xi64>) outs(%flat_result : memref<?xi64>) {
  ^bb0(%a : i64, %b : i64):
    %c0 = constant 0 : i64
    %a_pos = cmpi sge, %a, %c0 : i64
    %a_neg = subi %c0, %a : i64
    %a_abs = select %a_pos, %a, %a_neg : i64
    linalg.yield %a_abs : i64
  }
  %result = memref_reshape %flat_result(%arg_shape)
      : (memref<?xi64>, memref<?xindex>) -> memref<*xi64>
  return %result : memref<*xi64>
}

// CHECK-LABEL: @old_buffer_alias_outside_block
func @old_buffer_alias_outside_block(%arg: memref<3xf32>)
    attributes {llvm.emit_c_interface, tf_entry} {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %true = constant true

  // Alias outside of the block with the new buffer allocation.
  %alias = memref_cast %arg : memref<3xf32> to memref<3xf32>

  scf.if %true {

    // Allocation and use of new buffer.
    // CHECK: alloc
    // CHECK-SAME: reuse_input_candidates = [0 : i32]
    %mem = alloc() : memref<3xf32>
    %use = load %mem[%c0] : memref<3xf32>

  } else {
  }
  return
}

// CHECK-LABEL: @index_element_type
func @index_element_type(%arg : memref<2x3xindex>) -> memref<2x3xindex>
    attributes {tf_entry} {
  // CHECK: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32]
  %result = alloc() : memref<2x3xindex>
  return %result : memref<2x3xindex>
}

// Example as it occurs in the `tf.Abs` kernel for `f32`.
// CHECK-LABEL: @abs_f32
func @abs_f32(%arg0: memref<*xf32>) -> memref<*xf32>
    attributes {llvm.emit_c_interface, tf_entry} {
  %c0 = constant 0 : index
  %0 = shape.shape_of %arg0 : memref<*xf32> -> tensor<?xindex>
  %1 = shape.num_elements %0 : tensor<?xindex> -> index
  // CHECK-LABEL: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %2 = alloc() : memref<1xindex>
  store %1, %2[%c0] : memref<1xindex>
  %3 = memref_reshape %arg0(%2)
      : (memref<*xf32>, memref<1xindex>) -> memref<?xf32>
  %4 = dim %3, %c0 : memref<?xf32>
  %5 = index_cast %4 : index to i64
  // CHECK-LABEL: alloc
  // CHECK-SAME: reuse_input_candidates = []
  %6 = alloc() : memref<1xi64>
  store %5, %6[%c0] : memref<1xi64>
  %7 = load %6[%c0] : memref<1xi64>
  %8 = index_cast %7 : i64 to index
  // CHECK-LABEL: alloc
  // CHECK-SAME: reuse_input_candidates = [0 : i32]
  %9 = alloc(%8) : memref<?xf32>
  linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%3 : memref<?xf32>) outs(%9 : memref<?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %12 = absf %arg1 : f32
    linalg.yield %12 : f32
  }
  %10 = tensor_to_memref %0 : memref<?xindex>
  %11 = memref_reshape %9(%10)
      : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
  return %11 : memref<*xf32>
}
