// RUN: emitters_opt --allow-unregistered-dialect %s -split-input-file -xla-gpu-simplify-affine -cse | FileCheck %s

func.func @op_and_for_ranges(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %1 = gpu.block_id  x {xla.range = [0 : index, 3071 : index]}
  scf.for %arg3 = %c0 to %c4 step %c1 {
    %2 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 512 + s1 * 4 + s2 + (s1 floordiv 128) + (s2 floordiv 4))>()[%1, %0, %arg3]
    %3 = arith.index_castui %2 : index to i64
    %4 = llvm.getelementptr %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %5 = llvm.load %4 invariant : !llvm.ptr -> f32
    %8 = llvm.getelementptr %arg1[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %9 = llvm.load %8 invariant : !llvm.ptr -> f32
    %10 = arith.cmpf oge, %5, %9 : f32
    %11 = llvm.getelementptr %arg2[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %10, %11 : i1, !llvm.ptr
  }
  return
}

// CHECK-LABEL: @op_and_for_ranges
// CHECK-DAG: %[[C512:.*]] = arith.constant 512
// CHECK-DAG: %[[C4:.*]] = arith.constant 4
// CHECK-DAG: %[[TID_X:.*]] = gpu.thread_id x
// CHECK-DAG: %[[BID_X:.*]] = gpu.block_id x
// CHECK:     scf.for %[[I:.*]] =
// CHECK-DAG:   %[[CAST_TID_X:.*]] = arith.index_castui %[[TID_X]]
// CHECK-DAG:   %[[CAST_BID_X:.*]] = arith.index_castui %[[BID_X]]
// CHECK-DAG:   %[[BLOCK_OFFSET:.*]] = arith.muli %[[CAST_BID_X]], %[[C512]]
// CHECK-DAG:   %[[THREAD_OFFSET:.*]] = arith.muli %[[CAST_TID_X]], %[[C4]]
// CHECK:       %[[OFFSET:.*]] = arith.addi %[[BLOCK_OFFSET]], %[[THREAD_OFFSET]]
// CHECK:       %[[I64:.*]] = arith.index_castui %[[I]]
// CHECK:       arith.addi %[[OFFSET]], %[[I64]]

// -----

func.func @arg_ranges(%arg0: index {xla.range = [0 : index, 42 : index]}, %arg1: index {xla.range = [0 : index, 1000 : index]}) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100)>()[%arg0, %arg1]
  return %0 : index
}

// CHECK-LABEL: @arg_ranges
// CHECK:  %[[C100:.*]] = arith.constant 100
// CHECK:  %[[RET:.*]] = arith.divui %{{.*}}, %[[C100]]
// CHECT:  return {{.*}} : index

// -----

func.func @cant_lower(%arg0: index {xla.range = [-10 : index, 42 : index]}, %arg1: index {xla.range = [0 : index, 1000 : index]}) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100)>()[%arg0, %arg1]
  return %0 : index
}

// CHECK-LABEL:       @cant_lower
// CHECK:       affine.apply

// -----

func.func @op_and_for_ranges(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = gpu.thread_id  x
  %1 = gpu.block_id  x
  scf.for %i = %c0 to %c4 step %c1 {
    %2 = xla.apply_indexing
      #xla.indexing_map<
        "()[s0, s1, s2] -> (s0 * 512 + s1 * 4 + s2 + (s1 floordiv 128) + (s2 floordiv 4)),"
        "domain: s0 in [0, 3071], s1 in [0, 127], s2 in [0, 3]">[%1, %0, %i]
    %3 = arith.index_castui %2 : index to i64
    %4 = llvm.getelementptr %arg0[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %5 = llvm.load %4 invariant : !llvm.ptr -> f32
    %8 = llvm.getelementptr %arg1[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %9 = llvm.load %8 invariant : !llvm.ptr -> f32
    %10 = arith.cmpf oge, %5, %9 : f32
    %11 = llvm.getelementptr %arg2[%3] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    llvm.store %10, %11 : i1, !llvm.ptr
  }
  return
}

// CHECK-LABEL: @op_and_for_ranges
// CHECK-DAG: %[[C512:.*]] = arith.constant 512
// CHECK-DAG: %[[C4:.*]] = arith.constant 4
// CHECK-DAG: %[[TID_X:.*]] = gpu.thread_id x
// CHECK-DAG: %[[BID_X:.*]] = gpu.block_id x
// CHECK:     scf.for %[[I:.*]] =
// CHECK-DAG:   %[[CAST_TID_X:.*]] = arith.index_castui %[[TID_X]]
// CHECK-DAG:   %[[CAST_BID_X:.*]] = arith.index_castui %[[BID_X]]
// CHECK-DAG:   %[[BLOCK_OFFSET:.*]] = arith.muli %[[CAST_BID_X]], %[[C512]]
// CHECK-DAG:   %[[THREAD_OFFSET:.*]] = arith.muli %[[CAST_TID_X]], %[[C4]]
// CHECK:       %[[OFFSET:.*]] = arith.addi %[[BLOCK_OFFSET]], %[[THREAD_OFFSET]]
// CHECK:       %[[I64:.*]] = arith.index_castui %[[I]]
// CHECK:       arith.addi %[[OFFSET]], %[[I64]]

// -----

func.func @arg_ranges(%arg0: index, %arg1: index) -> index {
  %0 = xla.apply_indexing
    #xla.indexing_map<
      "()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100),"
      "domain: s0 in [0, 42], s1 in [0, 1000]">[%arg0, %arg1]
  return %0 : index
}

// CHECK-LABEL: @arg_ranges
// CHECK:  %[[C100:.*]] = arith.constant 100
// CHECK:  %[[RET:.*]] = arith.divui %{{.*}}, %[[C100]] : i64
// CHECK:  return {{.*}} : index

// -----

func.func @cant_lower(%arg0: index, %arg1: index) -> (index, index) {
  %0:2 = xla.apply_indexing
    #xla.indexing_map<"()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100, s0 + s1),"
  "domain: s0 in [-10, 42], s1 in [0, 1000]">[%arg0, %arg1]
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: @cant_lower
// CHECK:         affine.apply
// CHECK:         arith.addi

// -----

func.func @order_summands(%arg1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %arg2 = %c0 to %c4 step %c1 {
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = xla.apply_indexing
        #xla.indexing_map<
          "()[s0, s1, s2] -> ((s0 + s1) floordiv 3 + s0 * 512 + s1 * 4 + s2 * 10),"
          "domain: s0 in [0, 3], s1 in [0, 3], s2 in [0, 3]">[%arg2, %arg1, %arg3]
      "dummy.op"(%0) : (index) -> ()
    }
  }
  return
}

// CHECK-LABEL: @order_summands
// CHECK-SAME:    (%[[ARG1:.*]]: index)
// CHECK: scf.for %[[ARG2:.*]] =
// CHECK: scf.for %[[ARG3:.*]] =
// CHECK-DAG: %[[ARG1_CAST:.*]] = arith.index_castui %[[ARG1]]
// CHECK-DAG: %[[ARG2_CAST:.*]] = arith.index_castui %[[ARG2]]
// CHECK-DAG: arith.muli %[[ARG1_CAST]]
// CHECK-DAG: arith.muli %[[ARG2_CAST]]
// CHECK:     arith.addi
// CHECK:     arith.addi %[[ARG1_CAST]], %[[ARG2_CAST]]
// CHECK:     arith.divui
// CHECK:     arith.addi
// CHECK:     %[[ARG3_CAST:.*]] = arith.index_castui %[[ARG3]]
// CHECK:     arith.muli %[[ARG3_CAST]]
// CHECK:     arith.addi %7, %9 : i64

// -----

func.func @index_binop_canonicalization(
    %p0: i64, %p1: i64, %p2: i64, %p3: i64, %p4: i64, %p5: i64) -> i64 {
  %i0 = arith.index_castui %p0 : i64 to index
  %i1 = arith.index_castui %p1 : i64 to index
  %i2 = arith.index_castui %p2 : i64 to index
  %i3 = arith.index_castui %p3 : i64 to index
  %i4 = arith.index_castui %p4 : i64 to index
  %i5 = arith.index_castui %p5 : i64 to index
  %add = arith.addi %i0, %i1 : index
  %sub = arith.subi %add, %i2 : index
  %mul = arith.muli %sub, %i3 : index
  %div = arith.divui %mul, %i4 : index
  %rem = arith.remui %div, %i5 : index
  %result = arith.index_castui %rem : index to i64
  return %result : i64
}

// CHECK-LABEL: @index_binop_canonicalization
// CHECK: arith.addi {{.*}} : i64
// CHECK: arith.subi {{.*}} : i64
// CHECK: arith.muli {{.*}} : i64
// CHECK: arith.divui {{.*}} : i64
// CHECK: arith.remui {{.*}} : i64
// CHECK-NOT: arith.index_castui
