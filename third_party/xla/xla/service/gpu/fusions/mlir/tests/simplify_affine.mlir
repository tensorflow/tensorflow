// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-simplify-affine | FileCheck %s

func.func @op_and_for_ranges(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %1 = gpu.block_id  x {xla.range = [0 : index, 3071 : index]}
  scf.for %arg3 = %c0 to %c4 step %c1 {
    %2 = affine.apply affine_map<()[s0, s1, s2] -> (s0 * 512 + s1 * 4 + s2 - ((s1 * 4 + s2) floordiv 256) * 256 + (s1 floordiv 64) * 256 - ((s0 * 2 + s1 floordiv 64) floordiv 3) * 768 + ((s0 * 128 + s1) floordiv 192) * 768 - (((s0 * 128 + s1) floordiv 192) floordiv 1024) * 786432 + (s0 floordiv 1536) * 786432)>()[%1, %0, %arg3]
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
// CHECK:       %[[BLOCK_OFFSET:.*]] = arith.muli %[[BID_X]], %[[C512]]
// CHECK:       %[[THREAD_OFFSET:.*]] = arith.muli %[[TID_X]], %[[C4]]
// CHECK:       %[[OFFSET:.*]] = arith.addi %[[BLOCK_OFFSET]], %[[THREAD_OFFSET]]
// CHECK:       arith.addi %[[OFFSET]], %[[I]]

// -----

func.func @arg_ranges(%arg0: index {xla.range = [0 : index, 42 : index]}, %arg1: index {xla.range = [0 : index, 1000 : index]}) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100)>()[%arg0, %arg1]
  return %0 : index
}

// CHECK-LABEL: @arg_ranges
// CHECK-NEXT:  %[[C100:.*]] = arith.constant 100
// CHECK-NEXT:  %[[RET:.*]] = arith.divui %{{.*}}, %[[C100]]
// CHECK-NEXT:  return %[[RET]]

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
    %2 = xla_gpu.apply_indexing affine_map<()[s0, s1, s2] -> (s0 * 512 + s1 * 4 + s2 - ((s1 * 4 + s2) floordiv 256) * 256 + (s1 floordiv 64) * 256 - ((s0 * 2 + s1 floordiv 64) floordiv 3) * 768 + ((s0 * 128 + s1) floordiv 192) * 768 - (((s0 * 128 + s1) floordiv 192) floordiv 1024) * 786432 + (s0 floordiv 1536) * 786432)>
      [%1 in [0, 3071], %0 in [0, 127], %i in [0, 3]]
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
// CHECK:       %[[BLOCK_OFFSET:.*]] = arith.muli %[[BID_X]], %[[C512]]
// CHECK:       %[[THREAD_OFFSET:.*]] = arith.muli %[[TID_X]], %[[C4]]
// CHECK:       %[[OFFSET:.*]] = arith.addi %[[BLOCK_OFFSET]], %[[THREAD_OFFSET]]
// CHECK:       arith.addi %[[OFFSET]], %[[I]]

// -----

func.func @arg_ranges(%arg0: index, %arg1: index) -> index {
  %0 = xla_gpu.apply_indexing
    affine_map<()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100)>
    [%arg0 in [0, 42], %arg1 in [0, 1000]]
  return %0 : index
}

// CHECK-LABEL: @arg_ranges
// CHECK-NEXT:  %[[C100:.*]] = arith.constant 100
// CHECK-NEXT:  %[[RET:.*]] = arith.divui %{{.*}}, %[[C100]]
// CHECK-NEXT:  return %[[RET]]

// -----

func.func @cant_lower(%arg0: index, %arg1: index) -> index {
  %0 = xla_gpu.apply_indexing
    affine_map<()[s0, s1] -> (s0 floordiv 100 + s1 floordiv 100)>
    [%arg0 in [-10, 42], %arg1 in [0, 1000]]
  return %0 : index
}

// CHECK-LABEL:       @cant_lower
// CHECK:       affine.apply
