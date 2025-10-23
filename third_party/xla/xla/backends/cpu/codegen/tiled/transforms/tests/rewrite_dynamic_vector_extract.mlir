// RUN: emitters_opt %s \
// RUN: -xtile-cpu-rewrite-dynamic-vector-extract -canonicalize \
// RUN: -split-input-file | FileCheck %s

func.func @fold_vector_extract_into_transfer_read(
  %buffer: memref<8x4x2xf32>,
  %idx0: index,
  %idx1: index) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %mask = vector.create_mask %c7, %c3, %c1 : vector<8x4x2xi1>
  %original_vector = vector.transfer_read %buffer[%c0, %c0, %c0],
    %c0_f32, %mask : memref<8x4x2xf32>, vector<8x4x2xf32>
  %subvector = vector.extract %original_vector[%idx0, %idx1]
    : vector<2xf32> from vector<8x4x2xf32>
  return %subvector : vector<2xf32>
}

// CHECK:      func.func @fold_vector_extract_into_transfer_read(
// CHECK-SAME:   %[[BUFFER:.*]]: memref<8x4x2xf32>,
// CHECK-SAME:   %[[IDX0:.*]]: index,
// CHECK-SAME:   %[[IDX1:.*]]: index) -> vector<2xf32> {
// CHECK-DAG:    %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:    %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:    %[[C7:.*]] = arith.constant 7 : index
// CHECK:        %[[SHIFT_IDX0:.*]] = arith.subi %[[C7]], %[[IDX0]] : index
// CHECK:        %[[SHIFT_SUBIDX1:.*]] = arith.subi %[[C3]], %[[IDX1]] : index
// CHECK:        %[[SHIFT_MASK:.*]] = vector.create_mask
// CHECK-SAME:     %[[SHIFT_IDX0]], %[[SHIFT_SUBIDX1]], %[[C1]] : vector<8x4x2xi1>
// CHECK:        %[[SUBMASK:.*]] = vector.extract %[[SHIFT_MASK]][0, 0]
// CHECK-SAME:     : vector<2xi1> from vector<8x4x2xi1>
// CHECK:        %[[SUBVECTOR:.*]] = vector.transfer_read
// CHECK-SAME:     %[[BUFFER]][%[[IDX0]], %[[IDX1]], %[[C0]]], %[[PAD]], %[[SUBMASK]]
// CHECK-SAME:     {in_bounds = [true]} : memref<8x4x2xf32>, vector<2xf32>
// CHECK:        return %[[SUBVECTOR]] : vector<2xf32>
// CHECK:      }


// -----

func.func @unroll_dependent_vector_extract(%input: vector<8x2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_f32 = arith.constant 0. : f32
  %init = vector.broadcast %c0_f32 : f32 to vector<2xf32>
  %result = scf.for %index = %c0 to %c8 step %c1 iter_args(%carry = %init) -> vector<2xf32> {
    %extract = vector.extract %input[%index] : vector<2xf32> from vector<8x2xf32>
    %add = arith.addf %carry, %extract : vector<2xf32>
    scf.yield %add : vector<2xf32>
  }
  return %result : vector<2xf32>
}

// CHECK-LABEL:    func.func @unroll_dependent_vector_extract(
// CHECK-NOT:       scf.for
// CHECK-COUNT-8:     vector.extract

// -----

func.func @unroll_indirect_dependent_vector_extract(%input: vector<8x2xf32>) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c0_f32 = arith.constant 0. : f32
  %init = vector.broadcast %c0_f32 : f32 to vector<2xf32>
  %result = scf.for %index = %c0 to %c4 step %c1 iter_args(%carry = %init) -> vector<2xf32> {
    %strided_index = arith.muli %index, %c2 : index
    %extract = vector.extract %input[%strided_index] : vector<2xf32> from vector<8x2xf32>
    %add = arith.addf %carry, %extract : vector<2xf32>
    scf.yield %add : vector<2xf32>
  }
  return %result : vector<2xf32>
}

// CHECK-LABEL:    func.func @unroll_indirect_dependent_vector_extract(
// CHECK-NOT:        scf.for
// CHECK-COUNT-4:     vector.extract

// -----

func.func @does_not_unroll_independent_vector_extract(%input: vector<8x2xf32>, %arg_index: index) -> vector<2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_f32 = arith.constant 0. : f32
  %init = vector.broadcast %c0_f32 : f32 to vector<2xf32>
  %result = scf.for %index = %c0 to %c8 step %c1 iter_args(%carry = %init) -> vector<2xf32> {
    %extract = vector.extract %input[%arg_index] : vector<2xf32> from vector<8x2xf32>
    %add = arith.addf %carry, %extract : vector<2xf32>
    scf.yield %add : vector<2xf32>
  }
  return %result : vector<2xf32>
}

// CHECK-LABEL:    func.func @does_not_unroll_independent_vector_extract(
// CHECK:            scf.for
// CHECK-COUNT-1:     vector.extract
