// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK-DAG: #[[SUB:.*]] = ()[s0, s1] -> (s0 - s1)

func @fold_constants(%arg0: !linalg.buffer<?xf32>) -> (index, index, index, index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %R02 = linalg.range %c0:%c2:%c1 : !linalg.range
  %R03 = linalg.range %c0:%c3:%c1 : !linalg.range
  %R04 = linalg.range %c0:%c4:%c1 : !linalg.range
  %R12 = linalg.range %c1:%c2:%c1 : !linalg.range
  %R13 = linalg.range %c1:%c3:%c1 : !linalg.range
  %R14 = linalg.range %c1:%c4:%c1 : !linalg.range

  %v = linalg.view %arg0[%R02, %R14] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  // Expected 2.
  %v0 = linalg.dim %v, 0 : !linalg.view<?x?xf32>
  // Expected 3.
  %v1 = linalg.dim %v, 1 : !linalg.view<?x?xf32>

  %s = linalg.slice %v[%c1, %R12] : !linalg.view<?x?xf32>, index, !linalg.range, !linalg.view<?xf32>
  // Expected 1.
  %s0 = linalg.dim %s, 0 : !linalg.view<?xf32>

  %sv = linalg.subview %v[%v0, %v1, %c1, %c2, %c4, %c1] : !linalg.view<?x?xf32>
  // Expected 1.
  %sv0 = linalg.dim %sv, 0 : !linalg.view<?x?xf32>
  // Expected 2.
  %sv1 = linalg.dim %sv, 1 : !linalg.view<?x?xf32>

  return %v0, %v1, %s0, %sv0, %sv1 : index, index, index, index, index
}

// CHECK-LABEL: fold_constants
//   CHECK-DAG:   %[[c1:.*]] = constant 1 : index
//   CHECK-DAG:   %[[c2:.*]] = constant 2 : index
//   CHECK-DAG:   %[[c3:.*]] = constant 3 : index
//       CHECK:   return %[[c2]], %[[c3]], %[[c1]], %[[c1]], %[[c2]]


func @fold_indices(%arg0: !linalg.buffer<?xf32>, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %R = linalg.range %arg1:%arg3:%c1 : !linalg.range

  %v = linalg.view %arg0[%R, %R] : !linalg.buffer<?xf32> -> !linalg.view<?x?xf32>
  // Expected %arg3 - %arg1.
  %v0 = linalg.dim %v, 0 : !linalg.view<?x?xf32>
  // Expected %arg3 - %arg1.
  %v1 = linalg.dim %v, 1 : !linalg.view<?x?xf32>

  %arg1_p_arg2 = addi %arg1, %arg2: index
  %arg1_p_arg2_affine = affine.apply (i, j) -> (i + j) (%arg1, %arg2)
  %sv = linalg.subview %v[%arg1, %arg1_p_arg2, %c1, %arg1, %arg1_p_arg2_affine, %c1] : !linalg.view<?x?xf32>
  // Expected %arg2 but can't fold affine.apply with addi.
  %sv0 = linalg.dim %sv, 0 : !linalg.view<?x?xf32>
  // Expected %arg2.
  %sv1 = linalg.dim %sv, 1 : !linalg.view<?x?xf32>

  return %v0, %v1, %sv0, %sv1 : index, index, index, index
}

// CHECK-LABEL: fold_indices
//       CHECK: (%[[arg0:.*]]: !linalg.buffer<?xf32>, %[[arg1:.*]]: index, %[[arg2:.*]]: index, %[[arg3:.*]]: index
//       CHECK:   %[[r0:.*]] = affine.apply #[[SUB]]()[%[[arg3]], %[[arg1]]]
//       CHECK:   %[[r1:.*]] = affine.apply #[[SUB]]()[%[[arg3]], %[[arg1]]]
//       CHECK:   %[[add:.*]] = addi %[[arg1]], %[[arg2]] : index
//       CHECK:   %[[aff:.*]] = affine.apply #[[SUB]]()[%[[add]], %[[arg1]]]
//       CHECK:   return %[[r0]], %[[r1]], %[[aff]], %[[arg2]]