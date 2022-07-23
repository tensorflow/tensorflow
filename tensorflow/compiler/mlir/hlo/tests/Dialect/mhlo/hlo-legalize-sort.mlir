// RUN: mlir-hlo-opt -hlo-legalize-sort %s | FileCheck %s

func.func @sort(%arg0 : tensor<2xi32>, %arg1 : tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %result:2 = "mhlo.sort"(%arg0, %arg1) ({
    ^bb0(%00: tensor<i32>, %01: tensor<i32>, %10: tensor<i32>, %11: tensor<i32>):
      %50 = tensor.extract %00[] : tensor<i32>
      %51 = tensor.extract %01[] : tensor<i32>
      %52 = arith.cmpi sgt, %50, %51 : i32
      %cmp_result = tensor.from_elements %52 : tensor<i1>
      "mhlo.return"(%cmp_result) : (tensor<i1>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<2xi32>, tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
  func.return %result#0, %result#1 : tensor<2xi32>, tensor<2xi32>
}

// CHECK-LABEL:   func @sort(
// CHECK-SAME:               %[[ARG0:.*]]: tensor<2xi32>,
// CHECK-SAME:               %[[ARG1:.*]]: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
// Iterate through dimension 0
// CHECK-DAG:       %[[C0_0:.*]] = arith.constant 0
// CHECK-DAG:       %[[C0_1:.*]] = arith.constant 0
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[ARG0]], %[[C0_1]] : tensor<2xi32>
// CHECK-DAG:       %[[C1_0:.*]] = arith.constant 1 : index
// Iterate through dimension 1
// CHECK:           %[[VAL_6:.*]]:2 = scf.for %[[VAL_7:.*]] = %[[C0_0]] to %[[VAL_4]] step %[[C1_0]] iter_args(%[[VAL_8:.*]] = %[[ARG0]], %[[VAL_9:.*]] = %[[ARG1]]) -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK-DAG:         %[[C0_2:.*]] = arith.constant 0
// CHECK-DAG:         %[[C1_1:.*]] = arith.constant 1
// CHECK-DAG:         %[[C2_0:.*]] = arith.constant 2
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[C2_0]], %[[C1_1]] : index
// Iterate through sorted dimension
// CHECK:             %[[VAL_14:.*]]:2 = scf.for %[[VAL_15:.*]] = %[[C0_2]] to %[[VAL_13]] step %[[C1_1]] iter_args(%[[VAL_16:.*]] = %[[VAL_8]], %[[VAL_17:.*]] = %[[VAL_9]]) -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Extract each value twice because we are comparing both directions and haven't run CSE yet
// CHECK:               %[[VAL_19:.*]] = tensor.extract %[[VAL_16]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:               %[[VAL_20:.*]] = tensor.from_elements %[[VAL_19]] : tensor<i32>
// CHECK:               %[[VAL_21:.*]] = tensor.extract %[[VAL_16]]{{\[}}%[[VAL_18]]] : tensor<2xi32>
// CHECK:               %[[VAL_22:.*]] = tensor.from_elements %[[VAL_21]] : tensor<i32>
// CHECK:               %[[VAL_23:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:               %[[VAL_24:.*]] = tensor.from_elements %[[VAL_23]] : tensor<i32>
// CHECK:               %[[VAL_25:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[VAL_18]]] : tensor<2xi32>
// CHECK:               %[[VAL_26:.*]] = tensor.from_elements %[[VAL_25]] : tensor<i32>
// CHECK:               %[[VAL_27:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_28:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_30:.*]] = tensor.from_elements %[[VAL_29]] : tensor<i1>
// CHECK:               %[[VAL_31:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_32:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_34:.*]] = tensor.from_elements %[[VAL_33]] : tensor<i1>
// Extract comparison results that were packed back into tensors by mhlo
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_34]][] : tensor<i1>
// CHECK:               %[[VAL_36:.*]] = tensor.extract %[[VAL_30]][] : tensor<i1>
// Determine if swapping should occur which happens only if NOT(CMP(A,B))  && CMP(B,A)
// CHECK:               %[[TRUE:.*]] = arith.constant true
// CHECK:               %[[VAL_38:.*]] = arith.xori %[[VAL_35]], %[[TRUE]] : i1
// CHECK:               %[[VAL_39:.*]] = arith.andi %[[VAL_38]], %[[VAL_36]] : i1
// CHECK:               %[[VAL_40:.*]]:2 = scf.if %[[VAL_39]] -> (tensor<2xi32>, tensor<2xi32>) {
// CHECK:                 %[[VAL_41:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Swap first pair of values
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:                 %[[VAL_43:.*]] = tensor.insert %[[VAL_42]] into %[[VAL_16]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:                 %[[VAL_45:.*]] = tensor.insert %[[VAL_44]] into %[[VAL_43]]{{\[}}%[[VAL_41]]] : tensor<2xi32>
// Swap second pair of values
// CHECK:                 %[[VAL_46:.*]] = tensor.extract %[[VAL_26]][] : tensor<i32>
// CHECK:                 %[[VAL_47:.*]] = tensor.insert %[[VAL_46]] into %[[VAL_17]]{{\[}}%[[VAL_15]]] : tensor<2xi32>
// CHECK:                 %[[VAL_48:.*]] = tensor.extract %[[VAL_24]][] : tensor<i32>
// CHECK:                 %[[VAL_49:.*]] = tensor.insert %[[VAL_48]] into %[[VAL_47]]{{\[}}%[[VAL_41]]] : tensor<2xi32>
// CHECK:                 scf.yield %[[VAL_45]], %[[VAL_49]] : tensor<2xi32>, tensor<2xi32>
// CHECK:               } else {
// Don't swap
// CHECK:                 scf.yield %[[VAL_16]], %[[VAL_17]] : tensor<2xi32>, tensor<2xi32>
// CHECK:               }
// Propagate values back through the loops
// CHECK:               scf.yield %[[VAL_40:.*]]#0, %[[VAL_40]]#1 : tensor<2xi32>, tensor<2xi32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<2xi32>, tensor<2xi32>
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : tensor<2xi32>, tensor<2xi32>
// CHECK:         }

func.func @dyn_sort(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
  %result:2 = "mhlo.sort"(%arg0, %arg1) ({
    ^bb0(%00: tensor<i32>, %01: tensor<i32>, %10: tensor<i32>, %11: tensor<i32>):
      %50 = tensor.extract %00[] : tensor<i32>
      %51 = tensor.extract %01[] : tensor<i32>
      %52 = arith.cmpi sgt, %50, %51 : i32
      %cmp_result = tensor.from_elements %52 : tensor<i1>
      "mhlo.return"(%cmp_result) : (tensor<i1>) -> ()
    }) {dimension = 0 : i64, is_stable = true} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  func.return %result#0, %result#1 : tensor<?xi32>, tensor<?xi32>
}
// CHECK-LABEL:   func @dyn_sort(
// CHECK-SAME:               %[[ARG0:.*]]: tensor<?xi32>,
// CHECK-SAME:               %[[ARG1:.*]]: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
// Iterate through dimension 0
// CHECK-DAG:       %[[C0_0:.*]] = arith.constant 0
// CHECK-DAG:       %[[C0_1:.*]] = arith.constant 0
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[ARG0]], %[[C0_1]] : tensor<?xi32>
// CHECK-DAG:       %[[C1_0:.*]] = arith.constant 1 : index
// Iterate through dimension 1
// CHECK:           %[[VAL_6:.*]]:2 = scf.for %[[VAL_7:.*]] = %[[C0_0]] to %[[VAL_4]] step %[[C1_0]] iter_args(%[[VAL_8:.*]] = %[[ARG0]], %[[VAL_9:.*]] = %[[ARG1]]) -> (tensor<?xi32>, tensor<?xi32>) {
// CHECK-DAG:         %[[C0_2:.*]] = arith.constant 0
// CHECK-DAG:         %[[C1_1:.*]] = arith.constant 1
// CHECK-DAG:         %[[C0_3:.*]] = arith.constant 0
// CHECK:             %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0_3]] : tensor<?xi32>
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[DIM]], %[[C1_1]] : index
// Iterate through sorted dimension
// CHECK:             %[[VAL_14:.*]]:2 = scf.for %[[VAL_15:.*]] = %[[C0_2]] to %[[VAL_13]] step %[[C1_1]] iter_args(%[[VAL_16:.*]] = %[[VAL_8]], %[[VAL_17:.*]] = %[[VAL_9]]) -> (tensor<?xi32>, tensor<?xi32>) {
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Extract each value twice because we are comparing both directions and haven't run CSE yet
// CHECK:               %[[VAL_19:.*]] = tensor.extract %[[VAL_16]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:               %[[VAL_20:.*]] = tensor.from_elements %[[VAL_19]] : tensor<i32>
// CHECK:               %[[VAL_21:.*]] = tensor.extract %[[VAL_16]]{{\[}}%[[VAL_18]]] : tensor<?xi32>
// CHECK:               %[[VAL_22:.*]] = tensor.from_elements %[[VAL_21]] : tensor<i32>
// CHECK:               %[[VAL_23:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:               %[[VAL_24:.*]] = tensor.from_elements %[[VAL_23]] : tensor<i32>
// CHECK:               %[[VAL_25:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[VAL_18]]] : tensor<?xi32>
// CHECK:               %[[VAL_26:.*]] = tensor.from_elements %[[VAL_25]] : tensor<i32>
// CHECK:               %[[VAL_27:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_28:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_30:.*]] = tensor.from_elements %[[VAL_29]] : tensor<i1>
// CHECK:               %[[VAL_31:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:               %[[VAL_32:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:               %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:               %[[VAL_34:.*]] = tensor.from_elements %[[VAL_33]] : tensor<i1>
// Extract comparison results that were packed back into tensors by mhlo
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_34]][] : tensor<i1>
// CHECK:               %[[VAL_36:.*]] = tensor.extract %[[VAL_30]][] : tensor<i1>
// Determine if swapping should occur which happens only if NOT(CMP(A,B))  && CMP(B,A)
// CHECK:               %[[TRUE:.*]] = arith.constant true
// CHECK:               %[[VAL_38:.*]] = arith.xori %[[VAL_35]], %[[TRUE]] : i1
// CHECK:               %[[VAL_39:.*]] = arith.andi %[[VAL_38]], %[[VAL_36]] : i1
// CHECK:               %[[VAL_40:.*]]:2 = scf.if %[[VAL_39]] -> (tensor<?xi32>, tensor<?xi32>) {
// CHECK:                 %[[VAL_41:.*]] = arith.addi %[[VAL_15]], %[[C1_1]] : index
// Swap first pair of values
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_22]][] : tensor<i32>
// CHECK:                 %[[VAL_43:.*]] = tensor.insert %[[VAL_42]] into %[[VAL_16]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_20]][] : tensor<i32>
// CHECK:                 %[[VAL_45:.*]] = tensor.insert %[[VAL_44]] into %[[VAL_43]]{{\[}}%[[VAL_41]]] : tensor<?xi32>
// Swap second pair of values
// CHECK:                 %[[VAL_46:.*]] = tensor.extract %[[VAL_26]][] : tensor<i32>
// CHECK:                 %[[VAL_47:.*]] = tensor.insert %[[VAL_46]] into %[[VAL_17]]{{\[}}%[[VAL_15]]] : tensor<?xi32>
// CHECK:                 %[[VAL_48:.*]] = tensor.extract %[[VAL_24]][] : tensor<i32>
// CHECK:                 %[[VAL_49:.*]] = tensor.insert %[[VAL_48]] into %[[VAL_47]]{{\[}}%[[VAL_41]]] : tensor<?xi32>
// CHECK:                 scf.yield %[[VAL_45]], %[[VAL_49]] : tensor<?xi32>, tensor<?xi32>
// CHECK:               } else {
// Don't swap
// CHECK:                 scf.yield %[[VAL_16]], %[[VAL_17]] : tensor<?xi32>, tensor<?xi32>
// CHECK:               }
// Propagate values back through the loops
// CHECK:               scf.yield %[[VAL_40:.*]]#0, %[[VAL_40]]#1 : tensor<?xi32>, tensor<?xi32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<?xi32>, tensor<?xi32>
// CHECK:           }
// CHECK:           return %[[VAL_6:.*]]#0, %[[VAL_6]]#1 : tensor<?xi32>, tensor<?xi32>
// CHECK:         }