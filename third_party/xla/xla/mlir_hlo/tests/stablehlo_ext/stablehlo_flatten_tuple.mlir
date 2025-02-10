// RUN: mlir-hlo-opt -split-input-file -stablehlo-ext-flatten-tuple %s | FileCheck %s

// CHECK-LABEL: @custom_call
// CHECK-SAME: %[[X:.*]]: tensor<6x3xf32>
func.func @custom_call(%x: tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) {
  // CHECK: %[[CALL:.+]]:2 = stablehlo.custom_call @f(%[[X]]) {api_version = 2 : i32} : (tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>)
  %0 = "stablehlo.custom_call"(%x) {api_version = 2 : i32, call_target_name = "f"}
    : (tensor<6x3xf32>) -> tuple<tensor<6xf32>, tensor<3xf32>>
  %1 = "stablehlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<6xf32>
  %2 = "stablehlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<3xf32>
  return %1, %2 : tensor<6xf32>, tensor<3xf32>
}

// -----

// CHECK-LABEL: @custom_call_tupled_operand
func.func @custom_call_tupled_operand(%arg0: tuple<tensor<ui32>, tensor<i32>>)
  -> (tensor<i32>, tensor<ui32>) {
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<1> : tensor<ui32>
  %0 = stablehlo.constant dense<1> : tensor<ui32>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<10> : tensor<i32>
  %1 = stablehlo.constant dense<10> : tensor<i32>
  // CHECK-NEXT: %[[TUPLE:.*]] = stablehlo.tuple %[[C0]], %[[C1]], %arg
  %2 = stablehlo.tuple %0, %1, %arg0 : tuple<tensor<ui32>, tensor<i32>,
                                       tuple<tensor<ui32>, tensor<i32>>>
  // CHECK-NEXT: %[[VAR1:.*]] = stablehlo.get_tuple_element %[[TUPLE]][0]
  // CHECK-NEXT: %[[VAR2:.*]] = stablehlo.get_tuple_element %[[TUPLE]][1]
  // CHECK-NEXT: %[[VAR3:.*]] = stablehlo.get_tuple_element %[[TUPLE]][2]
  // CHECK-NEXT: %[[VAR4:.*]] = stablehlo.get_tuple_element %[[VAR3]][0]
  // CHECK-NEXT: %[[VAR5:.*]] = stablehlo.get_tuple_element %[[VAR3]][1]
  // CHECK-NEXT: stablehlo.custom_call @ScalarProgramDummyConstant(%[[VAR1]], %[[VAR2]], %[[VAR4]], %[[VAR5]])
  %3 = stablehlo.custom_call @ScalarProgramDummyConstant(%2)
    : (tuple<tensor<ui32>, tensor<i32>, tuple<tensor<ui32>, tensor<i32>>>)
    -> tensor<ui32>
  return %1, %3 : tensor<i32>, tensor<ui32>
}
