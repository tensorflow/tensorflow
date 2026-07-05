// RUN: ifrt-opt %s -ifrt-add-ctrl-dependencies | FileCheck %s

#sharding = #ifrt.sharding_param<2 to [0] on 2>
!array0 = !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
!array1 = !ifrt.array<tensor<2xi32>, #sharding, [2,3]>

// CHECK-LABEL: @add_ctrl_dependencies_between_call_ops_on_same_devices
func.func @add_ctrl_dependencies_between_call_ops_on_same_devices(%arg0: !array0, %arg1: !array1)
  -> (!array0, !array1) attributes {ifrt.function} {
  // CHECK-NEXT: %[[CALL_0:.*]], %[[CTRL_0:.*]] = ifrt.Call @identity1(%arg0) on devices [0, 1]
  // CHECK-NEXT: %[[COPY:.*]], %[[CTRL_1:.*]] = ifrt.CopyArrays(%[[CALL_0]])
  // CHECK-NEXT: %[[CALL_1:.*]]:2, %[[CTRL_2:.*]] = ifrt.Call @identity2(%arg1, %[[COPY]]) on devices [2, 3]
  // CHECK-NEXT: %[[CALL_3:.*]], %[[CTRL_3:.*]] = ifrt.Call @identity1(%arg0) after %[[CTRL_0]] on devices [0, 1]
  // CHECK-NEXT: return %[[CALL_3]], %[[CALL_1]]#1
  %0, %ctrl_0 = ifrt.Call @identity1(%arg0) on devices [0,1] : (!array0) -> (!array0)
  %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array0) -> !array1
  %2:2, %ctrl_2 = ifrt.Call @identity2(%arg1, %1) on devices [2,3] : (!array1, !array1) -> (!array1, !array1)
  %3, %ctrl_3 = ifrt.Call @identity1(%arg0) on devices [0,1] : (!array0) -> (!array0)
  return %3, %2#1 : !array0, !array1
}

func.func @identity1(%arg0: tensor<2xi32>) -> (tensor<2xi32>) {
  return %arg0 : tensor<2xi32>
}

func.func @identity2(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
  -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
