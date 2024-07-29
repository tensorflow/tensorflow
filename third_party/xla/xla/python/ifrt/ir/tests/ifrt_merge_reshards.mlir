// RUN: ifrt-opt %s -ifrt-merge-reshards | FileCheck %s

#sharding = #ifrt.sharding_param<2 to [0] on 2>
!array0 = !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
!array1 = !ifrt.array<tensor<2xi32>, #sharding, [2,3]>

// CHECK-LABEL: @merge_reshards_of_call_results
func.func @merge_reshards_of_call_results(%arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[CALL:.*]]:2, %{{.*}} = ifrt.Call @identity(%arg0, %arg1)
// CHECK-NEXT: %[[MERGED:.*]]:2, %{{.*}} = ifrt.Reshard(%[[CALL]]#0, %[[CALL]]#1)
// CHECK-NEXT: return %[[MERGED]]#0, %[[MERGED]]#1
  %0:2, %ctrl_0 = ifrt.Call @identity(%arg0, %arg1) on devices [0,1]
      : (!array0, !array0) -> (!array0, !array0)
  %1, %ctrl_1 = ifrt.Reshard(%0#0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%0#1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

// CHECK-LABEL: @merge_reshards_of_func_args
func.func @merge_reshards_of_func_args(%arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
// CHECK-NEXT: return %[[MERGED]]#0, %[[MERGED]]#1
  %1, %ctrl_1 = ifrt.Reshard(%arg0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%arg1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

// CHECK-LABEL: @merge_reshards_for_same_devices_only
func.func @merge_reshards_for_same_devices_only(
    %arg0: !array0, %arg1: !array0, %arg2: !array0, %arg3: !array0)
  -> (!array1, !array1, !array0, !array0) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED1:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
// CHECK-NEXT: %[[MERGED2:.*]]:2, %{{.*}} = ifrt.Reshard(%arg2, %arg3)
// CHECK-NEXT: return %[[MERGED1]]#0, %[[MERGED1]]#1, %[[MERGED2]]#0, %[[MERGED2]]#1
  %1, %ctrl_1 = ifrt.Reshard(%arg0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%arg1) : (!array0) -> !array1
  %3, %ctrl_3 = ifrt.Reshard(%arg2) : (!array0) -> !array0
  %4, %ctrl_4 = ifrt.Reshard(%arg3) : (!array0) -> !array0
  return %1, %2, %3, %4 : !array1, !array1, !array0, !array0
}

// CHECK-LABEL: @merge_reshards_for_same_donated_only
func.func @merge_reshards_for_same_donated_only(
    %arg0: !array0, %arg1: !array0, %arg2: !array0, %arg3: !array0)
  -> (!array1, !array1, !array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED1:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1) {donated = true} :
// CHECK-NEXT: %[[MERGED2:.*]]:2, %{{.*}} = ifrt.Reshard(%arg2, %arg3)
// CHECK-NOT:      {donated = true}
// CHECK-NEXT: return %[[MERGED1]]#0, %[[MERGED1]]#1, %[[MERGED2]]#0, %[[MERGED2]]#1
  %1, %ctrl_1 = ifrt.Reshard(%arg0) {donated = true} : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%arg1) {donated = true} : (!array0) -> !array1
  %3, %ctrl_3 = ifrt.Reshard(%arg2) {donated = false} : (!array0) -> !array1
  %4, %ctrl_4 = ifrt.Reshard(%arg3) : (!array0) -> !array1
  return %1, %2, %3, %4 : !array1, !array1, !array1, !array1
}

// CHECK-LABEL: @dont_merge_if_any_control_dependencies
func.func @dont_merge_if_any_control_dependencies(
    %arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[CALL:.*]]:2, %[[CTRL:.*]] = ifrt.Call @identity(%arg0, %arg1)
// CHECK-NEXT: %[[R1:.*]], %{{.*}} = ifrt.Reshard(%[[CALL]]#0) after %[[CTRL]]
// CHECK-NEXT: %[[R2:.*]], %{{.*}} = ifrt.Reshard(%[[CALL]]#1)
// CHECK-NEXT: return %[[R1]], %[[R2]]
  %0:2, %ctrl_0 = ifrt.Call @identity(%arg0, %arg1) on devices [0,1]
      : (!array0, !array0) -> (!array0, !array0)
  %1, %ctrl_1 = ifrt.Reshard(%0#0) after %ctrl_0 : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%0#1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

func.func private @identity(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
  -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
