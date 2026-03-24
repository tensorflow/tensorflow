// RUN: ifrt-opt %s -ifrt-merge-copies-and-reshards | FileCheck %s

#sharding = #ifrt.sharding_param<2 to [0] on 2>
!array0 = !ifrt.array<tensor<2xi32>, #sharding, [0,1]>
!array1 = !ifrt.array<tensor<2xi32>, #sharding, [2,3]>
!array2 = !ifrt.array<tensor<2xi32>, #sharding, [4,5]>
!array3 = !ifrt.array<tensor<2xi32>, #sharding, [6,7]>
!array_default_mk = !ifrt.array<tensor<2xi32>, #sharding, [0,1], memory_kind = "(default)">
!array_pinned_host_mk = !ifrt.array<tensor<2xi32>, #sharding, [0,1], memory_kind = "pinned_host">
!array_minor_to_major = !ifrt.array<tensor<2xi32>, #sharding, [0,1], layout = "{0,1}">
!array_major_to_minor = !ifrt.array<tensor<2xi32>, #sharding, [0,1], layout = "{1,0}">


// CHECK-LABEL: @merge_copies_of_call_results
func.func @merge_copies_of_call_results(%arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[CALL:.*]]:2, %{{.*}} = ifrt.Call @identity2(%arg0, %arg1)
// CHECK-NEXT: %[[MERGED:.*]]:2, %{{.*}} = ifrt.CopyArrays(%[[CALL]]#0, %[[CALL]]#1)
// CHECK-NEXT: return %[[MERGED]]#0, %[[MERGED]]#1
  %0:2, %ctrl_0 = ifrt.Call @identity2(%arg0, %arg1) on devices [0,1]
      : (!array0, !array0) -> (!array0, !array0)
  %1, %ctrl_1 = ifrt.CopyArrays(%0#0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.CopyArrays(%0#1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

// CHECK-LABEL: @merge_copies_of_func_args
func.func @merge_copies_of_func_args(%arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1)
// CHECK-NEXT: return %[[MERGED]]#0, %[[MERGED]]#1
  %1, %ctrl_1 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.CopyArrays(%arg1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

// CHECK-LABEL: @copies_into_return_op_are_grouped_by_destination
func.func @copies_into_return_op_are_grouped_by_destination(
    %arg0: !array0, %arg1: !array0, %arg2: !array0)
  -> (!array1, !array1, !array2) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY0:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1)
  // CHECK-NEXT: %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%arg2)
  // CHECK-NEXT: return %[[COPY0]]#0, %[[COPY0]]#1, %[[COPY1]]
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %1, %ctrl_1 = ifrt.CopyArrays(%arg1) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.CopyArrays(%arg2) : (!array0) -> !array2
  return %0, %1, %2 : !array1, !array1, !array2
}

// CHECK-LABEL: @merge_copies_for_same_devices_only
func.func @merge_copies_for_same_devices_only(
    %arg0: !array0, %arg1: !array0, %arg2: !array0, %arg3: !array0, %arg4: !array1, %arg5: !array1)
  -> (!array1, !array1, !array0, !array0, !array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED1:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1)
// CHECK-NEXT: %[[MERGED2:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg2, %arg3)
// CHECK-NEXT: %[[MERGED3:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg4, %arg5)
// CHECK-NEXT: return %[[MERGED1]]#0, %[[MERGED1]]#1, %[[MERGED2]]#0, %[[MERGED2]]#1, %[[MERGED3]]#0, %[[MERGED3]]#1
  %1, %ctrl_1 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.CopyArrays(%arg1) : (!array0) -> !array1
  %3, %ctrl_3 = ifrt.CopyArrays(%arg2) : (!array0) -> !array0
  %4, %ctrl_4 = ifrt.CopyArrays(%arg3) : (!array0) -> !array0
  %5, %ctrl_5 = ifrt.CopyArrays(%arg4) : (!array1) -> !array1
  %6, %ctrl_6 = ifrt.CopyArrays(%arg5) : (!array1) -> !array1
  return %1, %2, %3, %4, %5, %6 : !array1, !array1, !array0, !array0, !array1, !array1
}

// CHECK-LABEL: @merge_reshards_for_same_devices_only
func.func @merge_reshards_for_same_devices_only(
    %arg0: !array0, %arg1: !array0, %arg2: !array0, %arg3: !array0, %arg4: !array1, %arg5: !array1)
  -> (!array1, !array1, !array0, !array0, !array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED1:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
// CHECK-NEXT: %[[MERGED2:.*]]:2, %{{.*}} = ifrt.Reshard(%arg2, %arg3)
// CHECK-NEXT: %[[MERGED3:.*]]:2, %{{.*}} = ifrt.Reshard(%arg4, %arg5)
// CHECK-NEXT: return %[[MERGED1]]#0, %[[MERGED1]]#1, %[[MERGED2]]#0, %[[MERGED2]]#1, %[[MERGED3]]#0, %[[MERGED3]]#1
  %1, %ctrl_1 = ifrt.Reshard(%arg0) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%arg1) : (!array0) -> !array1
  %3, %ctrl_3 = ifrt.Reshard(%arg2) : (!array0) -> !array0
  %4, %ctrl_4 = ifrt.Reshard(%arg3) : (!array0) -> !array0
  %5, %ctrl_5 = ifrt.Reshard(%arg4) : (!array1) -> !array1
  %6, %ctrl_6 = ifrt.Reshard(%arg5) : (!array1) -> !array1
  return %1, %2, %3, %4, %5, %6 : !array1, !array1, !array0, !array0, !array1, !array1
}


// CHECK-LABEL: @merge_copies_for_same_donated_only
func.func @merge_copies_for_same_donated_only(
    %arg0: !array0, %arg1: !array0, %arg2: !array0, %arg3: !array0)
  -> (!array1, !array1, !array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[MERGED1:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1) {donated = true} :
// CHECK-NEXT: %[[MERGED2:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg2, %arg3)
// CHECK-NOT:      {donated = true}
// CHECK-NEXT: return %[[MERGED1]]#0, %[[MERGED1]]#1, %[[MERGED2]]#0, %[[MERGED2]]#1
  %1, %ctrl_1 = ifrt.CopyArrays(%arg0) {donated = true} : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.CopyArrays(%arg1) {donated = true} : (!array0) -> !array1
  %3, %ctrl_3 = ifrt.CopyArrays(%arg2) {donated = false} : (!array0) -> !array1
  %4, %ctrl_4 = ifrt.CopyArrays(%arg3) : (!array0) -> !array1
  return %1, %2, %3, %4 : !array1, !array1, !array1, !array1
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

// CHECK-LABEL: @dont_merge_copies_if_any_control_dependencies
func.func @dont_merge_copies_if_any_control_dependencies(
    %arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[CALL:.*]]:2, %[[CTRL:.*]] = ifrt.Call @identity2(%arg0, %arg1)
// CHECK-NEXT: %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[CALL]]#0) after %[[CTRL]]
// CHECK-NEXT: %[[COPY2:.*]], %{{.*}} = ifrt.CopyArrays(%[[CALL]]#1)
// CHECK-NEXT: return %[[COPY1]], %[[COPY2]]
  %0:2, %ctrl_0 = ifrt.Call @identity2(%arg0, %arg1) on devices [0,1]
      : (!array0, !array0) -> (!array0, !array0)
  %1, %ctrl_1 = ifrt.CopyArrays(%0#0) after %ctrl_0 : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.CopyArrays(%0#1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

// CHECK-LABEL: @dont_merge_reshards_if_any_control_dependencies
func.func @dont_merge_reshards_if_any_control_dependencies(
    %arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
// CHECK-NEXT: %[[CALL:.*]]:2, %[[CTRL:.*]] = ifrt.Call @identity2(%arg0, %arg1)
// CHECK-NEXT: %[[R1:.*]], %{{.*}} = ifrt.Reshard(%[[CALL]]#0) after %[[CTRL]]
// CHECK-NEXT: %[[R2:.*]], %{{.*}} = ifrt.Reshard(%[[CALL]]#1)
// CHECK-NEXT: return %[[R1]], %[[R2]]
  %0:2, %ctrl_0 = ifrt.Call @identity2(%arg0, %arg1) on devices [0,1]
      : (!array0, !array0) -> (!array0, !array0)
  %1, %ctrl_1 = ifrt.Reshard(%0#0) after %ctrl_0 : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Reshard(%0#1) : (!array0) -> !array1
  return %1, %2 : !array1, !array1
}

// CHECK-LABEL: @merge_copies_interleaved_with_calls
func.func @merge_copies_interleaved_with_calls(
    %arg0: !array0, %arg1: !array0)
  -> (!array0, !array1, !array2, !array3) attributes {ifrt.function} {
  // CHECK-NEXT: %[[CALL0:.*]]:2, %{{.*}} = ifrt.Call
  // CHECK-NEXT: %[[COPY0:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %[[CALL0]]#0)
  // CHECK-NEXT: %[[CALL1:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[COPY0]]#1, %[[COPY0]]#0)
  // CHECK-NEXT: %[[COPY1:.*]]:2, %{{.*}} = ifrt.CopyArrays(%[[COPY0]]#0, %[[CALL1]]#0)
  // CHECK-NEXT: %[[CALL2:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[COPY1]]#1, %[[COPY1]]#0)
  // CHECK-NEXT: %[[COPY2:.*]]:2, %{{.*}} = ifrt.CopyArrays(%[[COPY1]]#0, %[[CALL2]]#0)
  // CHECK-NEXT: %[[CALL3:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[COPY2]]#1, %[[COPY2]]#0)
  // CHECK-NEXT: return %[[CALL0]]#1, %[[CALL1]]#1, %[[CALL2]]#1, %[[CALL3]]#1
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array1) -> !array2
  %2, %ctrl_2 = ifrt.CopyArrays(%1) : (!array2) -> !array3
  %3:2, %ctrl_3 = ifrt.Call @identity2(%arg0, %arg0) on devices [0,1] : (!array0, !array0) -> (!array0, !array0)
  %4, %ctrl_4 = ifrt.CopyArrays(%3#0) : (!array0) -> !array1
  %5:2, %ctrl_5 = ifrt.Call @identity2(%4, %0) on devices [2,3] : (!array1, !array1) -> (!array1, !array1)
  %6, %ctrl_6 = ifrt.CopyArrays(%5#0) : (!array1) -> !array2
  %7:2, %ctrl_7 = ifrt.Call @identity2(%6, %1) on devices [4,5] : (!array2, !array2) -> (!array2, !array2)
  %8, %ctrl_8 = ifrt.CopyArrays(%7#0) : (!array2) -> !array3
  %9:2, %ctrl_9 = ifrt.Call @identity2(%8, %2) on devices [6,7] : (!array3, !array3) -> (!array3, !array3)
  func.return %3#1, %5#1, %7#1, %9#1 : !array0, !array1, !array2, !array3
}

// CHECK-LABEL: @merge_parallel_copies
func.func @merge_parallel_copies(
    %arg0: !array0, %arg1: !array0)
  -> (!array3, !array3) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY0:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1)
  // CHECK-NEXT: %[[COPY1:.*]]:2, %{{.*}} = ifrt.CopyArrays(%[[COPY0]]#0, %[[COPY0]]#1)
  // CHECK-NEXT: %[[COPY2:.*]]:2, %{{.*}} = ifrt.CopyArrays(%[[COPY1]]#0, %[[COPY1]]#1)
  // CHECK-NEXT: %[[CALL0:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[COPY2]]#1, %[[COPY2]]#1)
  // CHECK-NEXT: return %[[COPY2]]#0, %[[CALL0]]#0
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array1) -> !array2
  %2, %ctrl_2 = ifrt.CopyArrays(%1) : (!array2) -> !array3
  %3, %ctrl_3 = ifrt.CopyArrays(%arg1) : (!array0) -> !array1
  %4, %ctrl_4 = ifrt.CopyArrays(%3) : (!array1) -> !array2
  %5, %ctrl_5 = ifrt.CopyArrays(%4) : (!array2) -> !array3
  %6:2, %ctrl_6 = ifrt.Call @identity2(%5, %5) on devices [6,7] : (!array3, !array3) -> (!array3, !array3)
  func.return %2, %6#0 : !array3, !array3
}

// CHECK-LABEL: @copies_after_first_group_user_are_not_merged
func.func @copies_after_first_group_user_are_not_merged(
    %arg0: !array0, %arg1: !array0, %arg2: !array0)
  -> (!array1, !array1, !array1, !array1) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY0:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1)
  // CHECK-NEXT: %[[C0:.*]], %{{.*}} = ifrt.Call @identity1(%[[COPY0]]#1)
  // CHECK-NEXT: %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%arg2)
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %1, %ctrl_1 = ifrt.CopyArrays(%arg1) : (!array0) -> !array1
  %2, %ctrl_2 = ifrt.Call @identity1(%1) on devices [2,3] : (!array1) -> (!array1)
  %3, %ctrl_3 = ifrt.CopyArrays(%arg2) : (!array0) -> !array1
  return %0, %1, %2, %3 : !array1, !array1, !array1, !array1
}

// CHECK-LABEL: @merge_copies_with_default_and_no_memory_kind
func.func @merge_copies_with_default_and_no_memory_kind(
    %arg0: !array_default_mk, %arg1: !array0)
  -> (!array_default_mk, !array0) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY1:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1)
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array_default_mk) -> !array_default_mk
  %1, %ctrl_1 = ifrt.CopyArrays(%arg1) : (!array0) -> !array0
  return %0, %1 : !array_default_mk, !array0
}

// CHECK-LABEL: @dont_merge_copies_with_different_memory_kinds
func.func @dont_merge_copies_with_different_memory_kinds(
    %arg0: !array_default_mk, %arg1: !array_pinned_host_mk)
  -> (!array_default_mk, !array_pinned_host_mk) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%arg0)
  // CHECK-NEXT: %[[COPY2:.*]], %{{.*}} = ifrt.CopyArrays(%arg1)
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array_default_mk) -> !array_default_mk
  %1, %ctrl_1 = ifrt.CopyArrays(%arg1) : (!array_pinned_host_mk) -> !array_pinned_host_mk
  return %0, %1 : !array_default_mk, !array_pinned_host_mk
}

// CHECK-LABEL: @merge_reshards_with_different_memory_kinds
func.func @merge_reshards_with_different_memory_kinds(
    %arg0: !array_default_mk, %arg1: !array_pinned_host_mk)
  -> (!array_default_mk, !array_pinned_host_mk) attributes {ifrt.function} {
  // CHECK-NEXT: %[[R0:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
  // CHECK-NEXT: return %[[R0]]#0, %[[R0]]#1
  %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array_default_mk) -> !array_default_mk
  %1, %ctrl_1 = ifrt.Reshard(%arg1) : (!array_pinned_host_mk) -> !array_pinned_host_mk
  return %0, %1 : !array_default_mk, !array_pinned_host_mk
}

// CHECK-LABEL: @dont_merge_copies_with_different_layouts
func.func @dont_merge_copies_with_different_layouts(
    %arg0: !array_minor_to_major, %arg1: !array_major_to_minor)
  -> (!array_minor_to_major, !array_major_to_minor) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%arg0)
  // CHECK-NEXT: %[[COPY2:.*]], %{{.*}} = ifrt.CopyArrays(%arg1)
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array_minor_to_major) -> !array_minor_to_major
  %1, %ctrl_1 = ifrt.CopyArrays(%arg1) : (!array_major_to_minor) -> !array_major_to_minor
  return %0, %1 : !array_minor_to_major, !array_major_to_minor
}

// CHECK-LABEL: @merge_reshards_with_different_layouts
func.func @merge_reshards_with_different_layouts(
    %arg0: !array_minor_to_major, %arg1: !array_major_to_minor)
  -> (!array_minor_to_major, !array_major_to_minor) attributes {ifrt.function} {
  // CHECK-NEXT: %[[R0:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %arg1)
  // CHECK-NEXT: return %[[R0]]#0, %[[R0]]#1
  %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array_minor_to_major) -> !array_minor_to_major
  %1, %ctrl_1 = ifrt.Reshard(%arg1) : (!array_major_to_minor) -> !array_major_to_minor
  return %0, %1 : !array_minor_to_major, !array_major_to_minor
}

// CHECK-LABEL: @chain_of_copies_is_sunk
func.func @chain_of_copies_is_sunk(%arg0: !array0, %arg1: !array0)
  -> (!array2, !array2) attributes {ifrt.function} {
  // CHECK-NEXT: %[[C0:.*]]:2, %{{.*}} = ifrt.Call @identity2(%arg0, %arg1) on devices [0, 1]
  // CHECK-NEXT: %[[COPY0:.*]]:2, %{{.*}} = ifrt.CopyArrays(%arg0, %[[C0]]#0)
  // CHECK-NEXT: %[[C1:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[COPY0]]#0, %[[COPY0]]#1) on devices [2, 3]
  // CHECK-NEXT: %[[COPY1:.*]]:2, %{{.*}} = ifrt.CopyArrays(%[[COPY0]]#0, %[[C1]]#0)
  // CHECK-NEXT: %[[C2:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[COPY1]]#0, %[[COPY1]]#1) on devices [4, 5]
  // CHECK-NEXT: return %[[C2]]#0, %[[C2]]#1
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %1, %ctrl_1 = ifrt.CopyArrays(%0) : (!array1) -> !array2
  %2:2, %ctrl_2 = ifrt.Call @identity2(%arg0, %arg1) on devices [0,1] : (!array0, !array0) -> (!array0, !array0)
  %3, %ctrl_3 = ifrt.CopyArrays(%2#0) : (!array0) -> !array1
  %4:2, %ctrl_4 = ifrt.Call @identity2(%0, %3) on devices [2,3] : (!array1, !array1) -> (!array1, !array1)
  %5, %ctrl_5 = ifrt.CopyArrays(%4#0) : (!array1) -> !array2
  %6:2, %ctrl_6 = ifrt.Call @identity2(%1, %5) on devices [4,5] : (!array2, !array2) -> (!array2, !array2)
  return %6#0, %6#1 : !array2, !array2
}

// CHECK-LABEL: @chain_of_reshards_is_sunk
func.func @chain_of_reshards_is_sunk(%arg0: !array0, %arg1: !array0)
  -> (!array2, !array2) attributes {ifrt.function} {
  // CHECK-NEXT: %[[C0:.*]]:2, %{{.*}} = ifrt.Call @identity2(%arg0, %arg1) on devices [0, 1]
  // CHECK-NEXT: %[[R0:.*]]:2, %{{.*}} = ifrt.Reshard(%arg0, %[[C0]]#0)
  // CHECK-NEXT: %[[C1:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[R0]]#0, %[[R0]]#1) on devices [2, 3]
  // CHECK-NEXT: %[[R1:.*]]:2, %{{.*}} = ifrt.Reshard(%[[R0]]#0, %[[C1]]#0)
  // CHECK-NEXT: %[[C2:.*]]:2, %{{.*}} = ifrt.Call @identity2(%[[R1]]#0, %[[R1]]#1) on devices [4, 5]
  // CHECK-NEXT: return %[[C2]]#0, %[[C2]]#1
  %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
  %1, %ctrl_1 = ifrt.Reshard(%0) : (!array1) -> !array2
  %2:2, %ctrl_2 = ifrt.Call @identity2(%arg0, %arg1) on devices [0,1] : (!array0, !array0) -> (!array0, !array0)
  %3, %ctrl_3 = ifrt.Reshard(%2#0) : (!array0) -> !array1
  %4:2, %ctrl_4 = ifrt.Call @identity2(%0, %3) on devices [2,3] : (!array1, !array1) -> (!array1, !array1)
  %5, %ctrl_5 = ifrt.Reshard(%4#0) : (!array1) -> !array2
  %6:2, %ctrl_6 = ifrt.Call @identity2(%1, %5) on devices [4,5] : (!array2, !array2) -> (!array2, !array2)
  return %6#0, %6#1 : !array2, !array2
}

// CHECK-LABEL: @dont_merge_copies_and_reshards
func.func @dont_merge_copies_and_reshards(%arg0: !array0, %arg1: !array0)
  -> (!array1, !array1) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%arg0)
  // CHECK-NEXT: %[[R0:.*]], %{{.*}} = ifrt.Reshard(%arg1)
  // CHECK-NEXT: return %[[COPY0]], %[[R0]]
  %0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
  %1, %crtl_1 = ifrt.Reshard(%arg1) : (!array0) -> !array1
  return %0, %1 : !array1, !array1
}

// CHECK-LABEL: @merge_copies_with_multiple_inputs
func.func @merge_copies_with_multiple_inputs(
    %arg0: !array0, %arg1: !array0, %arg2: !array0, %arg3: !array0)
  -> (!array1, !array1, !array1, !array1) attributes {ifrt.function} {
  // CHECK-NEXT: %[[COPY0:.*]]:4, %{{.*}} = ifrt.CopyArrays(%arg0, %arg1, %arg2, %arg3)
  // CHECK-NEXT: return %[[COPY0]]#0, %[[COPY0]]#1, %[[COPY0]]#2
  %0, %1, %ctrl_0 = ifrt.CopyArrays(%arg0, %arg1) : (!array0, !array0) -> (!array1, !array1)
  %2, %3, %ctrl_1 = ifrt.CopyArrays(%arg2, %arg3) : (!array0, !array0) -> (!array1, !array1)
  return %0, %1, %2, %3 : !array1, !array1, !array1, !array1
}

// CHECK-LABEL: @dont_merge_reshards_on_different_devices
func.func @dont_merge_reshards_on_different_devices(
  %arg0: !array0, %arg1: !array1) -> (!array1, !array2) attributes {ifrt.function} {
  // CHECK-NEXT: %[[R0:.*]], %{{.*}} = ifrt.Reshard(%arg0)
  // CHECK-NEXT: %[[R1:.*]], %{{.*}} = ifrt.Reshard(%arg1)
  // CHECK-NEXT: return %[[R0]], %[[R1]]
  %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> (!array1)
  %1, %ctrl_1 = ifrt.Reshard(%arg1) : (!array1) -> (!array2)
  return %0, %1 : !array1, !array2
}

func.func @identity1(%arg0: tensor<2xi32>) -> (tensor<2xi32>) {
  return %arg0 : tensor<2xi32>
}

func.func @identity2(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
  -> (tensor<2xi32>, tensor<2xi32>) {
  return %arg0, %arg1 : tensor<2xi32>, tensor<2xi32>
}
