// RUN: ifrt-opt %s -ifrt-insert-copy-arrays-for-returned-many-times -split-input-file | FileCheck %s

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @non_donated_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[COPY:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK:        return %[[COPY]] : !ifrt.array<
func.func @non_donated_return(%arg0: !array) -> !array attributes {ifrt.function} {
  return %arg0: !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @donated_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[COPY:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) {donated = true}
// CHECK:        return %[[COPY]] : !ifrt.array<
func.func @donated_return(%arg0: !array {ifrt.donated}) -> !array attributes {ifrt.function} {
  return %arg0: !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @duplicate_non_donated_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK-DAG:    %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK-DAG:    %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK:        return %[[COPY0]], %[[COPY1]] : !ifrt.array<
func.func @duplicate_non_donated_return(%arg0: !array) -> (!array, !array) attributes {ifrt.function} {
  return %arg0, %arg0: !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @donated_triplicate_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK-DAG:    %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK-DAG:    %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK-DAG:    %[[COPY2:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) {donated = true}
// CHECK:        return %[[COPY0]], %[[COPY1]], %[[COPY2]] : !ifrt.array<
func.func @donated_triplicate_return(%arg0: !array {ifrt.donated}) -> (!array, !array, !array) attributes {ifrt.function} {
  return %arg0, %arg0, %arg0: !array, !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @returned_non_block_arg(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[COPY:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]])
// CHECK-NOT:    ifrt.CopyArrays
// CHECK:        return %[[COPY]]
func.func @returned_non_block_arg(%arg0: !array) -> !array attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array) -> !array
  return %0: !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

func.func private @foo(%arg0: !array) -> !array

// CHECK-LABEL: func.func @returned_input_used_by_other_op(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[CALL:.*]], %{{.*}} = ifrt.Call @foo(%[[ARG0]]) on devices [0]
// CHECK:        %[[COPY:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK:        return %[[COPY]], %[[CALL]]
func.func @returned_input_used_by_other_op(%arg0: !array) -> (!array, !array) attributes {ifrt.function} {
  %0, %ctrl = ifrt.Call @foo(%arg0) on devices [0] : (!array) -> !array
  return %arg0, %0: !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @returned_op_output_multiple_times(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]])
// CHECK:        %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[COPY0]]) :
// CHECK:        return %[[COPY1]], %[[COPY0]]
func.func @returned_op_output_multiple_times(%arg0: !array) -> (!array, !array) attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array) -> !array
  return %0, %0: !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @donated_duplicate_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK-DAG:    %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK-DAG:    %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) {donated = true}
// CHECK:        return %[[COPY0]], %[[COPY1]] : !ifrt.array<
func.func @donated_duplicate_return(%arg0: !array {ifrt.donated}) -> (!array, !array) attributes {ifrt.function} {
  return %arg0, %arg0: !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @multiple_arrays(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<{{.*}}>, %[[ARG1:.*]]: !ifrt.array<
// CHECK-DAG:    %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK-DAG:    %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) :
// CHECK-DAG:    %[[COPY2:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG1]]) :
// CHECK-DAG:    %[[COPY3:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG1]]) :
// CHECK:        return %[[COPY0]], %[[COPY1]], %[[COPY2]], %[[COPY3]] : !ifrt.array<
func.func @multiple_arrays(%arg0: !array, %arg1: !array) -> (!array, !array, !array, !array) attributes {ifrt.function} {
  return %arg0, %arg0, %arg1, %arg1: !array, !array, !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @ignore_non_ifrt_function(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK-NOT:    ifrt.CopyArrays
// CHECK:        return %[[ARG0]], %[[ARG0]] : !ifrt.array<
func.func @ignore_non_ifrt_function(%arg0: !array) -> (!array, !array) {
  return %arg0, %arg0: !array, !array
}
