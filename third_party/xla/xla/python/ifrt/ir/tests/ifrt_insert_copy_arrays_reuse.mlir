// RUN: ifrt-opt %s -ifrt-insert-copy-arrays-reuse -split-input-file | FileCheck %s

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @single_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK-NOT:    ifrt.CopyArrays
// CHECK:        return %[[ARG0]] : !ifrt.array<
func.func @single_return(%arg0: !array) -> !array attributes {ifrt.function} {
  return %arg0: !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @duplicate_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[COPY:.*]], %[[CTRL:.*]] = ifrt.CopyArrays(%[[ARG0]]) {reuse = true}
// CHECK:        return %[[ARG0]], %[[COPY]] : !ifrt.array<
func.func @duplicate_return(%arg0: !array) -> (!array, !array) attributes {ifrt.function} {
  return %arg0, %arg0: !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @triplicate_return(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK-DAG:    %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) {reuse = true}
// CHECK-DAG:    %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) {reuse = true}
// CHECK:        return %[[ARG0]], %[[COPY0]], %[[COPY1]] : !ifrt.array<
func.func @triplicate_return(%arg0: !array) -> (!array, !array, !array) attributes {ifrt.function} {
  return %arg0, %arg0, %arg0: !array, !array, !array
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @multiple_arrays(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<{{.*}}>, %[[ARG1:.*]]: !ifrt.array<
// CHECK-DAG:    %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) {reuse = true}
// CHECK-DAG:    %[[COPY1:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG1]]) {reuse = true}
// CHECK:        return %[[ARG0]], %[[COPY0]], %[[ARG1]], %[[COPY1]] : !ifrt.array<
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

// -----

!array0 = !ifrt.array<tensor<2x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x4xi32>, #ifrt.sharding_param<1x2 to [0] on 2>, [2,3]>

// CHECK-LABEL: func.func @with_ifrt_ops(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[COPY0:.*]], %{{.*}} = ifrt.CopyArrays(%[[ARG0]]) : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
// CHECK:        %[[REUSE:.*]], %{{.*}} = ifrt.CopyArrays(%[[COPY0]]) {reuse = true} : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
// CHECK:        return %[[COPY0]], %[[REUSE]] : !ifrt.array<
func.func @with_ifrt_ops(%arg0: !array0) -> (!array1, !array1) attributes {ifrt.function} {
  %res, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> (!array1)
  return %res, %res : !array1, !array1
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>

// CHECK-LABEL: func.func @with_ifrt_call(
// CHECK-SAME:   %[[ARG0:.*]]: !ifrt.array<
// CHECK:        %[[CALL:.*]], %[[CTRL:.*]] = ifrt.Call @add_one(%[[ARG0]]) on devices [0] : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
// CHECK:        %[[COPY:.*]], %{{.*}} = ifrt.CopyArrays(%[[CALL]]) {reuse = true} : (!ifrt.array<{{.*}}>) -> !ifrt.array<{{.*}}>
// CHECK:        return %[[CALL]], %[[COPY]] : !ifrt.array<
func.func @with_ifrt_call(%arg0: !array) -> (!array, !array) attributes {ifrt.function} {
  %res, %ctrl = ifrt.Call @add_one(%arg0) on devices [0] : (!array) -> (!array)
  return %res, %res : !array, !array
}

func.func @add_one(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  return %arg0 : tensor<2xi32>
}