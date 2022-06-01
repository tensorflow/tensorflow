// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @case0
tfg.func @case0() -> (tensor<i32>) {
  %A, %ctl = A() : () -> (tensor<i32>)
  return(%A) : tensor<i32>
}

// CHECK: tfg.func @case1
tfg.func @case1(%arg0: tensor<f32> {tf._a}) -> () {
  %B, %ctl = B(%arg0) : (tensor<f32>) -> (tensor<i32>)
  return
}

// CHECK: tfg.func @case2
tfg.func @case2() -> () {
  %C, %ctl = C() : () -> (tensor<i32>)
  return
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[INDEX:.*]], %[[CTL:.*]] = Index
  %Index, %ctl = Index : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DATA:.*]], %[[CTL_0:.*]] = Data
  %Data, %ctl_0 = Data : () -> (tensor<f32>)
  // CHECK-NEXT: %[[CASE:.*]], %[[CTL_1:.*]] = CaseRegion %[[INDEX]]  {
  // CHECK-NEXT:   %[[A:.*]], %[[CTL_2:.*]] = A
  // CHECK-NEXT:   yield(%[[A]])
  // CHECK-NEXT: } {_some_attr = 1 : index, branch_attrs = [{}]
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  %Case0, %ctl_1 = Case(%Index) {
    _some_attr = 1 : index,
    Tin = [], Tout = [i32], output_shapes = [],
    branches = [#tf_type.func<@case0, {}>]
  } : (tensor<i32>) -> (tensor<i32>)

  // CHECK-NEXT: %[[CTL_2:.*]] = CaseRegion %[[INDEX]]  {
  // CHECK-NEXT:   %[[B:.*]], %[[CTL_4:.*]] = B(%[[DATA]])
  // CHECK-NEXT:   yield
  // CHECK-NEXT: } {_some_attr = 1 : index, branch_attrs = [{}]
  // CHECK-SAME: (tensor<i32>) -> ()
  %ctl_2 = Case(%Index, %Data) {
    _some_attr = 1 : index,
    Tin = [f32], Tout = [], output_shapes = [],
    branches = [#tf_type.func<@case1, {}>]
  } : (tensor<i32>, tensor<f32>) -> ()

  // CHECK-NEXT: %[[CTL_3:.*]] = CaseRegion %[[INDEX]]  {
  // CHECK-NEXT:   %[[C:.*]], %[[CTL_5:.*]] = C
  // CHECK-NEXT:   yield
  // CHECK-NEXT: } {_some_attr = 1 : index, branch_attrs = [{}]
  // CHECK-SAME: (tensor<i32>) -> ()
  %ctl_3 = Case(%Index) {
    _some_attr = 1 : index,
    Tin = [], Tout = [], output_shapes = [],
    branches = [#tf_type.func<@case2, {}>]
  } : (tensor<i32>) -> ()
}
