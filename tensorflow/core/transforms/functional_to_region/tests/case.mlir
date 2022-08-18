// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @case0
tfg.func @case0(%arg0: tensor<f32>) -> (tensor<i32>) {
  %A, %ctl = A(%arg0) : (tensor<f32>) -> (tensor<i32>)
  return(%A) : tensor<i32>
}

// Argument attributes are dropped when converted to implicit capture.
// CHECK: tfg.func @case1
tfg.func @case1(%arg0: tensor<f32> {tf._a}) -> (tensor<i32>) {
  %B, %ctl = B(%arg0) : (tensor<f32>) -> (tensor<i32>)
  return(%B) : tensor<i32>
}

// CHECK: tfg.func @case2
tfg.func @case2(%arg0: tensor<f32>) -> (tensor<i32>) {
  %C, %ctl = C(%arg0) : (tensor<f32>) -> (tensor<i32>)
  return(%C) : tensor<i32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK:      %[[INDEX:.*]], %[[CTL:.*]] = Index
  %Index, %ctl = Index : () -> (tensor<i32>)
  // CHECK-NEXT: %[[DATA:.*]], %[[CTL_0:.*]] = Data
  %Data, %ctl_0 = Data : () -> (tensor<f32>)
  // CHECK-NEXT: %[[CASE:.*]], %[[CTL_1:.*]] = CaseRegion %[[INDEX]]  {
  // CHECK-NEXT:   %[[A:.*]], %[[CTL_2:.*]] = A(%[[DATA]])
  // CHECK-NEXT:   yield(%[[A]])
  // CHECK-NEXT: },  {
  // CHECK-NEXT:   %[[B:.*]], %[[CTL_2:.*]] = B(%[[DATA]])
  // CHECK-NEXT:   yield(%[[B]])
  // CHECK-NEXT: },  {
  // CHECK-NEXT:   %[[C:.*]], %[[CTL_2:.*]] = C(%[[DATA]])
  // CHECK-NEXT:   yield(%[[C]])
  // CHECK-NEXT: } {_some_attr = 1 : index, branch_attrs = [{}, {}, {}]
  // CHECK-SAME: (tensor<i32>) -> tensor<{{.*}}>
  %Case, %ctl_1 = Case(%Index, %Data) {
    _some_attr = 1 : index,
    // Test fix output shapes.
    Tin = [f32], Tout = [i32], output_shapes = [],
    branches = [#tf_type.func<@case0, {}>,
                #tf_type.func<@case1, {}>,
                #tf_type.func<@case2, {}>]
  } : (tensor<i32>, tensor<f32>) -> (tensor<i32>)
}
