// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[INDEX:.*]], %[[CTRL:.*]] = Index
  %Index, %ctl = Index : () -> (tensor<i32>)
  // CHECK: %[[DATA:.*]], %[[CTRL_0:.*]] = Data
  %Data, %ctl_0 = Data : () -> (tensor<f32>)
  // CHECK: %[[CASE_0:.*]], %[[CTRL_1:.*]] = Case(%[[INDEX]])
  // CHECK-SAME: branches = [#tf_type.func<@[[CASE_FUNC_0:.*]], {}>
  // CHECK-SAME: (tensor<i32>) -> (tensor<i32>)
  %CaseRegion, %ctl_1 = CaseRegion %Index {
    %A, %ctl_4 = A : () -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } {_some_attr = 1 : index, branch_attrs = [{}], region_attrs = [#tfg.region_attrs<{sym_name = "case0"} [] [{}]>]} : (tensor<i32>) -> tensor<i32>

  // CHECK: %[[CTRL_2:.*]] = Case(%[[INDEX]], %[[DATA]])
  // CHECK-SAME: branches = [#tf_type.func<@[[CASE_FUNC_1:.*]], {}>
  // CHECK-SAME: tensor<i32>, tensor<f32>
  %ctl_2 = CaseRegion %Index {
    %B, %ctl_4 = B(%Data) : (tensor<f32>) -> (tensor<i32>)
    yield
  } {_some_attr = 1 : index, branch_attrs = [{}], region_attrs = [#tfg.region_attrs<{sym_name = "case1"} [] []>]} : (tensor<i32>) -> ()

  // CHECK: %[[CTRL_3:.*]] = Case(%[[INDEX]])
  // CHECK-SAME: branches = [#tf_type.func<@[[CASE_FUNC_2:.*]], {}>
  // CHECK-SAME: tensor<i32>
  %ctl_3 = CaseRegion %Index {
    %C, %ctl_4 = C : () -> (tensor<i32>)
    yield
  } {_some_attr = 1 : index, branch_attrs = [{}], region_attrs = [#tfg.region_attrs<{sym_name = "case2"} [] []>]} : (tensor<i32>) -> ()
}

// CHECK: tfg.func @[[CASE_FUNC_0]]()
// CHECK-NEXT: -> (tensor<i32
// CHECK: tfg.func @[[CASE_FUNC_1]](%[[ARG0:.*]])
// CHECK-NOT: ->
// CHECK: {
// CHECK: tfg.func @[[CASE_FUNC_2]]()
// CHECK-NOT: ->
// CHECK: {
