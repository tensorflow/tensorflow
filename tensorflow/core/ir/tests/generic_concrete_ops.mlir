// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// Check that the concrete ops can be generic.

// CHECK-LABEL: tfg.func generic @generic_func
// CHECK-SAME: %[[A:.*]]: !tf_type.tensor
// CHECK-NEXT: %[[B:.*]]: !tf_type.tensor
tfg.func generic @generic_func(%a: !tf_type.tensor, %b: !tf_type.tensor) -> (!tf_type.tensor) {
  // CHECK: If(%[[A]], %[[B]])
  %If, %ctlIf = If(%a, %b) {
    then_branch = #tf_type.func<@then, {}>, else_branch = #tf_type.func<@else, {}>
  } : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
  // CHECK: Case(%[[A]], %[[B]])
  %Case, %ctlCase = Case(%a, %b) {
    branches = [#tf_type.func<@case, {}>]
  } : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
  // CHECK: While(%[[A]], %[[B]])
  %While, %ctlWhile = While(%a, %b) {
    cond = #tf_type.func<@cond, {}>, body = #tf_type.func<@body, {}>, parallel_iterations = 10 : i64
  } : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
  // CHECK: For(%[[A]], %[[A]], %[[A]], %[[B]])
  %For, %ctlFor = For(%a, %a, %a, %b) {
    body = #tf_type.func<@body, {}>
  } : (!tf_type.tensor, !tf_type.tensor, !tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
  return(%a) : !tf_type.tensor
}
