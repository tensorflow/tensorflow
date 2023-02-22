// RUN: tf-opt %s --tf-remove-unused-while-results --split-input-file | FileCheck %s

// Check that we remove the first result and operand from the loop because the
// result is unused and all other conditions are satisfied, that means:
// - the corresponding first block arguments are unused (except for `tf.OpB`
//   which is the defining op of the first result and will be pruned)
// - `tf.OpB` is not stateful
// - `tf.OpB` has only one result, and that result is only passed through to the
//   unused result of the while loop

// CHECK-LABEL: remove_first_result
// CHECK:       tf.WhileRegion
// CHECK-SAME:    (%[[ARG:[a-zA-Z0-9_]+]])
// CHECK:       ^bb0
// CHECK:         (%[[CARG:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:         (%[[BARG:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK-NOT:   tf.OpB
// CHECK:       tf.OpC
// CHECK-SAME:    %[[BARG]]
func.func @remove_first_result(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpB"(%arg2) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.OpC"(%arg3) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#1 : tensor<*xf32>
}

// -----

// Check that we remove the second result and operand from the loop because the
// result is unused and all other conditions are satisfied.

// CHECK-LABEL: remove_second_result
// CHECK:       tf.WhileRegion
// CHECK-SAME:    (%[[ARG:[a-zA-Z0-9_]+]])
// CHECK:       ^bb0
// CHECK:         (%[[CARG:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:         (%[[BARG:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpB
// CHECK-SAME:    %[[BARG]]
// CHECK-NOT:   tf.OpC
func.func @remove_second_result(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpB"(%arg2) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.OpC"(%arg3) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#0 : tensor<*xf32>
}

// -----

// Check that we don't remove the first result and operand from the loop (even
// though the result is unused) because the corresponding block argument is used
// in the while condition (`%arg2` is used by `tf.OpA`).

// CHECK-LABEL: result_used_in_cond
// CHECK:       tf.WhileRegion
// CHECK-SAME:    (%[[ARG0:[a-zA-Z0-9_]+]], %[[ARG1:[a-zA-Z0-9_]+]])
// CHECK:       ^bb0
// CHECK:         (%[[CARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[CARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:         (%[[BARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[BARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpB
// CHECK:       tf.OpC
func.func @result_used_in_cond(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpA"(%arg2) {is_stateless = true} : (tensor<*xf32>) -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpB"(%arg2) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.OpC"(%arg3) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#1 : tensor<*xf32>
}

// -----

// Check that we don't remove the first result and operand from the loop (even
// though the result is unused) because the op that defines the result (passed
// through via `tf.Yield`) is stateful
// (`"tf.OpB"(%arg2) {is_stateless = false}`).

// CHECK-LABEL: defining_op_is_stateful
// CHECK:       tf.WhileRegion
// CHECK-SAME:    (%[[ARG0:[a-zA-Z0-9_]+]], %[[ARG1:[a-zA-Z0-9_]+]])
// CHECK:       ^bb0
// CHECK:         (%[[CARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[CARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:         (%[[BARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[BARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpB
// CHECK:       tf.OpC
func.func @defining_op_is_stateful(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpB"(%arg2) {is_stateless = false} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.OpC"(%arg3) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#1 : tensor<*xf32>
}

// -----

// Check that we don't remove the first result and operand from the loop (even
// though the result is unused) because the single result of the defining op has
// more than one use (`%1` is used by `tf.OpC` in addition to `tf.Yield`).

// CHECK-LABEL: defining_op_has_other_used_results
// CHECK:       tf.WhileRegion
// CHECK-SAME:    (%[[ARG0:[a-zA-Z0-9_]+]], %[[ARG1:[a-zA-Z0-9_]+]])
// CHECK:       ^bb0
// CHECK:         (%[[CARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[CARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:         (%[[BARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[BARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpB
// CHECK:       tf.OpC
func.func @defining_op_has_other_used_results(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpB"(%arg2) {is_stateless = true} : (tensor<*xf32>) -> tensor<*xf32>
    %2 = "tf.OpC"(%arg3, %1) {is_stateless = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#1 : tensor<*xf32>
}

// -----

// Check that we don't remove the first result and operand from the loop (even
// though the result is unused) because the defining op has another result that
// is used (`%1#1` is used by `tf.OpC`).

// CHECK-LABEL: defining_op_has_other_used_results
// CHECK:       tf.WhileRegion
// CHECK-SAME:    (%[[ARG0:[a-zA-Z0-9_]+]], %[[ARG1:[a-zA-Z0-9_]+]])
// CHECK:       ^bb0
// CHECK:         (%[[CARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[CARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpA
// CHECK:       ^bb0
// CHECK:         (%[[BARG0:[a-zA-Z0-9_]+]]: tensor<*xf32>, %[[BARG1:[a-zA-Z0-9_]+]]: tensor<*xf32>)
// CHECK:       tf.OpB
// CHECK:       tf.OpC
func.func @defining_op_has_other_used_results(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0:2 = "tf.WhileRegion"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1 = "tf.OpA"() {is_stateless = true} : () -> tensor<i1>
    "tf.Yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
    %1:2 = "tf.OpB"(%arg2) {is_stateless = true} : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
    %2 = "tf.OpC"(%arg3, %1#1) {is_stateless = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tf.Yield"(%1#0, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()
  }) {is_stateless = true, parallel_iterations = 10 : i64} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#1 : tensor<*xf32>
}
