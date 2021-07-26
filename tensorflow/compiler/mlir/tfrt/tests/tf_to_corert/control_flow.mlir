// RUN: tf-tfrt-opt -tf-to-tfrt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: func @cond_false(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
func @cond_false(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @cond_true(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
func @cond_true(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @cond(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle, %arg2: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
func @cond(%arg0: tensor<i1>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: [[cond:%.*]] = tfrt_fallback_async.predicate
  // CHECK: [[cond_res:%.*]]:2 = tfrt.cond [[cond]]
  // CHECK-SAME: @cond_true @cond_false(%arg0, %arg2) : (!tfrt.chain, !corert.tensorhandle)
  %2 = "tf.If"(%arg0, %arg1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = true} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  // CHECK: [[out_ch:%.*]] = tfrt.merge.chains [[cond_res]]#0, %arg0 : !tfrt.chain, !tfrt.chain
  // CHECK: tfrt.return [[out_ch]], [[cond_res]]#1 : !tfrt.chain, !corert.tensorhandle
  return %2 : tensor<i32>
}

// CHECK-LABEL: func @cond_stateful(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
func @cond_stateful(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: [[cond_res:%.*]]:2 = tfrt.cond
  // CHECK-SAME: @cond_true @cond_false(%arg0, %arg1) : (!tfrt.chain, !corert.tensorhandle)
  %2 = "tf.If"(%1, %arg0) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  // Note: returns %out_op_chain.
  // CHECK: tfrt.return [[cond_res]]#0, [[cond_res]]#1 : !tfrt.chain, !corert.tensorhandle
  return %2 : tensor<i32>
}

// CHECK-LABEL: func @while_cond_lt9
// CHECK-SAME: ({{%.+}}: !tfrt.chain, {{%.+}}: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
func @while_cond_lt9(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

// CHECK-LABEL: func @while_body_add2
// CHECK-SAME: ({{%.+}}: !tfrt.chain, {{%.+}}: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
func @while_body_add2(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @while_test
// CHECK-SAME: ([[ARG0:%.+]]: !tfrt.chain) -> (!tfrt.chain, !corert.tensorhandle)
func @while_test() -> (tensor<i32>) {
  // CHECK: [[CONST:%.+]] = corert.const_dense_tensor dense<0> : tensor<i32>
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[pred_res:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"([[ARG0]], [[CONST]]) : (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, i1)
  // CHECK: [[while_res:%.]]:2 = tfrt.while [[pred_res]]#1 @"while_body_add2/tfrt_body"([[pred_res]]#0, [[CONST]]) : (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
  %1 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: [[out_chain:%.*]] = tfrt.merge.chains [[while_res]]#0, [[ARG0]]
  // CHECK: tfrt.return [[out_chain]], [[while_res]]#1 : !tfrt.chain, !corert.tensorhandle
  return %1 : tensor<i32>
}
// CHECK: func @"while_body_add2/tfrt_body"([[ch:%.*]]: !tfrt.chain, [[arg:%.*]]: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle, i1)
// CHECK: [[body_res:%.*]]:2 = tfrt.call @while_body_add2([[ch]], [[arg]]) : (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
// CHECK: [[pred_res:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"([[body_res]]#0, [[body_res]]#1) : (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, i1)
// CHECK: tfrt.return [[pred_res]]#0, [[body_res]]#1, [[pred_res]]#1 : !tfrt.chain, !corert.tensorhandle, i1

// CHECK: func @"while_cond_lt9/tfrt_predicate"([[ch:%.*]]: !tfrt.chain, [[arg:%.*]]: !corert.tensorhandle) -> (!tfrt.chain, i1)
// CHECK: [[cond_res:%.*]]:2 = tfrt.call @while_cond_lt9([[ch]], [[arg]]) : (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
// CHECK: [[cond:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[cond_res]]#1
// CHECK-SAME: (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
// CHECK: [[bool_cond:%.*]] = tfrt_fallback_async.predicate [[cond]]
// CHECK: tfrt.return [[cond_res]]#0, [[bool_cond]] : !tfrt.chain, i1

// CHECK-LABEL: func @multi_while_test
func @multi_while_test() -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[pred_0:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"
  // CHECK: tfrt.while [[pred_0]]#1 @"while_body_add2/tfrt_body"
  // CHECK: [[pred_1:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"
  // CHECK: tfrt.while [[pred_1]]#1 @"while_body_add2/tfrt_body"
  %2 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  %3 = "tf.While"(%1) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false} : (tensor<i32>) -> (tensor<i32>)
  return %2, %3 : tensor<i32>, tensor<i32>
}

func @callee(%arg0: tensor<i32>) -> (tensor<i32>) {
  return %arg0: tensor<i32>
}

// CHECK-LABEL: func @call_test
// CHECK-SAME: ([[chain:%.*]]: !tfrt.chain,
func @call_test(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.Add"(%arg0, %arg0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[results_0:%.*]]:2 = tfrt.call @callee([[chain]]
  // CHECK-SAME: (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
  %1 = "tf.StatefulPartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: [[results_1:%.*]]:2 = tfrt.call @callee([[chain]]
  // CHECK-SAME: (!tfrt.chain, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)
  %2 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: [[results_0]]#1, [[results_1]]#1
  return %1, %2 : tensor<i32>, tensor<i32>
}

func @branch0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Add" (%arg0, %arg1) {device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func @branch1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Add" (%arg0, %arg1) {device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tf.Add" (%arg0, %0) {device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @case_test
// CHECK-SAME: ([[chain:%.*]]: !tfrt.chain, [[idx:%.*]]: !corert.tensorhandle, [[arg:%.*]]: !corert.tensorhandle
func @case_test(%arg0: tensor<i32>, %arg1: tensor<f32>,  %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[res_idx:%.*]] = corert.tensorhandle_to_int32 [[idx:%.*]]
  // CHECK-NEXT: [[out_chain:%.*]], [[out:%.*]] = tfrt.case [[res_idx:%.*]] [@branch0, @branch1]([[arg:%.*]]
  %0 = "tf.Case"(%arg0, %arg1, %arg2) {_lower_using_switch_merge = true, branches = [@branch0, @branch1], is_stateless = true} : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
