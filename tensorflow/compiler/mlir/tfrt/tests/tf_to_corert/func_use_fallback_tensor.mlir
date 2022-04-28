// RUN: tf-tfrt-opt -tf-to-tfrt="func-use-fallback-tensor=true enable-while-parallel-iterations=true" %s | FileCheck %s --dump-input=fail

// This file tests the correctness of `func-use-fallback-tensor` option when
// converting from TF to TFRT. Since func op is used by the control flow ops,
// the test cases here should cover the control flow ops.

// CHECK-LABEL: func @cond_false(%arg0: !tfrt.chain, %arg1: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @cond_false(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @cond_true(%arg0: !tfrt.chain, %arg1: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @cond_true(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @cond(%arg0: !tfrt.chain, %arg1: !tfrt_fallback.tf_tensor, %arg2: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @cond(%arg0: tensor<i1>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: [[cond:%.*]] = tfrt_fallback_async.predicate
  // CHECK: [[cond_res:%.*]]:2 = tfrt.cond [[cond]]
  // CHECK-SAME: @cond_true @cond_false(%arg0, %arg2) : (!tfrt.chain, !tfrt_fallback.tf_tensor)
  %2 = "tf.If"(%arg0, %arg1) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = true} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  // CHECK: [[out_ch:%.*]] = tfrt.merge.chains [[cond_res]]#0, %arg0 : !tfrt.chain, !tfrt.chain
  // CHECK: tfrt.return [[out_ch]], [[cond_res]]#1 : !tfrt.chain, !tfrt_fallback.tf_tensor
  func.return %2 : tensor<i32>
}

// CHECK-LABEL: func @cond_stateful(%arg0: !tfrt.chain, %arg1: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @cond_stateful(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK: [[cond_res:%.*]]:2 = tfrt.cond
  // CHECK-SAME: @cond_true @cond_false(%arg0, %arg1) : (!tfrt.chain, !tfrt_fallback.tf_tensor)
  %2 = "tf.If"(%1, %arg0) {else_branch = @cond_false, then_branch = @cond_true, is_stateless = false} : (tensor<i1>, tensor<i32>) -> tensor<i32>
  // Note: returns %out_op_chain.
  // CHECK: tfrt.return [[cond_res]]#0, [[cond_res]]#1 : !tfrt.chain, !tfrt_fallback.tf_tensor
  func.return %2 : tensor<i32>
}

// CHECK-LABEL: func @while_cond_lt9
// CHECK-SAME: ({{%.+}}: !tfrt.chain, {{%.+}}: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @while_cond_lt9(%arg0: tensor<i32>) -> tensor<i1> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

// CHECK-LABEL: func @while_body_add2
// CHECK-SAME: ({{%.+}}: !tfrt.chain, {{%.+}}: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @while_body_add2(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: func @while_test
// CHECK-SAME: ([[ARG0:%.+]]: !tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
func.func @while_test() -> (tensor<i32>) {
  // CHECK: [[CONST_TH:%.*]] = corert.const_dense_tensor dense<0> : tensor<i32>
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[CONST:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[CONST_TH]]
  // CHECK: (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  // CHECK: [[pred_res:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"([[ARG0]], [[CONST]]) : (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, i1)
  // CHECK: [[while_res:%.]]:2 = tfrt.while [[pred_res]]#1 @"while_body_add2/tfrt_body_1"([[pred_res]]#0, [[CONST]])
  // CHECK-SAME: (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  %1 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: [[out_chain:%.*]] = tfrt.merge.chains [[while_res]]#0, [[ARG0]]
  // CHECK: tfrt.return [[out_chain]], [[while_res]]#1 : !tfrt.chain, !tfrt_fallback.tf_tensor
  func.return %1 : tensor<i32>
}
// CHECK: func @"while_body_add2/tfrt_body_1"([[ch:%.*]]: !tfrt.chain, [[arg:%.*]]: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor, i1)
// CHECK: [[body_res:%.*]]:2 = tfrt.call @while_body_add2([[ch]], [[arg]]) : (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
// CHECK: [[pred_res:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"([[body_res]]#0, [[body_res]]#1) : (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, i1)
// CHECK: tfrt.return [[pred_res]]#0, [[body_res]]#1, [[pred_res]]#1 : !tfrt.chain, !tfrt_fallback.tf_tensor, i1

// CHECK: func @"while_cond_lt9/tfrt_predicate"([[ch:%.*]]: !tfrt.chain, [[arg:%.*]]: !tfrt_fallback.tf_tensor) -> (!tfrt.chain, i1)
// CHECK: [[cond_res:%.*]]:2 = tfrt.call @while_cond_lt9([[ch]], [[arg]]) : (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
// CHECK: [[bool_cond:%.*]] = tfrt_fallback_async.predicate [[cond_res]]#1
// CHECK: tfrt.return [[cond_res]]#0, [[bool_cond]] : !tfrt.chain, i1

// CHECK-LABEL: func @multi_while_test
func.func @multi_while_test() -> (tensor<i32>, tensor<i32>) {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: [[pred_0:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"
  // CHECK: tfrt.while [[pred_0]]#1 @"while_body_add2/tfrt_body_10"
  // CHECK-SAME: parallel_iterations(10)
  // CHECK: [[pred_1:%.*]]:2 = tfrt.call @"while_cond_lt9/tfrt_predicate"
  // CHECK: tfrt.while [[pred_1]]#1 @"while_body_add2/tfrt_body_1"
  // CHECK-SAME: parallel_iterations(1)
  %2 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 10} : (tensor<i32>) -> (tensor<i32>)
  %3 = "tf.While"(%1) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
  func.return %2, %3 : tensor<i32>, tensor<i32>
}

func.func @side_effect_while_cond_lt9(%arg: tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i1> {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.ReadVariableOp"(%arg) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf.Less"(%1, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %2 : tensor<i1>
}

func.func @side_effect_while_body_add2(%arg: tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<!tf_type.resource<tensor<i32>>>) {
  %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.ReadVariableOp"(%arg) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %2 = "tf.Add"(%1, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  "tf.AssignVariableOp"(%arg, %2) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return %arg : tensor<!tf_type.resource<tensor<i32>>>
}

// CHECK-LABEL: func @side_effect_while_test
func.func @side_effect_while_test() -> (tensor<i32>) {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "c", shared_name = "v"} : () -> tensor<!tf_type.resource<tensor<i32>>>
  // CHECK: [[while_res:%.]]:2 = tfrt.while {{%.*}} @"side_effect_while_body_add2/tfrt_body_1"
  // CHECK: [[out_ch:%.*]], [[res:%.*]] = tfrt_fallback_async.executeop.seq([[while_res]]#0) {{.*}} "tf.ReadVariableOp"
  %1 = "tf.While"(%0) { cond = @side_effect_while_cond_lt9, body = @side_effect_while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<!tf_type.resource<tensor<i32>>>) -> (tensor<!tf_type.resource<tensor<i32>>>)
  %2 = "tf.ReadVariableOp"(%1) {device = "/device:CPU:0", dtype = i32} : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

func.func @tensor_array_while_cond(%index: tensor<i32>, %size: tensor<i32>, %flow_0: tensor<f32>, %flow_1: tensor<f32>, %handle_0: tensor<2x!tf_type.resource<tensor<?x100xf32>>>, %handle_1: tensor<2x!tf_type.resource<tensor<?x512xf32>>>) -> (tensor<i1>) {
  %0 = "tf.Less"(%index, %size) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

func.func @tensor_array_while_body(%index: tensor<i32>, %size: tensor<i32>, %flow_0: tensor<f32>, %flow_1: tensor<f32>, %handle_0: tensor<2x!tf_type.resource<tensor<?x100xf32>>>, %handle_1: tensor<2x!tf_type.resource<tensor<?x512xf32>>>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>) {
  %cst = "tf.Const"() {value = dense<1.1> : tensor<100x512xf32>} : () -> tensor<100x512xf32>
  %one = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %x = "tf.TensorArrayReadV3"(%handle_0, %index, %flow_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<i32>, tensor<f32>) -> tensor<?x100xf32>
  %y = "tf.MatMul"(%x, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x100xf32>, tensor<100x512xf32>) -> (tensor<?x512xf32>)
  %flow_1_out = "tf.TensorArrayWriteV3"(%handle_1, %index, %y, %flow_1) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<?x512xf32>>>, tensor<i32>, tensor<?x512xf32>, tensor<f32>) -> tensor<f32>
  %next_index = "tf.AddV2"(%index, %one) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %next_index, %size, %flow_0, %flow_1_out, %handle_0, %handle_1 : tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>
}

// CHECK-LABEL: func @tensor_array_while_test
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain
func.func @tensor_array_while_test(%indices: tensor<?xi32>, %input_0: tensor<?x?x?xf32>, %input_1: tensor<?x?x?xf32>) -> (tensor<?x?x512xf32>, tensor<?x?x512xf32>) {
  %index = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  %size = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> (tensor<i32>)
  %handle_0, %flow_0 = "tf.TensorArrayV3"(%size) {clear_after_read = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = f32, dynamic_size = false, element_shape = #tf_type.shape<?x100>, identical_element_shapes = true, tensor_array_name = "processed_embeddings/bidirectional_rnn/bw/bw/dynamic_rnn/input_0"} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<f32>)
  %handle_1, %flow_1 = "tf.TensorArrayV3"(%size) {clear_after_read = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = f32, dynamic_size = false, element_shape = #tf_type.shape<?x512>, identical_element_shapes = true, tensor_array_name = "processed_embeddings/bidirectional_rnn/bw/bw/dynamic_rnn/output_0"} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<?x512xf32>>>, tensor<f32>)
  %flow_01 = "tf.TensorArrayScatterV3"(%handle_0, %indices, %input_0, %flow_0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<?xi32>, tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: [[pred_0:%.*]]:2 = tfrt.call @"tensor_array_while_cond/tfrt_predicate"([[in_chain]]
  // CHECK: [[while_res_0:%.*]]:7 = tfrt.while {{%.*}} @"tensor_array_while_body/tfrt_body_10"([[pred_0]]#0
  // CHECK-SAME: parallel_iterations(10)
  %res_0:6 = "tf.While"(%index, %size, %flow_01, %flow_1, %handle_0, %handle_1) {body = @tensor_array_while_body, cond = @tensor_array_while_cond, device = "", is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>)
  %output_0 = "tf.TensorArrayGatherV3"(%handle_1, %indices, %res_0#3) {device = "/job:localhost/replica:0/task:0/device:CPU:0", element_shape = #tf_type.shape<?x512>} : (tensor<2x!tf_type.resource<tensor<?x512xf32>>>, tensor<?xi32>, tensor<f32>) -> tensor<?x?x512xf32>

  %handle_2, %flow_2 = "tf.TensorArrayV3"(%size) {clear_after_read = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = f32, dynamic_size = false, element_shape = #tf_type.shape<?x100>, identical_element_shapes = true, tensor_array_name = "processed_embeddings/bidirectional_rnn/bw/bw/dynamic_rnn/input_0"} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<f32>)
  %handle_3, %flow_3 = "tf.TensorArrayV3"(%size) {clear_after_read = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", dtype = f32, dynamic_size = false, element_shape = #tf_type.shape<?x512>, identical_element_shapes = true, tensor_array_name = "processed_embeddings/bidirectional_rnn/bw/bw/dynamic_rnn/output_0"} : (tensor<i32>) -> (tensor<2x!tf_type.resource<tensor<?x512xf32>>>, tensor<f32>)
  %flow_21 = "tf.TensorArrayScatterV3"(%handle_2, %indices, %input_1, %flow_2) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<?xi32>, tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: [[pred_1:%.*]]:2 = tfrt.call @"tensor_array_while_cond/tfrt_predicate"([[in_chain]]
  // CHECK: [[while_res_1:%.*]]:7 = tfrt.while {{%.*}} @"tensor_array_while_body/tfrt_body_10"([[pred_1]]#0
  // CHECK-SAME: parallel_iterations(10)
  %res_1:6 = "tf.While"(%index, %size, %flow_21, %flow_3, %handle_2, %handle_3) {body = @tensor_array_while_body, cond = @tensor_array_while_cond, device = "", is_stateless = false, parallel_iterations = 10 : i64} : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>, tensor<2x!tf_type.resource<tensor<?x100xf32>>>, tensor<2x!tf_type.resource<tensor<?x512xf32>>>)
  %output_1 = "tf.TensorArrayGatherV3"(%handle_3, %indices, %res_1#3) {device = "/job:localhost/replica:0/task:0/device:CPU:0", element_shape = #tf_type.shape<?x512>} : (tensor<2x!tf_type.resource<tensor<?x512xf32>>>, tensor<?xi32>, tensor<f32>) -> tensor<?x?x512xf32>
  func.return %output_0, %output_1 : tensor<?x?x512xf32>, tensor<?x?x512xf32>
}

// CHECK: func @"tensor_array_while_body/tfrt_body_10"

func.func @callee(%arg0: tensor<i32>) -> (tensor<i32>) {
  func.return %arg0: tensor<i32>
}

// CHECK-LABEL: func @call_test
// CHECK-SAME: ([[chain:%.*]]: !tfrt.chain,
func.func @call_test(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %0 = "tf.Add"(%arg0, %arg0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: [[results_0:%.*]]:2 = tfrt.call @callee([[chain]]
  // CHECK-SAME: (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  %1 = "tf.StatefulPartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK-NEXT: [[results_1:%.*]]:2 = tfrt.call @callee([[chain]]
  // CHECK-SAME: (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  %2 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK-NEXT: [[results_2:%.*]]:2 = tfrt.call @callee([[chain]]
  // CHECK-SAME: (!tfrt.chain, !tfrt_fallback.tf_tensor) -> (!tfrt.chain, !tfrt_fallback.tf_tensor)
  %3 = "tf.LegacyCall"(%0) {f = @callee} : (tensor<i32>) -> (tensor<i32>)
  // CHECK: [[results_0]]#1, [[results_1]]#1, [[results_2]]#1
  func.return %1, %2, %3 : tensor<i32>, tensor<i32>, tensor<i32>
}

func.func @branch0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Add" (%arg0, %arg1) {device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

func.func @branch1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Add" (%arg0, %arg1) {device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tf.Add" (%arg0, %0) {device = "/device:CPU:0"}  : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %1 : tensor<f32>
}

// CHECK-LABEL: func @case_test
// CHECK-SAME: ([[chain:%.*]]: !tfrt.chain, [[tf_idx:%.*]]: !tfrt_fallback.tf_tensor, [[branch_arg0:%.*]]: !tfrt_fallback.tf_tensor, [[branch_arg1:%.*]]: !tfrt_fallback.tf_tensor)
func.func @case_test(%arg0: tensor<i32>, %arg1: tensor<f32>,  %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: [[th_idx:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[tf_idx]]
  // CHECK-NEXT: [[idx:%.*]] = corert.tensorhandle_to_int32 [[th_idx]]
  // CHECK-NEXT: [[out:%.*]] = tfrt.case [[idx]] [@branch0, @branch1]([[chain]], [[branch_arg0]], [[branch_arg1]])
  %0 = "tf.Case"(%arg0, %arg1, %arg2) {_lower_using_switch_merge = true, branches = [@branch0, @branch1], is_stateless = true} : (tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
