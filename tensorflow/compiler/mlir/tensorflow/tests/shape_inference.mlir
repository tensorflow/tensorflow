// RUN: tf-opt %s -tf-shape-inference=propagate-caller-callee-constants=false -verify-diagnostics | FileCheck %s -dump-input=fail
// RUN: tf-opt %s -tf-shape-inference=propagate-caller-callee-constants -verify-diagnostics | FileCheck %s -dump-input=fail

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 130 : i32}} {
// CHECK-LABEL: func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32>
  func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<*xi32> {
 // CHECK: %[[RESULT:.*]] = "tf.AddV2"
 // CHECK-SAME: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
 // CHECK: return %[[RESULT]] : tensor<1xi32>
    %0 = "tf.Cast"(%arg0) : (tensor<1xi32>) -> tensor<*xi32>
    %1 = "tf.Cast"(%arg1) : (tensor<1xi32>) -> tensor<*xi32>
    %2 = "tf.AddV2"(%0, %1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    return %2 : tensor<*xi32>
  }

// CHECK-LABEL: func @simple_chain
  func @simple_chain(%arg0: tensor<1xf32>) -> tensor<*xf32> {
// CHECK: %[[MUL:.*]] = "tf.Mul"{{.*}} (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[ADD:.*]] = "tf.Add"(%[[MUL]], %[[MUL]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[ADD]] : tensor<1xf32>
    %0 = "tf.Mul"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
    %1 = "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

// CHECK-LABEL: func @simple_chain_with_broadcast
  func @simple_chain_with_broadcast(%arg0: tensor<1xf32>, %arg1: tensor<10xf32>) -> tensor<*xf32> {
// CHECK: %[[MUL:.*]] = "tf.Mul"{{.*}} (tensor<1xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[ADD:.*]] = "tf.Add"(%[[MUL]], %[[MUL]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST:.*]] = "tf.Cast"(%[[ADD]]) {{.*}} : (tensor<10xf32>) -> tensor<*xf32>
// CHECK: %[[UNKNOWN:.*]] = addf %[[CAST]], %[[CAST]] : tensor<*xf32>
// CHECK: return %[[UNKNOWN]] : tensor<*xf32>
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<1xf32>, tensor<10xf32>) -> tensor<*xf32>
    %1 = "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = addf %1, %1 : tensor<*xf32>
    return %2 : tensor<*xf32>
  }

// CHECK-LABEL: func @unknown_op
  func @unknown_op(%arg0: tensor<1xf32>) -> tensor<*xf32> {
// CHECK: %[[MUL:.*]] = "tf.Mul"{{.*}} (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: %[[UNKNOWN:.*]] = "tf.Unknown"(%[[MUL]], %[[MUL]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
// CHECK: return %[[UNKNOWN]] : tensor<*xf32>
    %0 = "tf.Mul"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
    %1 = "tf.Unknown"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    return %1 : tensor<*xf32>
  }

// CHECK-LABEL: func @multiple_blocks_one_return(%arg0: tensor<?xf32>) -> tensor<?xf32>
func @multiple_blocks_one_return(%arg0: tensor<?xf32>) -> tensor<*xf32> {
  br ^bb1
^bb1:
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
// CHECK: return %[[IDENTITY]] : tensor<?xf32>
  %ret = "tf.Identity"(%arg0) : (tensor<?xf32>) -> tensor<*xf32>
  return %ret : tensor<*xf32>
}


// Tests the case where an inference opportunity relies on folding.

// CHECK-LABEL: func @simple_folding
  func @simple_folding(%arg0: tensor<1x1x1x1xi32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32> {
// CHECK: %[[SHAPE:.*]] = "tf.Shape"
// CHECK: %[[CONV:.*]] = "tf.Conv2DBackpropInput"(%[[SHAPE]]
// CHECK-SAME: (tensor<4xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK: return %[[CONV]] : tensor<1x1x1x1xf32>
    %0 = "tf.Shape"(%arg0) : (tensor<1x1x1x1xi32>) -> tensor<4xi32>
    %1 = "tf.Conv2DBackpropInput"(%0, %arg1, %arg1) {
      padding = "VALID", strides = [1, 1, 1, 1]
    } : (tensor<4xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
  }

// Tests where tf.Const's value needs to be refined.

  func @const_refine() -> tensor<*xi32> {
    %0 = "tf.Const"() {value = dense<[3, 2]> : tensor<2xi32>} : () -> tensor<*xi32>
    // CHECK: "tf.Const"
    // CHECK-SAME: -> tensor<2xi32>
    return %0 : tensor<*xi32>
  }

// Tests the case where an op's shape function returns non-fully-defined shapes.

// CHECK-LABEL: func @op_non_fully_defined_shape_fn
  func @op_non_fully_defined_shape_fn(%arg0: tensor<0xi32>, %arg1: tensor<0xi32>) -> tensor<?xi32> {
    // CHECK: tf.BroadcastGradientArgs
    // CHECK-SAME: (tensor<0xi32>, tensor<0xi32>) -> (tensor<?xi32>, tensor<?xi32>)
    %2:2 = "tf.BroadcastGradientArgs"(%arg0, %arg1) {T = "tfdtype$DT_INT32", name = "BroadcastGradientArgs"} : (tensor<0xi32>, tensor<0xi32>) -> (tensor<?xi32>, tensor<?xi32>)
    return %2#0 : tensor<?xi32>
  }

// CHECK-LABEL: func @shape_from_const_input
  func @shape_from_const_input(%arg0: tensor<3x3x32x64xf32>, %arg1: tensor<200x24x24x64xf32>) -> tensor<?x?x?x?xf32> {
    %0 = "tf.Const"() {value = dense<[200, 26, 26, 32]> : tensor<4xi32>} : () -> tensor<4xi32>
    // CHECK: tf.Conv2DBackpropInput
    // CHECK-SAME: (tensor<4xi32>, tensor<3x3x32x64xf32>, tensor<200x24x24x64xf32>) -> tensor<200x26x26x32xf32>
    %1 = "tf.Conv2DBackpropInput"(%0, %arg0, %arg1) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<4xi32>, tensor<3x3x32x64xf32>, tensor<200x24x24x64xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
  }

  // CHECK-LABEL: func @shape_from_if_to_branch_functions
  func @shape_from_if_to_branch_functions(%arg0: tensor<i1>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
    %0 = "tf.If"(%arg0, %arg1) {Tcond = i1, Tin = ["tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _xla_propagate_compile_time_consts = true, device = "", else_branch = @if_else_branch, is_stateless = true, name = "if", output_shapes = [#tf.shape<>], then_branch = @if_then_branch} : (tensor<i1>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }

  // CHECK-LABEL: func @if_then_branch
  // CHECK-SAME: (%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  func @if_then_branch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    // CHECK: return
    // CHECK-SAME: tensor<1x2x3xf32>
    return %arg0 : tensor<*xf32>
  }

  // CHECK-LABEL: func @if_else_branch
  // CHECK-SAME: (%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  func @if_else_branch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    // CHECK: "tf.Identity"(%arg0) : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    %0 = "tf.Identity"(%arg0) : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: return
    // CHECK-SAME: tensor<1x2x3xf32>
    return %0 : tensor<*xf32>
  }

  // CHECK-LABEL: func @shape_from_while_to_cond_body_functions
  func @shape_from_while_to_cond_body_functions(%arg0: tensor<4xf32>, %arg1: tensor<!tf.resource<tensor<4xf32>>>, %arg2: tensor<!tf.resource<tensor<*xf32>>>) -> tensor<4xf32> {
    // CHECK: "tf.While"
    // CHECK-SAME: (tensor<4xf32>, tensor<!tf.resource<tensor<4xf32>>>, tensor<!tf.resource<tensor<*xf32>>>) -> (tensor<4xf32>, tensor<!tf.resource<tensor<4xf32>>>, tensor<!tf.resource<tensor<*xf32>>>)
    %0:3 = "tf.While"(%arg0, %arg1, %arg2) {cond = @while_cond_func, body = @while_body_func, is_stateless = true} : (tensor<4xf32>, tensor<!tf.resource<tensor<4xf32>>>, tensor<!tf.resource<tensor<*xf32>>>) -> (tensor<4xf32>, tensor<*x!tf.resource>, tensor<!tf.resource<tensor<*xf32>>>)
    return %0#0 : tensor<4xf32>
  }

  // CHECK-LABEL: func @while_cond_func
  // CHECK-SAME: (%arg0: tensor<4xf32>, %arg1: tensor<!tf.resource<tensor<4xf32>>>, %arg2: tensor<!tf.resource<tensor<*xf32>>>) -> tensor<i1>
  func @while_cond_func(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf.resource>, %arg2: tensor<!tf.resource<tensor<*xf32>>>) -> tensor<i1> {
    %0 = "tf.Const"() {value = dense<[1.000000e-04,2.000000e-04,3.000000e-04,4.000000e-04]> : tensor<4xf32>} : () -> tensor<4xf32>
    %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    // CHECK: tf.Equal
    // CHECK-SAME: (tensor<4xf32>, tensor<4xf32>) -> tensor<*xi1>
    // TODO(ycao): Investigate why result type of tf.Equal is not inferred.
    %2 = "tf.Equal"(%0, %arg0) : (tensor<4xf32>, tensor<*xf32>) -> tensor<*xi1>
    %3 = "tf.Any"(%2, %1) : (tensor<*xi1>, tensor<i32>) -> (tensor<i1>)
    return %3 : tensor<i1>
  }

  // CHECK-LABEL: func @while_body_func
  func @while_body_func(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf.resource>, %arg2: tensor<!tf.resource<tensor<*xf32>>>) -> (tensor<*xf32>, tensor<*x!tf.resource>, tensor<!tf.resource<tensor<*xf32>>>) {
    %0 = "tf.Const"() {value = dense<1.000000e-04> : tensor<f32>} : () -> tensor<f32>
    // CHECK: tf.AddV2
    // CHECK-SAME: (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
    %1 = "tf.AddV2"(%arg0, %0) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    // CHECK: "tf.Identity"
    // CHECK-SAME: (tensor<!tf.resource<tensor<4xf32>>>) -> tensor<!tf.resource<tensor<4xf32>>>
    %2 = "tf.Identity"(%arg1) : (tensor<*x!tf.resource>) -> tensor<*x!tf.resource>
    // CHECK: "tf.TPUReplicatedInput"
    // CHECK-SAME: (tensor<!tf.resource<tensor<4xf32>>>) -> tensor<!tf.resource<tensor<4xf32>>>
    %ri = "tf.TPUReplicatedInput"(%2) : (tensor<*x!tf.resource>) -> tensor<*x!tf.resource>
    // CHECK: "tf.ReadVariableOp"
    // CHECK-SAME: (tensor<!tf.resource<tensor<4xf32>>>) -> tensor<4xf32>
    %read = "tf.ReadVariableOp"(%ri) : (tensor<*x!tf.resource>) -> tensor<*xf32>
    // CHECK: "tf.ReadVariableOp"
    // CHECK-SAME: (tensor<!tf.resource<tensor<*xf32>>>) -> tensor<*xf32>
    %read1 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf.resource<tensor<*xf32>>>) -> tensor<*xf32>
    // CHECK: return
    // CHECK-SAME: tensor<4xf32>
    // CHECK-SAME: tensor<!tf.resource<tensor<4xf32>>>
    return %1, %arg1, %arg2 : tensor<*xf32>, tensor<*x!tf.resource>, tensor<!tf.resource<tensor<*xf32>>>
  }

  func @partitioned_call(%arg0: tensor<i32>) -> tensor<*xi32> {
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @partitioned_call_func} : (tensor<i32>) -> (tensor<*xi32>)
    return %0 : tensor<*xi32>
  }

  // CHECK-LABEL: func @partitioned_call_func
  // CHECK-SAME: (%arg0: tensor<i32>) -> tensor<i32>
  func @partitioned_call_func(%arg0: tensor<*xi32>) -> tensor<*xi32> {
    // CHECK: return
    // CHECK-SAME: tensor<i32>
    return %arg0 : tensor<*xi32>
  }

  // CHECK-LABEL: func @invalid_function_reused_by_control_flows
  func @invalid_function_reused_by_control_flows(%arg0: tensor<i1>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
	  // expected-warning @+1 {{unable to refine shape}}
    %0 = "tf.If"(%arg0, %arg1) {Tcond = i1, Tin = ["tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _xla_propagate_compile_time_consts = true, device = "", else_branch = @reused_if_else_branch, is_stateless = true, name = "if", output_shapes = [#tf.shape<>], then_branch = @reused_if_then_branch} : (tensor<i1>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
	  // expected-warning @+1 {{unable to refine shape}}
    %1 = "tf.If"(%arg0, %0) {Tcond = i1, Tin = ["tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _xla_propagate_compile_time_consts = true, device = "", else_branch = @reused_if_else_branch, is_stateless = true, name = "if", output_shapes = [#tf.shape<>], then_branch = @reused_if_then_branch} : (tensor<i1>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }

  // CHECK-LABEL: func @reused_if_then_branch
  // CHECK-SAME: (%arg0: tensor<*xf32>) -> tensor<*xf32>
	// expected-warning @+1 {{expected control flow function reused_if_then_branch to have exactly 1 use}}
  func @reused_if_then_branch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    // CHECK: return
    // CHECK-SAME: tensor<*xf32>
    return %arg0 : tensor<*xf32>
  }

  // CHECK-LABEL: func @reused_if_else_branch
  // CHECK-SAME: (%arg0: tensor<*xf32>) -> tensor<*xf32>
	// expected-warning @+1 {{expected control flow function reused_if_else_branch to have exactly 1 use}}
  func @reused_if_else_branch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    // CHECK: "tf.Identity"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
    %0 = "tf.Identity"(%arg0) : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: return
    // CHECK-SAME: tensor<*xf32>
    return %0 : tensor<*xf32>
  }

  // CHECK-LABEL: func @with_graph_and_islands
  // CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf.resource<tensor<4xf32>>>
  // CHECK-SAME: -> tensor<4xf32>
  func @with_graph_and_islands(%arg0: tensor<!tf.resource<tensor<4xf32>>>) -> tensor<*xf32> {
    %graph = tf_executor.graph {
      %island:2 = tf_executor.island {
        // CHECK: %[[ID_0:.*]] = "tf.IdentityN"(%[[ARG_0]])
        %id0 = "tf.IdentityN"(%arg0)
          : (tensor<!tf.resource<tensor<4xf32>>>) -> tensor<!tf.resource<tensor<4xf32>>>
        // CHECK-NEXT: %[[READ_0:.*]] = "tf.ReadVariableOp"(%[[ID_0]])
        // CHECK-SAME: (tensor<!tf.resource<tensor<4xf32>>>) -> tensor<4xf32>
        %read = "tf.ReadVariableOp"(%id0) : (tensor<!tf.resource<tensor<4xf32>>>) -> tensor<*xf32>
        // CHECK-NEXT: tf_executor.yield %[[READ_0]] : tensor<4xf32>
        tf_executor.yield %read : tensor<*xf32>
      }
      // CHECK: tf_executor.fetch
      // CHECK-SAME: tensor<4xf32>
      tf_executor.fetch %island#0 : tensor<*xf32>
    }
    // CHECK: return
    // CHECK-SAME: tensor<4xf32>
    return %graph : tensor<*xf32>
  }

  // CHECK-LABEL: func @next_iteration_user
  func @next_iteration_user(%arg0: tensor<32x?x256x4xf32>) -> tensor<?x?x?xf32> {
    %0 = tf_executor.graph {
      // CHECK: tf_executor.NextIteration.Source
      // CHECK-SAME: : tensor<32x?x4xf32>
      %1:3 = tf_executor.NextIteration.Source : tensor<?x?x?xf32>
      %out, %c_out = tf_executor.island {
        %dims = "tf.Const"() {value = dense<[32, -1, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
        // CHECK: "tf.Reshape"
        // CHECK-SAME: -> tensor<32x?x4xf32>
        %reshape = "tf.Reshape"(%arg0, %dims) : (tensor<32x?x256x4xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
        // CHECK: tf_executor.yield
        // CHECK-SAME: : tensor<32x?x4xf32>
        tf_executor.yield %reshape : tensor<?x?x?xf32>
      }
      // CHECK: tf_executor.NextIteration.Sink
      // CHECK-SAME: : tensor<32x?x4xf32>
      tf_executor.NextIteration.Sink[%1#1] %out : tensor<?x?x?xf32>
      tf_executor.fetch %1#0 : tensor<?x?x?xf32>
    }
    return %0 : tensor<?x?x?xf32>
  }

  // Check that supported tf_executor ops can receive data from ops on which
  // shape inference has inferred the result types, without throwing any errors.
  // CHECK-LABEL: func @supported_tf_executor_users
  func @supported_tf_executor_users(%arg0: tensor<32x?x256x4xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<i1>, %arg3: tensor<i32>) -> tensor<?x?x?xf32> {
    %0 = tf_executor.graph {
      %island:3 = tf_executor.island {
        %dims = "tf.Const"() {value = dense<[32, -1, 4]> : tensor<3xi32>} : () -> tensor<3xi32>
        %reshape = "tf.Reshape"(%arg0, %dims) : (tensor<32x?x256x4xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
        %cast = "tf.Cast"(%arg2) : (tensor<i1>) -> tensor<*xi1>
        tf_executor.yield %reshape, %cast : tensor<?x?x?xf32>, tensor<*xi1>
      }
      // CHECK: tf_executor.Merge
      // CHECK-SAME: : (tensor<32x?x4xf32>, tensor<?x?x?xf32>) ->
      // CHECK: tf_executor.Switch
      // CHECK-SAME: : (tensor<32x?x4xf32>, tensor<i1>) ->
      // CHECK: tf_executor.SwitchN
      // CHECK-SAME: : tensor<?x?x?xf32>
      // CHECK: tf_executor.Enter
      // CHECK-SAME: : (tensor<32x?x4xf32>) ->
      // CHECK: tf_executor.Exit
      // CHECK-SAME: : tensor<?x?x?xf32>
      // CHECK: tf_executor.LoopCond
      // CHECK-SAME: : tensor<*xi1>
      %merge:3 = "tf_executor.Merge"(%island#0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, tensor<i32>, !tf_executor.control)
      %switch:3 = "tf_executor.Switch"(%island#0, %arg2) : (tensor<?x?x?xf32>, tensor<i1>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, !tf_executor.control)
      %switchn:3 = "tf_executor.SwitchN"(%island#0, %arg3) {num_outs = 2} : (tensor<?x?x?xf32>, tensor<i32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>, !tf_executor.control)
      %enter:2 = "tf_executor.Enter"(%island#0) { frame_name = "frame"} : (tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, !tf_executor.control)
      %exit:2 = "tf_executor.Exit"(%island#0) : (tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>, !tf_executor.control)
      %loop_cond:2 = "tf_executor.LoopCond" (%island#1) : (tensor<*xi1>) -> (tensor<*xi1>, !tf_executor.control)
      tf_executor.fetch %enter#0 : tensor<?x?x?xf32>
    }
    return %0 : tensor<?x?x?xf32>
  }

  // Tests that tensor_cast result shapes are refined.
  // CHECK-LABEL: func @tensor_cast_refine
  func @tensor_cast_refine(%arg0: tensor<4xi32>) -> (tensor<*xi32>) {
    // CHECK: tensor_cast
    // CHECK-SAME: tensor<4xi32> to tensor<4xi32>
    %0 = tensor_cast %arg0 : tensor<4xi32> to tensor<*xi32>
    return %0 : tensor<*xi32>
  }

  // CHECK-LABEL: func @while_variant
  // CHECK-SAME: -> tensor<!tf.variant<tensor<16x1xf32>>>
  func @while_variant(%arg0: tensor<!tf.variant<tensor<16x1xf32>>>) -> tensor<!tf.variant> {
    // CHECK: tf.While
    // CHECK-SAME: -> tensor<!tf.variant<tensor<16x1xf32>>>
    %0 = "tf.While"(%arg0) {cond = @variant_cond_func, body = @variant_body_func, is_stateless = true} : (tensor<!tf.variant<tensor<16x1xf32>>>) -> tensor<!tf.variant>
    // CHECK: tf.ZerosLike
    // CHECK-SAME: -> tensor<!tf.variant<tensor<16x1xf32>>>
    %1 = "tf.ZerosLike"(%0) : (tensor<!tf.variant>) -> tensor<!tf.variant>
    // CHECK: tf.Identity
    // CHECK-SAME: -> tensor<!tf.variant<tensor<16x1xf32>>>
    %2 = "tf.Identity"(%1) : (tensor<!tf.variant>) -> tensor<!tf.variant>
    return %2 : tensor<!tf.variant>
  }
  // CHECK-LABEL: func @variant_cond_func
  func @variant_cond_func(%arg0: tensor<!tf.variant<tensor<16x1xf32>>>) -> tensor<i1> {
    %0 = "tf._SomeOp"() : () -> tensor<i1>
    return %0 : tensor<i1>
  }
  // CHECK-LABEL: func @variant_body_func
  func @variant_body_func(%arg0: tensor<!tf.variant<tensor<16x1xf32>>>) -> tensor<!tf.variant<tensor<16x1xf32>>> {
    return %arg0 : tensor<!tf.variant<tensor<16x1xf32>>>
  }

  // Test propagation from called functions to the call site.
  // CHECK-LABEL: func @stateful_partitioned_call(
  // CHECK-SAME: -> tensor<20xi32>
  func @stateful_partitioned_call(%arg0: tensor<20xi32>) -> tensor<*xi32> {
    // CHECK: tf.PartitionedCall
    // CHECK-SAME: (tensor<20xi32>) -> tensor<20xi32>
    %0 = "tf.PartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @a_called_func} : (tensor<20xi32>) -> (tensor<*xi32>)
    // CHECK: tf.StatefulPartitionedCall
    // CHECK-SAME: (tensor<20xi32>) -> tensor<20xi32>
    %1 = "tf.StatefulPartitionedCall"(%arg0) {config = "", config_proto = "", executor_type = "", f = @stateful_partitioned_call_func} : (tensor<20xi32>) -> (tensor<*xi32>)
    return %0 : tensor<*xi32>
  }
  func @a_called_func(%arg0: tensor<?xi32>) -> (tensor<?xi32>) {
    return %arg0 : tensor<?xi32>
  }
  func @stateful_partitioned_call_func(%arg0: tensor<?xi32>) -> (tensor<?xi32>) {
    return %arg0 : tensor<?xi32>
  }

  // Test propagation involving const values across caller and callee.
  func @partitioned_call_const(%arg0 : tensor<6xf32>) -> tensor<*xf32> {
    %0 = "tf.Const"() {value = dense<[3, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tf.PartitionedCall"(%0) {config = "", config_proto = "", executor_type = "", f = @partitioned_call_func_const} : (tensor<2xi32>) -> (tensor<2xi32>)
    // CHECK: "tf.Reshape"
    // CHECK-SAME: tensor<3x2xf32>
    %2 = "tf.Reshape"(%arg0, %1) : (tensor<6xf32>, tensor<2xi32>) -> tensor<*xf32>
    return %2 : tensor<*xf32>
  }

  // CHECK-LABEL: func @partitioned_call_func_const
  func @partitioned_call_func_const(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }

  // CHECK-LABEL: func @tensor_list_refine
  func @tensor_list_refine() {
    tf_executor.graph {
      %control = tf_executor.island {
        %0 = "tf.Const"() {device = "", value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
        %1 = "tf.Const"() {device = "", value = dense<3> : tensor<i32>} : () -> tensor<i32>
        // CHECK: TensorListReserve{{.*}}-> tensor<!tf.variant<tensor<2x2x!tf.variant>>>
        %2 = "tf.TensorListReserve"(%0, %1) {device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<*x!tf.variant>>>
        // CHECK: TensorListReserve{{.*}}-> tensor<!tf.variant<tensor<2x2xf32>>>
        %3 = "tf.TensorListReserve"(%0, %1) {device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<*xf32>>>
        %4 = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
        %5 = "tf.Const"() {device = "", value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : () -> tensor<2x2xf32>
        // CHECK: tf.TensorListSetItem{{.*}}: (tensor<!tf.variant<tensor<2x2xf32>>>, tensor<i32>, tensor<2x2xf32>) -> tensor<!tf.variant<tensor<2x2xf32>>>
        %6 = "tf.TensorListSetItem"(%3, %4, %5) {device = ""} : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>, tensor<2x2xf32>)-> tensor<*x!tf.variant>
        %7 = "tf.Const"() {device = "", value = dense<-1> : tensor<i32>} : () -> tensor<i32>
        // CHECK: tf.TensorListStack{{.*}}: (tensor<!tf.variant<tensor<2x2xf32>>>, tensor<i32>) -> tensor<?x2x2xf32>
        %8 = "tf.TensorListStack"(%6, %7) {device = "", num_elements = -1 : i64} : (tensor<*x!tf.variant>, tensor<i32>) -> tensor<*xf32>
        tf_executor.yield
      }
      tf_executor.fetch
    }
    return
  }

  // CHECK-LABEL: dont_update_for_ref
  func @dont_update_for_ref() -> () {
    // CHECK: () -> tensor<4x!tf.f32ref>
    %11 = "tf.VariableV2"() {container = "", device = "", shape = #tf.shape<4>, shared_name = ""} : () -> tensor<4x!tf.f32ref>
    // CHECK: (tensor<4x!tf.f32ref>) -> tensor<4xf32>
    %12 = "tf.Identity"(%11) {device = ""} : (tensor<4x!tf.f32ref>) -> tensor<4xf32>
    // CHECK: (tensor<4xf32>) -> tensor<4xf32>
    %13 = "tf.Neg"(%12) {device = ""} : (tensor<4xf32>) -> tensor<4xf32>
    return
  }

  // CHECK-LABEL: operand_as_shape
  func @operand_as_shape(%18: tensor<i32>, %39: tensor<1x4x4x32xf32>) -> () {
    %cst_5 = constant dense<512> : tensor<i32>
    %19 = "tf.Pack"(%18, %cst_5) {N = 2 : i64, T = i32, axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
    // CHECK: -> tensor<1x512xf32>
    %40 = "tf.Reshape"(%39, %19) {T = f32, Tshape = i32, device = ""} : (tensor<1x4x4x32xf32>, tensor<2xi32>) -> tensor<?x?xf32>
   return
  }

  // CHECK-LABEL: const_fold
  func @const_fold() -> () {
    // CHECK: tf.Const
    // CHECK-SAME: () -> tensor<4xi32>
    %0 = "tf.Const"() {value = dense<[200, 26, 26, 32]> : tensor<4xi32>} : () -> tensor<*xi32>
    // CHECK: tf.Const
    // CHECK-SAME: () -> tensor<4xi32>
    %1 = "tf.Const"() {value = dense<[200, 26, 26, 32]> : tensor<4xi32>} : () -> tensor<*xi32>
    // CHECK: tf.Add
    // CHECK-SAME: (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    %2 = "tf.Add"(%0, %1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    return
  }
}
