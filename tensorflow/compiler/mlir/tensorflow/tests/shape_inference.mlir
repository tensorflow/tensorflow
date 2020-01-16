// RUN: tf-opt %s -tf-shape-inference -verify-diagnostics | FileCheck %s -dump-input=fail -color

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 130 : i32}} {
// CHECK-LABEL: func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32>
  func @main(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<*xi32> {
 // CHECK: %[[ARG0:.*]] = "tf.Cast"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
 // CHECK: %[[ARG1:.*]] = "tf.Cast"(%arg1) : (tensor<1xi32>) -> tensor<1xi32>
 // CHECK: %[[RESULT:.*]] = "tf.AddV2"(%[[ARG0]], %[[ARG1]]) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
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
// CHECK: %[[UNKNOWN:.*]] = "unknown.A"(%[[CAST]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK: return %[[UNKNOWN]] : tensor<*xf32>
    %0 = "tf.Mul"(%arg0, %arg1) : (tensor<1xf32>, tensor<10xf32>) -> tensor<*xf32>
    %1 = "tf.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = "unknown.A"(%1) : (tensor<*xf32>) -> tensor<*xf32>
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

// Tests the case where an inference opportunity relies on folding.

// CHECK-LABEL: func @simple_folding
  func @simple_folding(%arg0: tensor<1x1x1x1xi32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32> {
// CHECK: %[[CST:.*]] = "tf.Const"{{.*}} {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK: %[[CONV:.*]] = "tf.Conv2DBackpropInput"(%[[CST]]
// CHECK-SAME: (tensor<4xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK: return %[[CONV]] : tensor<1x1x1x1xf32>
    %0 = "tf.Shape"(%arg0) : (tensor<1x1x1x1xi32>) -> tensor<4xi32>
    %1 = "tf.Conv2DBackpropInput"(%0, %arg1, %arg1) {
      padding = "VALID", strides = [1, 1, 1, 1]
    } : (tensor<4xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
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
    %0 = "tf.If"(%arg0, %arg1) {Tcond = i1, Tin = ["tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _xla_propagate_compile_time_consts = true, device = "", else_branch = @if_else_branch, is_stateless = true, name = "if", output_shapes = ["tfshape$"], then_branch = @if_then_branch} : (tensor<i1>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
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
    // CHECK "tf.While"
    // CHECK-SAME (tensor<4xf32>, tensor<!tf.resource<tensor<4xf32>>>, tensor<!tf.resource<tensor<*xf32>>>) -> (tensor<4xf32>, tensor<!tf.resource<tensor<4xf32>>>, tensor<!tf.resource<tensor<*xf32>>>)
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

  // CHECK-LABEL: func @invalid_function_reused_by_control_flows
  func @invalid_function_reused_by_control_flows(%arg0: tensor<i1>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
	  // expected-warning @+1 {{unable to refine shape}}
    %0 = "tf.If"(%arg0, %arg1) {Tcond = i1, Tin = ["tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _xla_propagate_compile_time_consts = true, device = "", else_branch = @reused_if_else_branch, is_stateless = true, name = "if", output_shapes = ["tfshape$"], then_branch = @reused_if_then_branch} : (tensor<i1>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
	  // expected-warning @+1 {{unable to refine shape}}
    %1 = "tf.If"(%arg0, %0) {Tcond = i1, Tin = ["tfdtype$DT_FLOAT"], Tout = ["tfdtype$DT_FLOAT"], _xla_propagate_compile_time_consts = true, device = "", else_branch = @reused_if_else_branch, is_stateless = true, name = "if", output_shapes = ["tfshape$"], then_branch = @reused_if_then_branch} : (tensor<i1>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
    return %0 : tensor<1x2x3xf32>
  }

  // CHECK-LABEL: func @reused_if_then_branch
  // CHECK-SAME: (%arg0: tensor<*xf32>) -> tensor<*xf32>
	// expected-error @+1 {{expected control flow function reused_if_then_branch to have exactly 1 use}}
  func @reused_if_then_branch(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    // CHECK: return
    // CHECK-SAME: tensor<*xf32>
    return %arg0 : tensor<*xf32>
  }

  // CHECK-LABEL: func @reused_if_else_branch
  // CHECK-SAME: (%arg0: tensor<*xf32>) -> tensor<*xf32>
	// expected-error @+1 {{expected control flow function reused_if_else_branch to have exactly 1 use}}
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
}
