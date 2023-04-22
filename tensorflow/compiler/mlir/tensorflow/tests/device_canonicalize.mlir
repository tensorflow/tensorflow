// RUN: tf-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// Test empty launch with no results is folded away.
// CHECK-LABEL: func @empty_launch_no_results
func @empty_launch_no_results() {
  "tf_device.launch"() ( {
    tf_device.return
  }) {device = "device"} : () -> ()
  return
}

// CHECK-NOT: tf_device.launch


// Test empty launch with some results is folded away.
// CHECK-LABEL: func @empty_launch
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<i1>, %[[ARG_1:[a-z0-9]*]]: tensor<i32>)
func @empty_launch(%arg0 : tensor<i1>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i1>) {
  %result:2 = "tf_device.launch"() ( {
    tf_device.return %arg0, %arg1 : tensor<i1>, tensor<i32>
  }) {device = "device"} : () -> (tensor<i1>, tensor<i32>)
  return %result#1, %result#0 : tensor<i32>, tensor<i1>
}

// CHECK-NOT: tf_device.launch
// CHECK: return %[[ARG_1]], %[[ARG_0]]


// CHECK-LABEL: func @eliminate_passthrough_args_cluster_op
func @eliminate_passthrough_args_cluster_op(%arg0 : tensor<!tf.string>, %arg1 : tensor<!tf.string>) -> (tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>) {
  // CHECK: %[[MUL:.*]] = "tf.MyStringConcat"
  %0 = "tf.MyStringConcat"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<!tf.string>
  // CHECK: %[[RESULT:.*]]:2 = "tf_device.cluster"
  %1:4 = "tf_device.cluster"() ( {
    // CHECK: %[[MATCH:.*]] = "tf.MyStringMatch"
    %2 = "tf.MyStringMatch"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<!tf.string>
    // CHECK: %[[PREFIX:.*]] = "tf.IsStringPrefix"
    %3 = "tf.IsStringPrefix"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<!tf.string>
    // CHECK: tf_device.return %[[MATCH]], %[[PREFIX]]
    tf_device.return %arg0, %2, %0, %3 : tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>
  }) : () -> (tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>)

  // CHECK: return %arg0, %[[RESULT]]#0, %[[MUL]], %[[RESULT]]#1
  return %1#0, %1#1, %1#2, %1#3 : tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>, tensor<!tf.string>
}

// Verifies handling op a cluster op with only pass through arguments.
// CHECK-LABEL: func @all_pass_through_args_cluster_op
func @all_pass_through_args_cluster_op(%arg0 : tensor<!tf.string>, %arg1 : tensor<!tf.string>) -> (tensor<!tf.string>, tensor<!tf.string>) {
  // CHECK: {{^ *}}"tf_device.cluster"
  %0:2 = "tf_device.cluster"() ( {
    // CHECK: "tf.Equal"
    %1 = "tf.Equal"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<i1>
    // CHECK: "tf.Assert"
    "tf.Assert"(%1, %arg0) : (tensor<i1>, tensor<!tf.string>) -> ()
    // CHECK: tf_device.return{{$}}
    tf_device.return %arg0, %arg1 : tensor<!tf.string>, tensor<!tf.string>
  }) : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK: return %arg0, %arg1
  return %0#0, %0#1 : tensor<!tf.string>, tensor<!tf.string>
}

// Verifies handling op a cluster op requiring no rewrites.
// CHECK-LABEL: func @canonical_cluster
func @canonical_cluster(%arg0 : tensor<!tf.string>, %arg1 : tensor<!tf.string>) -> (tensor<!tf.string>, tensor<!tf.string>) {
  // CHECK: %[[RESULT:.*]]:2 = "tf_device.cluster"
  %0:2 = "tf_device.cluster"() ( {
    // CHECK: %[[MATCH:.*]] = "tf.MyStringMatch"
    %1 = "tf.MyStringMatch"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<!tf.string>
    // CHECK: %[[PREFIX:.*]] = "tf.IsStringPrefix"
    %2 = "tf.IsStringPrefix"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<!tf.string>
    // CHECK: tf_device.return %[[MATCH]], %[[PREFIX]]
    tf_device.return %1, %2 : tensor<!tf.string>, tensor<!tf.string>
  }) : () -> (tensor<!tf.string>, tensor<!tf.string>)
  return %0#0, %0#1 : tensor<!tf.string>, tensor<!tf.string>
}


// Verifies handling op a cluster op whose results are used for resource assignment.
// CHECK-LABEL: func @cluster_result_for_resource_update
func @cluster_result_for_resource_update(%arg0 : tensor<!tf.string>, %arg1 : tensor<!tf.string>) -> (tensor<!tf.string>, tensor<!tf.string>) {
  %resource = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<*x!tf.string>>>

  // CHECK: %[[RESULT:.*]] = "tf_device.cluster"
  %0:2 = "tf_device.cluster"() ( {
    // CHECK: "tf.Equal"
    %1 = "tf.Equal"(%arg0, %arg1) : (tensor<!tf.string>, tensor<!tf.string>) -> tensor<i1>
    // CHECK: "tf.Assert"
    "tf.Assert"(%1, %arg0) : (tensor<i1>, tensor<!tf.string>) -> ()
    // CHECK: tf_device.return %arg1
    tf_device.return %arg0, %arg1 : tensor<!tf.string>, tensor<!tf.string>
  }) : () -> (tensor<!tf.string>, tensor<!tf.string>)
  // CHECK: "tf.AssignVariableOp"({{.*}}, %[[RESULT]]
  "tf.AssignVariableOp"(%resource, %0#1) {dtype = !tf.string} : (tensor<*x!tf.resource<tensor<*x!tf.string>>>, tensor<!tf.string>) -> ()

  // CHECK: return %arg0, %[[RESULT]]
  return %0#0, %0#1 : tensor<!tf.string>, tensor<!tf.string>
}

// Verifies that i32 pass through arguments are not rewritten.
// CHECK-LABEL: func @eliminate_passthrough_args_cluster_op_i32
func @eliminate_passthrough_args_cluster_op_i32(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK: %[[MUL:.*]] = "tf.Mul"
  %0 = "tf.Mul"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK: %[[RESULT:.*]]:4 = "tf_device.cluster"
  %1:4 = "tf_device.cluster"() ( {
    // CHECK: %[[ADD:.*]] = "tf.AddV2"
    %2 = "tf.AddV2"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK: %[[SUB:.*]] = "tf.Sub"
    %3 = "tf.Sub"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK: tf_device.return %arg0, %[[ADD]], %[[MUL]], %[[SUB]]
    tf_device.return %arg0, %2, %0, %3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
  }) : () -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>)

  // CHECK: return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2, %[[RESULT]]#3
  return %1#0, %1#1, %1#2, %1#3 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>
}
