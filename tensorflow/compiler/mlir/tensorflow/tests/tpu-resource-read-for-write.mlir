// RUN: tf-opt -tf-tpu-resource-read-for-write %s | FileCheck %s --dump-input=always

// CHECK-LABEL: func @write_only_resource
// CHECK-SAME: ([[ARG0:%.*]]: tensor<i32>, [[ARG1:%.*]]: tensor<f32>, [[ARG2:%.*]]: tensor<*x!tf_type.resource<tensor<i32>>>)
func.func @write_only_resource(%arg0: tensor<i32>, %arg1: tensor<f32>, %arg2: tensor<*x!tf_type.resource<tensor<i32>>>) {
  // CHECK-NEXT: [[READ:%.*]] = "tf.ReadVariableOp"([[ARG2]])
  // CHECK-NEXT: [[CLUSTER:%.*]]:2 = "tf_device.cluster_func"([[ARG0]], [[ARG1]], [[READ]])
  // CHECK-SAME: _replication_info = "write", _xla_compile_device_type = "TPU"
  %0:2 = "tf_device.cluster_func"(%arg0, %arg1) {_replication_info = "write", _xla_compile_device_type = "TPU", func = @write_func} : (tensor<i32>, tensor<f32>) -> (tensor<f32>, tensor<i32>)
  // CHECK-NEXT: "tf.AssignVariableOp"([[ARG2]], [[CLUSTER]]#1)
  "tf.AssignVariableOp"(%arg2, %0#1) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  // CHECK-NEXT: return
  func.return
}

// CHECK-LABEL: func @write_func
// CHECK-SAME: ({{%.*}}: tensor<i32>, {{%.*}}: tensor<f32>, {{%.*}}: tensor<i32>) -> (tensor<f32>, tensor<i32>)
func.func @write_func(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
  func.return %arg1, %arg0 : tensor<f32>, tensor<i32>
}

// CHECK-LABEL: func @read_write_resource
func.func @read_write_resource(%arg0: tensor<i32>, %arg1: tensor<f32>, %arg2: tensor<*x!tf_type.resource<tensor<i32>>>) {
  // CHECK-COUNT-1: tf.ReadVariableOp
  %0 = "tf.ReadVariableOp"(%arg2) : (tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %1:2 = "tf_device.cluster_func"(%arg0, %arg1, %0) {_xla_compile_device_type = "TPU", _replication_info = "read_write", func = @read_write_func} : (tensor<i32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
  "tf.AssignVariableOp"(%arg2, %1#1) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}

// CHECK-LABEL: func @read_write_func
// CHECK-SAME: ({{%.*}}: tensor<i32>, {{%.*}}: tensor<f32>) -> (tensor<f32>, tensor<i32>)
func.func @read_write_func(%arg0: tensor<i32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
  func.return %arg1, %arg0 : tensor<f32>, tensor<i32>
}

// CHECK-LABEL: func @multiple_write_resource
func.func @multiple_write_resource(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i32>>>) {
  // CHECK-NOT: tf.ReadVariableOp
  %0:2 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "multiple_write", func = @multiple_write_func} : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
  "tf.AssignVariableOp"(%arg1, %0#0) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  "tf.AssignVariableOp"(%arg1, %0#1) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return
}

// CHECK-LABEL: func @multiple_write_func
// CHECK-SAME: ({{%.*}}: tensor<i32>) -> (tensor<i32>, tensor<i32>)
func.func @multiple_write_func(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %arg0, %arg0 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func @multiple_result_user
func.func @multiple_result_user(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource<tensor<i32>>>) -> tensor<i32> {
  // CHECK-NOT: tf.ReadVariableOp
  %0 = "tf_device.cluster_func"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "multiple_uses", func = @multiple_result_user_func} : (tensor<i32>) -> tensor<i32>
  "tf.AssignVariableOp"(%arg1, %0) : (tensor<*x!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()
  func.return %0 : tensor<i32>
}

// CHECK-LABEL: func @multiple_result_user_func
// CHECK-SAME: ({{%.*}}: tensor<i32>) -> tensor<i32>
func.func @multiple_result_user_func(%arg0: tensor<i32>) -> tensor<i32> {
  func.return %arg0 : tensor<i32>
}
