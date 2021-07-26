// RUN: tf-tfrt-opt -tfrt-reorder-tf-assert %s | FileCheck %s

// CHECK-LABEL: @reorder_assert
func @reorder_assert(%key0: tensor<!tf.string>, %key1: tensor<!tf.string>) -> (tensor<i64>, tensor<i64>) {
  %error_message = "tf.Const"() {value = dense<"error"> : tensor<!tf.string>} : () -> tensor<!tf.string>
  %default = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %handle = "tf.HashTableV2"() {container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", key_dtype = !tf.string, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>

  // CHECK: tf.LookupTableFindV2
  // CHECK-NOT: tf.Assert
  // CHECK: tf.LookupTableFindV2
  // CHECK: tf.Assert

  %value0 = "tf.LookupTableFindV2"(%handle, %key0, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf.resource>, tensor<!tf.string>, tensor<i64>) -> tensor<i64>
  %cond = "tf.Equal"(%value0, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  "tf.Assert"(%cond, %error_message) {device = "/job:localhost/replica:0/task:0/device:CPU:0", summarize = 3 : i64} : (tensor<i1>, tensor<!tf.string>) -> ()
  %value1 = "tf.LookupTableFindV2"(%handle, %key1, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf.resource>, tensor<!tf.string>, tensor<i64>) -> tensor<i64>
  return %value0, %value1 : tensor<i64>, tensor<i64>
}

func private @else_branch(%arg0: tensor<i1>) -> tensor<i1> {
  %cst = "tf.Const"() {value = dense<"Empty SparseTensor with shape"> : tensor<!tf.string>} : () -> tensor<!tf.string>
  "tf.Assert"(%arg0, %cst) {device = "/job:localhost/replica:0/task:0/device:CPU:0", summarize = 3 : i64} : (tensor<i1>, tensor<!tf.string>) -> ()
  return %arg0 : tensor<i1>
}

func private @then_branch(%arg0: tensor<i1>) -> tensor<i1> {
  return %arg0 : tensor<i1>
}

// CHECK-LABEL: @reorder_assert_only_if
func @reorder_assert_only_if(%key0: tensor<!tf.string>, %key1: tensor<!tf.string>) -> (tensor<i64>, tensor<i64>) {
  %error_message = "tf.Const"() {value = dense<"error"> : tensor<!tf.string>} : () -> tensor<!tf.string>
  %default = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %handle = "tf.HashTableV2"() {container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", key_dtype = !tf.string, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf.resource>

  // CHECK: tf.LookupTableFindV2
  // CHECK-NOT: tf.If
  // CHECK: tf.LookupTableFindV2
  // CHECK: tf.If

  %value0 = "tf.LookupTableFindV2"(%handle, %key0, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf.resource>, tensor<!tf.string>, tensor<i64>) -> tensor<i64>
  %cond = "tf.Equal"(%value0, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %unused = "tf.If"(%cond, %cond) {device = "/job:localhost/replica:0/task:0/device:CPU:0", else_branch = @else_branch, is_stateless = false, then_branch = @then_branch} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %value1 = "tf.LookupTableFindV2"(%handle, %key1, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf.resource>, tensor<!tf.string>, tensor<i64>) -> tensor<i64>
  return %value0, %value1 : tensor<i64>, tensor<i64>
}
