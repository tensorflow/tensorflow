// RUN: tf-opt %s | tf-opt | FileCheck %s

// Tests printer for tf_executor.island "wraps" short form.

// CHECK-LABEL: func @island_wrap_print
func @island_wrap_print(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island wraps "tf.IdentityN"
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %1#0, %1#1 : tensor<i32>, tensor<f32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @island_no_wrap_print_mismatched_results
func @island_no_wrap_print_mismatched_results(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %1#1, %1#0 : tensor<f32>, tensor<i32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @island_no_wrap_print_mismatched_op_location
func @island_no_wrap_print_mismatched_op_location(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc(unknown)
      tf_executor.yield %1#0, %1#1 : tensor<i32>, tensor<f32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @island_no_wrap_print_mismatched_yield_location
func @island_no_wrap_print_mismatched_yield_location(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %1#0, %1#1 : tensor<i32>, tensor<f32> loc(unknown)
    } loc("identity@some_function")
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: func @island_no_wrap_print_multiple_ops
func @island_no_wrap_print_multiple_ops(%arg0: tensor<i32>, %arg1: tensor<f32>) {
  tf_executor.graph {
    // CHECK: tf_executor.island
    // CHECK-NOT: wraps
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      %2:2 = "tf.IdentityN"(%1#0, %1#1) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>) loc("identity@some_function")
      tf_executor.yield %2#0, %2#1 : tensor<i32>, tensor<f32> loc("identity@some_function")
    } loc("identity@some_function")
    tf_executor.fetch
  }
  return
}
