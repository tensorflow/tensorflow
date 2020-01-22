// RUN: tf-opt %s -tf-executor-prepare-export | FileCheck %s --dump-input-on-failure

// Checks empty tf_executor.island ops are populated with tf.NoOp/tf.Identity/
// tf.IdentityN ops depending on the number of data results the
// tf_executor.island has.

// CHECK-LABEL: empty_island_no_data_results
func @empty_island_no_data_results() {
  tf_executor.graph {
    %0 = tf_executor.island {
      // CHECK: "tf.NoOp"
      tf_executor.yield
    }
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: empty_island_single_data_result
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<*xf32>)
func @empty_island_single_data_result(%arg0: tensor<*xf32>) {
  tf_executor.graph {
    %0:2 = tf_executor.island {
      // CHECK: %[[IDENTITY:.*]] = "tf.Identity"
      // CHECK-SAME: (%[[ARG_0]])
      // CHECK: tf_executor.yield %[[IDENTITY]]
      tf_executor.yield %arg0 : tensor<*xf32>
    }
    tf_executor.fetch
  }
  return
}

// CHECK-LABEL: empty_island_multiple_data_results
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<*xf32>, %[[ARG_1:.*]]: tensor<*xi32>)
func @empty_island_multiple_data_results(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      // CHECK: %[[IDENTITY_N:.*]]:2 = "tf.IdentityN"
      // CHECK-SAME: (%[[ARG_0]], %[[ARG_1]])
      // CHECK: tf_executor.yield %[[IDENTITY_N]]#0, %[[IDENTITY_N]]#1
      tf_executor.yield %arg0, %arg1 : tensor<*xf32>, tensor<*xi32>
    }
    tf_executor.fetch
  }
  return
}
