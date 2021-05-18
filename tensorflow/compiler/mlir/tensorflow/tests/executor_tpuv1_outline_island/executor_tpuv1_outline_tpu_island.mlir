// RUN: tf-opt %s -tf-executor-tpu-v1-island-outlining | FileCheck %s


// CHECK-LABEL: @func0
func @func0(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
// CHECK: island
// CHECK: PartitionedCall
// CHECK-SAME: @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func0
    %1:2 = tf_executor.island {
     "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.opB"() : () -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }
    tf_executor.fetch %2#0 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// CHECK-LABEL: @func2
func @func2(%arg0 : tensor<i1>) -> tensor<i1> {
  %0 = tf_executor.graph {
    %1:2 = tf_executor.island {
      %4 = "tf.opB"() : () -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }
// CHECK: island
// CHECK: "tf.opB"
// CHECK: island
// CHECK: PartitionedCall
// CHECK-SAME: @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func1
    %2:3 = tf_executor.island {
     "tf.TPUReplicateMetadata"() {_tpu_replicate = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opA"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.SomeOp"(%arg0, %1#0) : (tensor<i1>, tensor<f32>) -> tensor<i32>
      tf_executor.yield %4, %5 : tensor<i1>, tensor<i32>
    }
    tf_executor.fetch %2#0 : tensor<i1>
  }
  return %0 : tensor<i1>
}

// CHECK: module
// CHECK-SAME: @_tpu_v1_compat_outlined
// CHECK-LABEL: func nested @_tpu_v1_compat_outlined_func0(%arg0: tensor<i1>) -> tensor<i1>
// CHECK-NEXT: tf.TPUReplicateMetadata
// CHECK-NEXT: tf.opA

// CHECK-LABEL: func nested @_tpu_v1_compat_outlined_func1(%arg0: tensor<i1>, %arg1: tensor<f32>) -> (tensor<i1>, tensor<i32>)
// CHECK-NEXT: tf.TPUReplicateMetadata
// CHECK-NEXT: tf.opA
// CHECK-NEXT: tf.opA
// CHECK-NEXT: tf.SomeOp
