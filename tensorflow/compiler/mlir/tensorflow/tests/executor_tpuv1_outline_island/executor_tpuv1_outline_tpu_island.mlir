// RUN: tf-opt %s -split-input-file -tf-executor-tpu-v1-island-outlining -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @func0
func.func @func0(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
// CHECK: island
// CHECK: PartitionedCall
// CHECK-SAME: @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func0
    %1:2 = tf_executor.island {
     "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.opB"() : () -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }
    tf_executor.fetch %2#0 : tensor<f32>
  }
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: @func2
func.func @func2(%arg0 : tensor<i1>) -> tensor<i1> {
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
     "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      %3 = "tf.opA"(%arg0) : (tensor<i1>) -> tensor<i1>
      %4 = "tf.opA"(%3) : (tensor<i1>) -> tensor<i1>
      %5 = "tf.SomeOp"(%arg0, %1#0) : (tensor<i1>, tensor<f32>) -> tensor<i32>
      tf_executor.yield %4, %5 : tensor<i1>, tensor<i32>
    }
    tf_executor.fetch %2#0 : tensor<i1>
  }
  func.return %0 : tensor<i1>
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

// -----

// Test single-core TPU case (no `_replication_info`, no `TPUReplicateMetadata`).

// CHECK-LABEL: @func3
func.func @func3(%arg0 : tensor<i1>) -> tensor<f32> {
  %0 = tf_executor.graph {
// CHECK: island
// CHECK: PartitionedCall
// CHECK-SAME: @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func0
    %1:2 = tf_executor.island {
     "tf.SomeTpuOp"() {_xla_compile_device_type = "TPU", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      %3 = "tf.SomeOtherTpuOp"(%arg0) {_xla_compile_device_type = "TPU"} : (tensor<i1>) -> tensor<i1>
      tf_executor.yield %3 : tensor<i1>
    }
    %2:2 = tf_executor.island(%1#1) {
      %4 = "tf.opB"() : () -> tensor<f32>
      tf_executor.yield %4 : tensor<f32>
    }
    tf_executor.fetch %2#0 : tensor<f32>
  }
  func.return %0 : tensor<f32>
}

// CHECK: module
// CHECK-SAME: @_tpu_v1_compat_outlined
// CHECK-LABEL: func nested @_tpu_v1_compat_outlined_func0(%arg0: tensor<i1>) -> tensor<i1>
// CHECK-NEXT: tf.SomeTpuOp
// CHECK-NEXT: tf.SomeOtherTpuOp

// -----

// Test single-core TPU case with single-op cluster.

// CHECK-LABEL: @func4
func.func @func4() {
  tf_executor.graph {
// CHECK: island
// CHECK: PartitionedCall
// CHECK-SAME: @_tpu_v1_compat_outlined::@_tpu_v1_compat_outlined_func0
    %1 = tf_executor.island {
     "tf.SomeTpuOp"() {_xla_compile_device_type = "TPU"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: module
// CHECK-SAME: @_tpu_v1_compat_outlined
// CHECK-LABEL: func nested @_tpu_v1_compat_outlined_func0()
// CHECK-NEXT: tf.SomeTpuOp

// -----

// Test that islands inside a function with `_skip_island_outlining = true` are
// skipped from outlining for both single-core and replicated case (i.e., the
// `_tpu_v1_compat_outlined` module must be empty and no `PartitionedCallOp` is
// created). Also check that `_skip_island_outlining` attribute is removed.

// CHECK-LABEL: @func5
// CHECK-NOT: _skip_island_outlining
// CHECK-NOT: PartitionedCallOp
// CHECK: _tpu_v1_compat_outlined {
// CHECK-NEXT: }
func.func @func5() attributes {_skip_island_outlining = true} {
  tf_executor.graph {
    %1 = tf_executor.island {
     "tf.SomeTpuOp"() {_xla_compile_device_type = "TPU"} : () -> ()
      "tf.SomeOtherTpuOp"() {_xla_compile_device_type = "TPU"} : () -> ()
      tf_executor.yield
    }
    %2 = tf_executor.island {
      "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate", device = "device", num_replicas = 1, topology = "topology"} : () -> ()
      "tf.SomeTpuOp"() {_xla_compile_device_type = "TPU", _replication_info = "replicate"} : () -> ()
      "tf.SomeOtherTpuOp"() {_xla_compile_device_type = "TPU", _replication_info = "replicate"} : () -> ()
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}
