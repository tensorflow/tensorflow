// RUN: dtensor-opt %s -dtensor-set-default-sharding | FileCheck %s

// CHECK-LABEL: func @empty_func
func.func @empty_func(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>)  -> (tensor<?xi32>, tensor<?xi32>) {
  func.return %arg0, %arg1 : tensor<?xi32>, tensor<?xi32>
}

// CHECK-LABEL: func @check_default_sharding_set
func.func @check_default_sharding_set() {
  %0 = "tf.A"() : () -> tensor<?xi32>
  %1 = "tf.B"() : () -> tensor<?xi32>
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: _tpu_replicate = "cluster0"
  // CHECK-SAME: input_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"],
  // CHECK-SAME: output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", "\08\01\1A\01\01\22\01\00"]
  %2, %3 = "tf_device.cluster_func"(%0, %1) {_tpu_replicate = "cluster0", func = @empty_func} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  func.return
}

// CHECK-LABEL: func @check_non_tpu_cluster_func_ignored
func.func @check_non_tpu_cluster_func_ignored() {
  %0 = "tf.A"() : () -> tensor<?xi32>
  %1 = "tf.B"() : () -> tensor<?xi32>
  // CHECK:      tf_device.cluster_func
  // CHECK-NOT: _tpu_replicate = "cluster0"
  // CHECK-NOT: input_sharding_configuration
  // CHECK-NOT: output_sharding_configuration
  %2, %3 = "tf_device.cluster_func"(%0, %1) {func = @empty_func} : (tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>)
  func.return
}
