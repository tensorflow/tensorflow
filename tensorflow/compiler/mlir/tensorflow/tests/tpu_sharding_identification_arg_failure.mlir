// RUN: tf-opt %s -tf-tpu-sharding-identification | FileCheck %s

// CHECK-LABEL: func @check_arg_sharding_errors
func @check_arg_sharding_errors(%arg0: tensor<1x2x3xi32>) {
  // CHECK:      tf_device.cluster_func
  // CHECK-SAME: use_spmd_for_xla_partitioning = false
  "tf_device.cluster_func"(%arg0) {func = @func} : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
  return
}

func @func(%arg0: tensor<1x2x3xi32>) -> tensor<1x2x3xi32> {
  // Use a four dimension sharding (devices=[1,1,1,1]0)
  // Since the input tensor only has three dimensions, we expect this to fail.
  %0 = "tf.XlaSharding"(%arg0) { _XlaSharding = "\08\03\1A\04\01\01\01\01\22\01\00" } : (tensor<1x2x3xi32>) -> tensor<1x2x3xi32>
  %1 = "tf.A"(%0) : (tensor<1x2x3xi32>) -> (tensor<1x2x3xi32>)
  return %1: tensor<1x2x3xi32>
}
