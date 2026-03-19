// RUN: tf-opt %s --tf-colocate-tpu-copy-with-dynamic-shape --split-input-file | FileCheck %s

// CHECK-LABEL test_tpu_execute
func.func @test_tpu_execute(%arg0: tensor<!tf_type.string>) {
     %0 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %1 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %2 = builtin.unrealized_conversion_cast to tensor<i32>
     %3 = builtin.unrealized_conversion_cast to tensor<i32>
     // CHECK: TPUCopyWithDynamicShape{{.*}}device = "foobar"
     %4, %5 = "tf.TPUCopyWithDynamicShape"(%0, %1, %2, %3) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
    "tf.TPUExecute"(%4, %arg0) {device = "foobar"} : (tensor<2048xi32>, tensor<!tf_type.string>) -> ()
    return
}

// -----

// CHECK-LABEL test_tpu_execute_and_update_variables
func.func @test_tpu_execute_and_update_variables(%arg0: tensor<2x!tf_type.string>) {
     %0 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %1 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %2 = builtin.unrealized_conversion_cast to tensor<i32>
     %3 = builtin.unrealized_conversion_cast to tensor<i32>
     // CHECK: TPUCopyWithDynamicShape{{.*}}device = "foobar"
     %4, %5 = "tf.TPUCopyWithDynamicShape"(%0, %1, %2, %3) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
    "tf.TPUExecuteAndUpdateVariables"(%4, %arg0) {
        device = "foobar",
        device_var_reads_indices = [],
        device_var_updates_indices = []} : (
                tensor<2048xi32>,
                tensor<2x!tf_type.string>) -> ()
    return
}

// -----

// CHECK-LABEL test_identity
func.func @test_identity(%arg0: tensor<!tf_type.string>) {
     %0 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %1 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %2 = builtin.unrealized_conversion_cast to tensor<i32>
     %3 = builtin.unrealized_conversion_cast to tensor<i32>
     // CHECK: TPUCopyWithDynamicShape{{.*}}device = "foobar"
     %4, %5 = "tf.TPUCopyWithDynamicShape"(%0, %1, %2, %3) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
     %6 = "tf.Identity"(%4) : (tensor<2048xi32>) -> tensor<2048xi32>
    "tf.TPUExecute"(%6, %arg0) {device = "foobar"} : (tensor<2048xi32>, tensor<!tf_type.string>) -> ()
    return
}

// -----

// CHECK-LABEL test_disconnected
func.func @test_disconnected(%arg0: tensor<!tf_type.string>) {
     %0 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %1 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     %2 = builtin.unrealized_conversion_cast to tensor<i32>
     %3 = builtin.unrealized_conversion_cast to tensor<i32>
     // CHECK: TPUCopyWithDynamicShape
     // CHECK-NOT: device = "foobar"
     %4, %5 = "tf.TPUCopyWithDynamicShape"(%0, %1, %2, %3) {operandSegmentSizes = array<i32: 2, 2>} : (tensor<2048xi32>, tensor<2048xi32>, tensor<i32>, tensor<i32>) -> (tensor<2048xi32>, tensor<2048xi32>)
     %6 = builtin.unrealized_conversion_cast to tensor<2048xi32>
     // CHECK: TPUExecute
    "tf.TPUExecute"(%6, %arg0) {device = "foobar"} : (tensor<2048xi32>, tensor<!tf_type.string>) -> ()
    return
}
