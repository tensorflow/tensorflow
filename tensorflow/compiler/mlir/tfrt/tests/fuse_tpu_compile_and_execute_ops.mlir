// RUN: tf-tfrt-opt -verify-diagnostics -split-input-file -tfrt-fuse-tpu-compile-and-execute-ops %s | FileCheck %s --dump-input=fail --dump-input-filter=all

module attributes {tf_saved_model.semantics} {

// Test fusing _TPUCompileMlirOp and TPUExecuteOp into TPUCompileMlirAndExecuteOp.

// CHECK-LABEL: func private @test_fuse_tpu_ops
func private @test_fuse_tpu_ops(%arg0: tensor<*xi32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<*xi32> {
  // CHECK-NOT: tf._TPUCompileMlirOp
  // CHECK-NOT: tf.TPUCompileSucceededAssert
  // CHECK-NOT: tf.TPUExecuteOp

  // CHECK-NEXT: %0 = "tf.ReadVariableOp"(%arg1)
  // CHECK:      [[exec_result:%.*]] = "tf.TPUCompileMlirAndExecute"(%arg0, %0) {metadata = "metadata", mlir_module = "mlir_module"} : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  // CHECK-NEXT: return [[exec_result]] : tensor<*xi32>

  %0 = "tf.ReadVariableOp"(%arg1) {device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<*xi32>
  %1 = "tf.Shape"(%arg0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %2 = "tf.Shape"(%0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %compilation_status, %program = "tf._TPUCompileMlir"(%1, %2) {device = "/CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : (tensor<?xi64>, tensor<?xi64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  %3 = "tf.TPUExecute"(%arg0, %0, %program) {device = "/TPU:0"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> tensor<*xi32>
  return %3 : tensor<*xi32>
}

}

// -----

module attributes {tf_saved_model.semantics} {

// Test models using Outside Compilation error out gracefully instead of crashing

func private @test_error_out_for_outside_compilation(%arg0: tensor<*xi32>, %arg1: tensor<*x!tf_type.resource>) -> tensor<*xi32> {
  %0 = "tf.ReadVariableOp"(%arg1) {device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<*xi32>
  %1 = "tf.Shape"(%arg0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %2 = "tf.Shape"(%0) {device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  // expected-error@+1 {{Some op other than TPUExecuteOp is taking _TPUCompileMlirOp as input. This is probably due to outside compilation being used in the model.}}
  %compilation_status, %program = "tf._TPUCompileMlir"(%1, %2) {device = "/CPU:0", metadata = "metadata", mlir_module = "mlir_module"} : (tensor<?xi64>, tensor<?xi64>) -> (tensor<!tf_type.string>, tensor<3x!tf_type.string>)
  "tf.TPUCompileSucceededAssert"(%compilation_status) {device = "/CPU:0"} : (tensor<!tf_type.string>) -> ()
  "tf._XlaSendFromHost"(%arg0, %0, %program) {_xla_has_host_transfer = true, device = "/job:localhost/replica:0/task:0/device:CPU:0", device_ordinal = 0 : i64, key = "host_compute_channel_0_retvals"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> ()
  %3 = "tf.TPUExecute"(%arg0, %0, %program) {device = "/TPU:0"} : (tensor<*xi32>, tensor<*xi32>, tensor<3x!tf_type.string>) -> tensor<*xi32>
  return %3 : tensor<*xi32>
}

}
