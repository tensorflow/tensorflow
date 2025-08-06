// RUN: tf-tfrt-opt --split-input-file -pass-pipeline='builtin.module(pre-parallel-tf-to-mlrt{use-tpu-host-allocator-for-inputs=true},tf-mlrt-parallelization{tfrt-cost-threshold=4},tf-to-mlrt)'  %s | FileCheck %s --dump-input=fail --dump-input-filter=all

func.func @callee(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  func.return %arg0: tensor<i32>
}

// CHECK-LABEL: func @batch_function
func.func @batch_function(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: [[batch_result_future:%.*]] = tf_mlrt.batch_function
  // CHECK: [[batch_result:%.*]] = tf_mlrt.await [[batch_result_future]]
  // CHECK-NEXT: [[rendezvous_key_base:%.*]] = tf_mlrt_tpu.compile_and_execute([[batch_result]])
  // CHECK-NEXT: return [[rendezvous_key_base]]
  %0 = "tf.BatchFunction"(%arg0, %arg0) {device = "/device:CPU:0", allowed_batch_sizes = [64], batch_timeout_micros = 1 : i64, batching_queue = "", container = "", f = @callee, max_batch_size = 256 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 1, 1>, shared_name = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %1 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// -----

func.func @executeop_input(%arg0: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK-NOT: tf_mlrt.executeop(
  // CHECK: [[device:%.*]] = tf_mlrt_tpu.get_tpu_host_device
  // CHECK: [[cast:%.*]] = tf_mlrt.executeop.device([[device]]){{.*}}op: \22Cast\22
  // CHECK: [[rendezvous_key_base:%.*]], [[result_future:%.*]] = tf_mlrt_tpu.compile_and_execute([[cast]])
  // CHECK: tf_mlrt.await [[result_future]]
  %0 = "tf.Cast"(%arg0) {__op_key = 0: i32, device = "/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  %1, %2 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<f32>) -> (tensor<i32>, tensor<i32>)
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

// -----

func.func @executeop_side_effecting_input(%arg0: tensor<!tf_type.resource<tensor<4xf32>>>, %indices: tensor<i32>) -> (tensor<i32>) {
  // CHECK-NOT: tf_mlrt.executeop(
  // CHECK: [[device:%.*]] = tf_mlrt_tpu.get_tpu_host_device
  // CHECK: [[var:%.*]] = tf_mlrt.executeop.device([[device]]){{.*}}op: \22ResourceGather\22
  // CHECK: [[rendezvous_key_base:%.*]] = tf_mlrt_tpu.compile_and_execute([[var]])
  %0 = "tf.ResourceGather"(%arg0, %indices) {__op_key = 0: i32, device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<4xf32>>>, tensor<i32>) -> tensor<f32>
  %1 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<f32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// -----

func.func @executeop_input_same_execute_op(%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> (tensor<i32>) {
  // CHECK-NOT: tf_mlrt.executeop(
  // CHECK: [[device:%.*]] = tf_mlrt_tpu.get_tpu_host_device
  // CHECK: [[split:%.*]]:2 = tf_mlrt.executeop.device([[device]])
  // CHECK: tf_mlrt_tpu.compile_and_execute([[split]]#0, [[split]]#1)
  %0, %1 = "tf.Split"(%arg0, %arg1) {__op_key = 0: i32, device = "/device:CPU:0"} : (tensor<i32>, tensor<2xf32>) -> (tensor<f32>, tensor<f32>)
  %2 = "tf.TPUCompileMlirAndExecute"(%0, %1) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 2, 0>, producer_name = "producer_name"} : (tensor<f32>, tensor<f32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

// -----

// Test that inputs are lowered correctly when they form a DAG.

// CHECK-LABEL: executeop_dag
func.func @executeop_dag(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK-NEXT: tf_mlrt_tpu.get_tpu_host_device
  // CHECK-NEXT: tf_mlrt.executeop.device{{.*}}op: \22Cast\22
  // CHECK-NEXT: tf_mlrt_tpu.get_tpu_host_device
  // CHECK-NEXT: tf_mlrt.executeop.device{{.*}}op: \22Relu\22
  // CHECK-NEXT: tf_mlrt_tpu.compile_and_execute
  %0 = "tf.Cast"(%arg0) {__op_key = 0: i32, device = "/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  %1 = "tf.Relu"(%0) {__op_key = 1: i32, device = "/device:CPU:0"} : (tensor<f32>) -> (tensor<f32>)
  %2 = "tf.TPUCompileMlirAndExecute"(%1, %0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 2, 0>, producer_name = "producer_name"} : (tensor<f32>, tensor<f32>) -> tensor<i32>
  func.return %2 : tensor<i32>
}

// -----

func.func @test_fuse_dynamic_dimension_ops(%arg0: tensor<*xi32>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>, %arg4: tensor<*xi32>, %arg5: tensor<?xi64>, %arg6: tensor<?xi64>, %arg7: tensor<?xi64>) -> tensor<*xi32> {
  %0 = "tf.ReadVariableOp"(%arg1) {__op_key = 0: i32, device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<*xi32>
  %1 = "tf.Shape"(%arg0) {__op_key = 1: i32, device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  %2 = "tf.Shape"(%0) {__op_key = 2: i32, device = "/CPU:0"} : (tensor<*xi32>) -> tensor<?xi64>
  // CHECK: [[rendezvous_key_base:%.*]], [[result_future:%.*]] = tf_mlrt_tpu.compile_and_execute
  // CHECK-SAME: constant_operand_indices = array<i32: 2>
  // CHECK-SAME: num_operands = 4
  // CHECK-SAME: operands_with_static_shape = array<i32: 0, 1, 3>
  %rendezvous_key_base, %results = "tf.TPUCompileMlirAndExecute"(%arg0, %2, %0, %1, %arg5, %arg6, %arg7) {operands_with_static_shape = [0 : i32, 1 : i32, 3 : i32], metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 4, 3>, producer_name = "producer_name"} : (tensor<*xi32>, tensor<?xi64>, tensor<*xi32>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) -> (tensor<3x!tf_type.string>, tensor<*xi32>)
  func.return %results : tensor<*xi32>
}

// -----

// Test async output of tf.TPUCompileMlirAndExecute to function is converted

// CHECK-LABEL: @executeop_input_stream_1
// CHECK-SAME: ([[future:%.*]]: !mlrt.future
// CHECK: [[tensor:%.*]] = tf_mlrt.await [[future]]
// CHECK: tf_mlrt.executeop([[tensor]])
// CHECK-SAME: StringFormat

// CHECK-LABEL: @executeop_input
func.func @executeop_input(%arg0: tensor<i32>) -> (tensor<i32>) {
  // CHECK: tf_mlrt.executeop
  %0 = "tf.Cast"(%arg0) {__op_key = 0: i32, device = "/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  // CHECK: [[rendezvous_key_base:%.*]], [[result:%.*]] = tf_mlrt_tpu.compile_and_execute
  %1, %2 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<f32>) -> (tensor<i32>, tensor<i32>)
  %3 = "tf.StringFormat"(%2) {__op_key = 1: i32, device = "/job:localhost/replica:0/task:0/device:CPU:0", placeholder = "{}", strtemplate = "%s", summarize = 3 : i64, template = "Outside compiled {}"} : (tensor<i32>) -> tensor<!tf_type.string>
  "tf.PrintV2"(%3) {__op_key = 2: i32, device = "/job:localhost/replica:0/task:0/device:CPU:0", end = "\0A", output_stream = "stderr"} : (tensor<!tf_type.string>) -> ()
  // CHECK: [[handle:%.*]] = mlrt.async([[result]])
  // CHECK-SAME: (!mlrt.future)
  // CHECK: mlrt.await_handle [[handle]]
  // CHECK: return [[rendezvous_key_base]]
  // CHECK-SAME: !tf_mlrt.tensor
  func.return %1 : tensor<i32>
}

// -----

// Test constant arguments to tf.TPUCompileMlirAndExecute are preserved during parallelization.

// CHECK-LABEL: @preserve_constant_args(
func.func @preserve_constant_args(%arg0: tensor<i32>, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*x!tf_type.resource>, %arg3: tensor<*x!tf_type.resource>) -> (tensor<i32>) {
  // CHECK-NOT: ReadVariableOp
  // CHECK: mlrt.async(
  %v0 = "tf.ReadVariableOp"(%arg1) {__op_key = 0: i32, device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<i32>
  %v1 = "tf.ReadVariableOp"(%arg2) {__op_key = 1: i32, device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<i32>
  // CHECK: [[cast:%.*]] = tf_mlrt.executeop(
  // CHECK-SAME: ReadVariableOp
  %v2 = "tf.ReadVariableOp"(%arg3) {__op_key = 2: i32, device = "/CPU:0"} : (tensor<*x!tf_type.resource>) -> tensor<i32>
  // CHECK: [[cast:%.*]] = tf_mlrt.executeop.device
  // CHECK-SAME: Cast
  %0 = "tf.Cast"(%arg0) {__op_key = 3: i32, device = "/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  // CHECK: tf_mlrt_tpu.compile_and_execute({{%.*}}, [[cast]]
  // CHECK-SAME: constant_operand_indices = array<i32: 1, 3, 4>
  %1, %2 = "tf.TPUCompileMlirAndExecute"(%0, %v1, %0, %v2, %v0, %arg0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 6, 0>, producer_name = "producer_name"} : (tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %2 : tensor<i32>
}

// -----

func.func @executeop_input_async() -> (tensor<i32>, tensor<i32>) {
  // CHECK-NOT: tf_mlrt.executeop(
  // CHECK: [[device:%.*]] = tf_mlrt_tpu.get_tpu_host_device
  // CHECK: [[recv_future:%.*]] = tf_mlrt.async_executeop.device([[device]]){{.*}}op: \22Recv\22
  // CHECK: [[recv:%.*]] = tf_mlrt.await [[recv_future]]
  // CHECK: [[rendezvous_key_base:%.*]], [[result_future:%.*]] = tf_mlrt_tpu.compile_and_execute([[recv]])
  // CHECK: tf_mlrt.await [[result_future]]
  %0 = "tf.Recv"() {__op_key = 0: i32, device = "/device:CPU:0", tensor_name = "tensor", send_device = "/device:CPU:0", send_device_incarnation = 0, recv_device = "/device:CPU:0"} : () -> tensor<f32>
  %1, %2 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<f32>) -> (tensor<i32>, tensor<i32>)
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

// -----

// Test the output from TPU op is properly awaited before its use by map_fn.
// CHECK-LABEL: @main
// CHECK-SAME: ([[input0:%.*]]: !tf_mlrt.tensor, [[input1:%.*]]: !tf_mlrt.tensor)
func.func @main(%input0: tensor<i32>, %input1: tensor<i32>, %input2: tensor<!tf_type.variant<tensor<*xf32>>> ) -> tensor<i32> {
  %0 = "tf.Cast"(%input0) {__op_key = 0: i32, device = "/device:CPU:0"} : (tensor<i32>) -> tensor<f32>
  // CHECK: tf_mlrt_tpu.compile_and_execute
  %1, %2 = "tf.TPUCompileMlirAndExecute"(%0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<f32>) -> (tensor<i32>, tensor<i32>)
  %max_iter = "tf.Const"() {__op_key = 1, value = dense<2> : tensor<i32>} : () -> tensor<i32>
  // CHECK: tf_mlrt.map_fn
  %result = "tf_mlrt.tf_map_fn"(%max_iter, %input2, %2) { operandSegmentSizes = array<i32: 1, 1, 1>, body_fn = @NopMapFnBody, num_tensor_list_or_flow_in = 1 : i32} : (tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}

// CHECK-LABEL: @NopMapFnBody
func.func private @NopMapFnBody(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant<tensor<*xf32>>>) -> () {
  %const = "tf.Const"() {__op_key = 2 : i32, value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %a = "tf.AddV2"(%arg2, %const) {__op_key = 3: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  return
}

// -----
func.func @callee(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  %1 = "tf.TPUCompileMlirAndExecute"(%arg0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<i32>) -> tensor<i32>
  %const = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %a = "tf.AddV2"(%arg1, %const) {__op_key = 3: i32}: (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1: tensor<i32>
}

// CHECK-LABEL: func @batch_function
func.func @batch_function(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: tf_mlrt.batch_function(%arg0, %arg1)
  // CHECK-NOT:  batch_function.device
  %0 = "tf.BatchFunction"(%arg0, %arg1) {device = "/device:CPU:0", allowed_batch_sizes = [64], batch_timeout_micros = 1 : i64, batching_queue = "", container = "", f = @callee, max_batch_size = 256 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 2, 0>, shared_name = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----
func.func @batched_func(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  %0 = "tf.TPUCompileMlirAndExecute"(%arg0) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<i32>) -> tensor<i32>
  %2 = "tf.TPUCompileMlirAndExecute"(%arg1) {metadata = "metadata", mlir_module = "mlir_module", operandSegmentSizes = array<i32: 1, 0>, producer_name = "producer_name"} : (tensor<i32>) -> tensor<i32>
  func.return %2: tensor<i32>
}

// CHECK-LABEL: func @batch_function
func.func @batch_function(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
  // CHECK: [[device:%.*]] = tf_mlrt_tpu.get_tpu_host_device
  // CHECK: [[batch_result_future:%.*]] = tf_mlrt.batch_function.device([[device]]) (%arg0, %arg1)
  // CHECK: [[batch_result:%.*]] = tf_mlrt.await [[batch_result_future]]
  // CHECK: return [[batch_result]]
  %0 = "tf.BatchFunction"(%arg0, %arg1) {device = "/device:CPU:0", allowed_batch_sizes = [64], batch_timeout_micros = 1 : i64, batching_queue = "", container = "", f = @batched_func, max_batch_size = 256 : i64, num_batch_threads = 2 : i64, operandSegmentSizes = array<i32: 2, 0>, shared_name = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}



