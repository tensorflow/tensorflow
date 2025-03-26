// RUN: tf-tfrt-opt -split-input-file -tf-executor-to-tfrt-pipeline="target-gpu=true" -tfrt-lower-tf-savedmodel=hoist-invariant-ops=true %s | FileCheck %s --dump-input=fail --dump-input-filter=all

func.func private @xla_func_0(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %1 : tensor<1x3xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<1x3xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: [[INPUT_0:%.*]] = gpurt.transfer_to_device
  // CHECK: [[VAR_0:%.*]] = gpurt.maybe_transfer_variable
  // CHECK: tfrt_fallback_async.executeop.seq{{.*}}"tf.XlaLaunch"([[INPUT_0]], [[VAR_0]])
  // CHECK-SAME: {function = "xla_func_0"}
  // CHECK: gpurt.transfer_from_device
  %2 = "tf.XlaLaunch"(%arg0, %1) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_0, operandSegmentSizes = array<i32: 0, 2, 0>} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<*xf32>
  func.return %2 : tensor<*xf32>
}

// Check the case when there are multiple XLA clusters.

func.func private @xla_func_1(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %1 : tensor<1x3xf32>
}

func.func private @xla_func_2(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg0) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %1 : tensor<1x3xf32>
}

// CHECK-LABEL: func @multi_clusters
func.func @multi_clusters(%arg0: tensor<1x3xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: [[INPUT_0:%.*]] = gpurt.transfer_to_device
  // CHECK: [[VAR_0:%.*]] = gpurt.maybe_transfer_variable
  // CHECK: tfrt_fallback_async.executeop.seq{{.*}}"tf.XlaLaunch"([[INPUT_0]], [[VAR_0]])
  // CHECK-SAME: {function = "xla_func_1"}
  // CHECK: [[RESULT_1:%.*]] = gpurt.transfer_from_device
  %2 = "tf.XlaLaunch"(%arg0, %1) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_1, operandSegmentSizes = array<i32: 0, 2, 0>} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>

  // The output of the above XLA cluster is consumed by the below XLA cluster.
  // Currently, the output is first transferred back to CPU and then
  // transferred to GPU again, which is unnecessary.
  // TODO(b/262280565): Remove unnecessary data transfers when there are
  // multiple XLA clusters.
  // CHECK: [[INPUT_1:%.*]] = gpurt.transfer_to_device [[RESULT_1]]
  // CHECK: tfrt_fallback_async.executeop.seq{{.*}}"tf.XlaLaunch"([[INPUT_1]])
  // CHECK-SAME: {function = "xla_func_2"}
  // CHECK: gpurt.transfer_from_device
  %3 = "tf.XlaLaunch"(%2) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_2, operandSegmentSizes = array<i32: 0, 1, 0>} : (tensor<1x3xf32>) -> tensor<*xf32>

  func.return %3 : tensor<*xf32>
}


// Check that unused outputs of the XLA cluster are not transferred.

func.func private @xla_func_3(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>) attributes {tf._XlaMustCompile = true, tf._noinline = true, tf._original_func_name = "should_not_be_used"} {
  %1 = "tf.AddV2"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  %2 = "tf.DIV"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  func.return %1, %2 : tensor<1x3xf32>, tensor<1x3xf32>
}

// CHECK-LABEL: func @skip_unused_output
func.func @skip_unused_output(%arg0: tensor<1x3xf32>) -> tensor<*xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "input:0", outputs = "output:0"}} {
  %0 = "tf.VarHandleOp"() {device = "/device:CPU:0", container = "", shared_name = "variable"} : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
  %1 = "tf.ReadVariableOp"(%0) {device = "/device:CPU:0"} : (tensor<!tf_type.resource<tensor<1x3xf32>>>) -> tensor<1x3xf32>
  // CHECK: [[INPUT_0:%.*]] = gpurt.transfer_to_device
  // CHECK: [[VAR_0:%.*]] = gpurt.maybe_transfer_variable
  // CHECK: tfrt_fallback_async.executeop.seq{{.*}}"tf.XlaLaunch"([[INPUT_0]], [[VAR_0]])
  // CHECK-SAME: {function = "xla_func_3"}
  // Since only one output of the XlaLaunch is used, there is only one data transfer.
  // CHECK: gpurt.transfer_from_device
  // CHECK-NOT: gpurt.transfer_from_device
  %2:2 = "tf.XlaLaunch"(%arg0, %1) {_noinline = true, _xla_compile_device_type = "GPU", device = "/device:GPU:0", function = @xla_func_3, operandSegmentSizes = array<i32: 0, 2, 0>} : (tensor<1x3xf32>, tensor<1x3xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  func.return %2#0 : tensor<*xf32>
}


