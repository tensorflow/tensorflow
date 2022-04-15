// RUN: tf-tfrt-opt -tf-executor-to-tfrt-pipeline="enable-native-ops=false enable-optimizer=true tfrt-cost-threshold=1024" %s | FileCheck %s --dump-input=fail

// CHECK: tfrt.cost_threshold = 1024 : i64
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 462 : i32}} {

// CHECK-LABEL: func @__forward_call_369
// CHECK-SAME: ([[in_chain:%.*]]: !tfrt.chain
// CHECK-SAME: [[arg1_th:%.*]]: !corert.tensorhandle {tf._user_specified_name = "inputs"},
// CHECK-SAME: [[arg2_th:%.*]]: !corert.tensorhandle, [[arg3_th:%.*]]: !corert.tensorhandle, [[arg4_th:%.*]]: !corert.tensorhandle, [[arg5_th:%.*]]: !corert.tensorhandle)
// CHECK-SAME: -> (!tfrt.chain
// CHECK-NEXT: [[arg1:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg1_th]] {device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT: [[arg4:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg4_th]] {device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT: [[arg5:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg5_th]] {device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT: [[arg2:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg2_th]] {device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT: [[arg3:%.*]] = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg3_th]] {device = "/job:localhost/replica:0/task:0/device:CPU:0"
// CHECK-NEXT: [[o1:%.*]] = tfrt_fallback_async.const_dense_tensor
// CHECK-NEXT: [[o2_chain:%.*]], [[o2:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(0) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.ReadVariableOp"([[arg3]])
// CHECK-NEXT: [[o3_chain:%.*]], [[o3:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(1) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.ReadVariableOp"([[arg2]])
// CHECK-NEXT: [[o4_chain:%.*]], [[o4:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(2) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.ReadVariableOp"([[arg5]])
// CHECK-NEXT: [[o5_chain:%.*]], [[o5:%.*]] = tfrt_fallback_async.executeop.seq([[in_chain]]) key(3) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.ReadVariableOp"([[arg4]])
// CHECK-NEXT: [[o6:%.*]] = tfrt_fallback_async.executeop key(4) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf._FusedConv2D"([[arg1]], [[o3]], [[o2]])
// CHECK-NEXT: [[o7:%.*]] = tfrt_fallback_async.executeop key(5) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.AvgPool"([[o6]])
// CHECK-NEXT: [[o8:%.*]] = tfrt_fallback_async.executeop key(6) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf.Reshape"([[o7]], [[o1]])
// CHECK-NEXT: [[o9:%.*]] = tfrt_fallback_async.executeop key(7) cost({{.*}}) device("/job:localhost/replica:0/task:0/device:CPU:0") "tf._FusedMatMul"([[o8]], [[o5]], [[o4]])
// CHECK-NEXT: [[out_chain:%.*]] = tfrt.merge.chains [[o2_chain]], [[o3_chain]], [[o4_chain]], [[o5_chain]]
// CHECK-NEXT: [[o9_th:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[o9]]
// CHECK-NEXT: [[o5_th:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[o5]]
// CHECK-NEXT: [[o8_th:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[o8]]
// CHECK-NEXT: [[o6_th:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[o6]]
// CHECK-NEXT: [[o3_th:%.*]] = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle [[o3]]
// CHECK-NEXT: tfrt.return [[out_chain]], [[o9_th]], [[o5_th]], [[o8_th]], [[o6_th]], [[arg1_th]], [[o3_th]] : !tfrt.chain, !corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle
  func.func @__forward_call_369(%arg0: tensor<16x224x224x3xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<*x!tf_type.resource>, %arg2: tensor<*x!tf_type.resource>, %arg3: tensor<*x!tf_type.resource>, %arg4: tensor<*x!tf_type.resource>) -> (tensor<?x?xf32>, tensor<*xf32>, tensor<?x16384xf32>, tensor<16x112x112x?xf32>, tensor<16x224x224x3xf32>, tensor<*xf32>) attributes {tf.entry_function = {control_outputs = "", inputs = "inputs_0,conv1_conv2d_readvariableop_resource,conv1_biasadd_readvariableop_resource,fc1000_matmul_readvariableop_resource,fc1000_biasadd_readvariableop_resource", outputs = "identity_RetVal,fc1000_matmul_readvariableop_RetVal,flatten_reshape_RetVal,relu_RetVal,inputs_RetVal,conv1_conv2d_readvariableop_RetVal"}} {
    %0:6 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.ReadVariableOp"(%arg4) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<*x!tf_type.resource>) -> tensor<*xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[-1, 16384]> : tensor<2xi32>} : () -> tensor<2xi32>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.Conv2D"(%arg0, %outputs_0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 2, 2, 1], use_cudnn_on_gpu = true} : (tensor<16x224x224x3xf32>, tensor<*xf32>) -> tensor<16x112x112x?xf32>
      %outputs_10, %control_11 = tf_executor.island wraps "tf.BiasAdd"(%outputs_8, %outputs) {data_format = "NHWC", device = ""} : (tensor<16x112x112x?xf32>, tensor<*xf32>) -> tensor<16x112x112x?xf32>
      %outputs_12, %control_13 = tf_executor.island wraps "tf.Relu"(%outputs_10) {device = ""} : (tensor<16x112x112x?xf32>) -> tensor<16x112x112x?xf32>
      %outputs_14, %control_15 = tf_executor.island wraps "tf.AvgPool"(%outputs_12) {data_format = "NHWC", device = "", ksize = [1, 7, 7, 1], padding = "VALID", strides = [1, 7, 7, 1]} : (tensor<16x112x112x?xf32>) -> tensor<16x16x16x?xf32>
      %outputs_16, %control_17 = tf_executor.island wraps "tf.Reshape"(%outputs_14, %outputs_6) {device = ""} : (tensor<16x16x16x?xf32>, tensor<2xi32>) -> tensor<?x16384xf32>
      %outputs_18, %control_19 = tf_executor.island wraps "tf.MatMul"(%outputs_16, %outputs_4) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x16384xf32>, tensor<*xf32>) -> tensor<?x?xf32>
      %outputs_20, %control_21 = tf_executor.island wraps "tf.BiasAdd"(%outputs_18, %outputs_2) {data_format = "NHWC", device = ""} : (tensor<?x?xf32>, tensor<*xf32>) -> tensor<?x?xf32>
      %outputs_22, %control_23 = tf_executor.island wraps "tf.Identity"(%outputs_20) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
      tf_executor.fetch %outputs_22, %outputs_4, %outputs_16, %outputs_12, %arg0, %outputs_0 : tensor<?x?xf32>, tensor<*xf32>, tensor<?x16384xf32>, tensor<16x112x112x?xf32>, tensor<16x224x224x3xf32>, tensor<*xf32>
    }
    func.return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<?x?xf32>, tensor<*xf32>, tensor<?x16384xf32>, tensor<16x112x112x?xf32>, tensor<16x224x224x3xf32>, tensor<*xf32>
  }

  func.func @while_cond_lt9(%arg0: tensor<i32>) -> tensor<i1> {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<9> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Less"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    func.return %1 : tensor<i1>
  }

  func.func @while_body_add2(%arg0: tensor<i32>) -> tensor<i32> {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Add"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    func.return %1 : tensor<i32>
  }

  // CHECK-LABEL: func @while_test
  // CHECK-SAME: ([[ARG0:%.+]]: !tfrt.chain) -> (!tfrt.chain, !corert.tensorhandle)
  func.func @while_test() -> (tensor<i32>) {
    // The predicate function should be inlined.
    // CHECK: corert.const_dense_tensor dense<0> : tensor<i32>
    // CHECK-NEXT: tfrt_fallback_async.const_dense_tensor dense<0> : tensor<i32>
    // CHECK-NEXT: tfrt_fallback_async.const_dense_tensor dense<9> : tensor<i32>
    // CHECK-NEXT: tfrt_fallback_async.executeop key({{.*}}) cost({{.*}}) device("/device:CPU:0") "tf.Less"
    // CHECK-NEXT: [[pred:%.*]] = tfrt_fallback_async.predicate
    // CHECK-NEXT: tfrt.while [[pred]] @"while_body_add2/tfrt_body_1"
    // CHECK-NEXT: tfrt.merge.chains
    // CHECK-NEXT: tfrt.return
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.While"(%0) { cond = @while_cond_lt9, body = @while_body_add2, is_stateless = false, parallel_iterations = 1} : (tensor<i32>) -> (tensor<i32>)
    func.return %1 : tensor<i32>
  }
  // CHECK: func @"while_body_add2/tfrt_body_1"
  // CHECK-NOT: tfrt.call

  // CHECK: func @"while_cond_lt9/tfrt_predicate"
}
