// RUN: tf-mlir-translate -mlir-to-graphdef %s | tf-mlir-translate -graphdef-to-mlir

// Test graphdef export is producing valid GraphDef. The test imports the
// generated GraphDef to make sure the export is valid.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 890 : i32}}  {
  func.func @main(%arg0: tensor<100x28x28x3xf32> {}) -> (tensor<*xf32> {}) attributes {tf.entry_function = {control_outputs = "", inputs = "a:0", outputs = "conv2d_2/BiasAdd:0"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.StatefulPartitionedCall"(%arg0) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_call_440} : (tensor<100x28x28x3xf32>) -> tensor<*xf32>
      %outputs_0, %control_1 = tf_executor.island(%control) wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
      %outputs_2, %control_3 = tf_executor.island(%control_1) wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<5x5x32x16xf32>} : () -> tensor<5x5x32x16xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Conv2D"(%outputs, %outputs_2) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<5x5x32x16xf32>) -> tensor<*xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.BiasAdd"(%outputs_4, %outputs_0) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<16xf32>) -> tensor<*xf32>
      tf_executor.fetch %outputs_6 : tensor<*xf32>
    }
    func.return %0 : tensor<*xf32>
  }
  func.func private @__inference_call_440(%arg0: tensor<?x28x28x3xf32> {tf._user_specified_name = "inputs"}) -> tensor<*xf32> attributes {tf._input_shapes = [#tf_type.shape<?x28x28x3>], tf.signature.is_stateful} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<32xf32>} : () -> tensor<32xf32>
      %outputs_0, %control_1 = tf_executor.island(%control) wraps "tf.Const"() {value = dense<0.000000e+00> : tensor<5x5x3x32xf32>} : () -> tensor<5x5x3x32xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Conv2D"(%arg0, %outputs_0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<?x28x28x3xf32>, tensor<5x5x3x32xf32>) -> tensor<*xf32>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.BiasAdd"(%outputs_2, %outputs) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<32xf32>) -> tensor<*xf32>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Identity"(%outputs_4) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
      tf_executor.fetch %outputs_6 : tensor<*xf32>
    }
    func.return %0 : tensor<*xf32>
  }
}


