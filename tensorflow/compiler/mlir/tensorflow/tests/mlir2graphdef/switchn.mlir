// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

"builtin.module"() ( {
  "builtin.func"() ( {
    "tf_executor.graph"() ( {
      %outputs, %control = "tf_executor.island"() ( {
        %0 = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
        "tf_executor.yield"(%0) : (tensor<i32>) -> ()
      }) : () -> (tensor<i32>, !tf_executor.control)
      %outputs_0:3, %control_1 = "tf_executor._SwitchN"(%outputs, %outputs) {T = i32, device = "", num_outs = 3 : i64} : (tensor<i32>, tensor<i32>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, !tf_executor.control)
      %outputs_2, %control_3 = "tf_executor.island"() ( {
        %0 = "tf.Identity"(%outputs_0#0) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
        "tf_executor.yield"(%0) : (tensor<*xi32>) -> ()
      }) : () -> (tensor<*xi32>, !tf_executor.control)
      %outputs_4, %control_5 = "tf_executor.island"(%control_3) ( {
        %0 = "tf.Const"() {device = "", value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
        "tf_executor.yield"(%0) : (tensor<f32>) -> ()
      }) : (!tf_executor.control) -> (tensor<f32>, !tf_executor.control)
      %outputs_6, %control_7 = "tf_executor.island"() ( {
        %0 = "tf.Identity"(%outputs_0#1) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
        "tf_executor.yield"(%0) : (tensor<*xi32>) -> ()
      }) : () -> (tensor<*xi32>, !tf_executor.control)
      %outputs_8, %control_9 = "tf_executor.island"(%control_7) ( {
        %0 = "tf.Const"() {device = "", value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
        "tf_executor.yield"(%0) : (tensor<f32>) -> ()
      }) : (!tf_executor.control) -> (tensor<f32>, !tf_executor.control)
      %outputs_10, %control_11 = "tf_executor.island"() ( {
        %0 = "tf.Identity"(%outputs_0#2) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
        "tf_executor.yield"(%0) : (tensor<*xi32>) -> ()
      }) : () -> (tensor<*xi32>, !tf_executor.control)
      %outputs_12, %control_13 = "tf_executor.island"(%control_11) ( {
        %0 = "tf.Const"() {device = "", value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
        "tf_executor.yield"(%0) : (tensor<f32>) -> ()
      }) : (!tf_executor.control) -> (tensor<f32>, !tf_executor.control)
      %outputs_14, %control_15 = "tf_executor.island"() ( {
        %0 = "tf.Const"() {device = "", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
        "tf_executor.yield"(%0) : (tensor<f32>) -> ()
      }) : () -> (tensor<f32>, !tf_executor.control)
      %outputs_16:2, %control_17 = "tf_executor._SwitchN"(%outputs_14, %outputs) {T = f32, _class = ["Case/input_0"], device = "", num_outs = 2 : i64} : (tensor<f32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control)
      %outputs_18, %control_19 = "tf_executor.island"() ( {
        %0 = "tf.Mul"(%outputs_16#0, %outputs_4) {device = ""} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
        "tf_executor.yield"(%0) : (tensor<*xf32>) -> ()
      }) : () -> (tensor<*xf32>, !tf_executor.control)
      %outputs_20, %control_21 = "tf_executor.island"() ( {
        %0 = "tf.Mul"(%outputs_16#1, %outputs_8) {device = ""} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
        "tf_executor.yield"(%0) : (tensor<*xf32>) -> ()
      }) : () -> (tensor<*xf32>, !tf_executor.control)
      %output, %value_index, %control_22 = "tf_executor.Merge"(%outputs_18, %outputs_20) {N = 2 : i64, T = f32, device = ""} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>, !tf_executor.control)
      %control_23 = "tf_executor.island"() ( {
        "tf._Retval"(%output) {T = f32, device = "/job:localhost/replica:0/task:0/device:CPU:0", index = 0 : i64} : (tensor<*xf32>) -> ()
        "tf_executor.yield"() : () -> ()
      }) : () -> !tf_executor.control
      "tf_executor.fetch"() : () -> ()
    }) : () -> ()
    "std.return"() : () -> ()
  }) {sym_name = "main", type = () -> ()} : () -> ()
}) {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 126 : i32}} : () -> ()

// CHECK: _SwitchN
