// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 175 : i32}} {
  func.func @main(%arg0: tensor<32x!tf_type.string>) -> (tensor<?x2xi64>) attributes {tf.entry_function = {inputs = "input0", outputs = "ParseExample/ParseExampleV2"}} {

    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", dtype = f32, value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", dtype = f32, value = dense<[]> : tensor<0xf32>} : () -> tensor<0xf32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {device = "", dtype = !tf_type.string, value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Const"() {device = "", dtype = !tf_type.string, value = dense<""> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.Const"() {device = "", dtype = !tf_type.string, value = dense<""> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
      %outputs_8, %control_9 = tf_executor.island wraps "tf.Const"() {device = "", dtype = !tf_type.string, value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>

      %outputs_10:8, %control_11 = tf_executor.island wraps "tf.ParseExampleV2"(%arg0, %outputs_4, %outputs_8, %outputs_2, %outputs_6, %outputs, %outputs_0) {Tdense = ["tfdtype$DT_FLOAT", "tfdtype$DT_FLOAT"], dense_shapes = [#tf_type.shape<>, #tf_type.shape<>], device = "", num_sparse = 2 : i64, ragged_split_types = [], ragged_value_types = [], result_segment_sizes = array<i32: 2, 2, 2, 2, 0, 0>, sparse_types = ["tfdtype$DT_STRING", "tfdtype$DT_INT64"]} : (tensor<32x!tf_type.string>, tensor<0x!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>, tensor<0x!tf_type.string>, tensor<0xf32>, tensor<0xf32>) -> (tensor<?x2xi64>, tensor<?x2xi64>, tensor<?x!tf_type.string>, tensor<?xi64>, tensor<2xi64>, tensor<2xi64>, tensor<32xf32>, tensor<32xf32>) loc("ParseExample")
      // CHECK:      name: "ParseExample"
      // CHECK-NEXT: op: "ParseExampleV2"
      // CHECK-NEXT: input: "input0"
      // CHECK-NEXT: input: "tf.Const3"
      // CHECK-NEXT: input: "tf.Const5"
      // CHECK-NEXT: input: "tf.Const2"
      // CHECK-NEXT: input: "tf.Const4"
      // CHECK-NEXT: input: "tf.Const"
      // CHECK-NEXT: input: "tf.Const1"
      // CHECK-NEXT: attr {
      // CHECK-NEXT:   key: "Tdense"
      // CHECK-NEXT:     value {
      // CHECK-NEXT:       list {
      // CHECK-NEXT:         type: DT_FLOAT
      // CHECK-NEXT:         type: DT_FLOAT
      // CHECK-NEXT:       }
      // CHECK-NEXT:     }
      // CHECK-NEXT:   }
      // CHECK-NEXT: attr {
      // CHECK:        key: "dense_shapes"
      // CHECK-NEXT:   value {
      // CHECK-NEXT:     list {
      // CHECK-NEXT:       shape {
      // CHECK-NEXT:       }
      // CHECK-NEXT:       shape {
      // CHECK-NEXT:       }
      // CHECK-NEXT:     }
      // CHECK-NEXT:   }
      // CHECK-NEXT: }
      // CHECK-NEXT: attr {
      // CHECK-NEXT:   key: "num_sparse"
      // CHECK-NEXT:   value {
      // CHECK-NEXT:     i: 2
      // CHECK-NEXT:   }
      // CHECK-NEXT: }
      // CHECK-NEXT: attr {
      // CHECK-NEXT:   key: "ragged_split_types"
      // CHECK-NEXT:   value {
      // CHECK-NEXT:     list {
      // CHECK-NEXT:     }
      // CHECK-NEXT:   }
      // CHECK-NEXT: }
      // CHECK-NEXT: attr {
      // CHECK-NEXT:   key: "ragged_value_types"
      // CHECK-NEXT:   value {
      // CHECK-NEXT:     list {
      // CHECK-NEXT:     }
      // CHECK-NEXT:   }
      // CHECK-NEXT: }
      // CHECK-NEXT: attr {
      // CHECK-NEXT:   key: "sparse_types"
      // CHECK-NEXT:   value {
      // CHECK-NEXT:     list {
      // CHECK-NEXT:       type: DT_STRING
      // CHECK-NEXT:       type: DT_INT64
      // CHECK-NEXT:     }
      // CHECK-NEXT:   }
      // CHECK-NEXT: }

      tf_executor.fetch %outputs_10#0 : tensor<?x2xi64>
    }
    func.return %0#0 : tensor<?x2xi64>
    // CHECK:      name: "ParseExample/ParseExampleV2"
    // CHECK-NEXT: op: "_Retval"
    // CHECK-NEXT: input: "ParseExample"

  }
}

