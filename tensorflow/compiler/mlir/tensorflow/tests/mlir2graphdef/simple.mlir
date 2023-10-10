// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  // CHECK: node {
  // CHECK-NEXT: name: "Empty/shape"
  // CHECK-NEXT: op: "Const"
  // CHECK: attr {
  // CHECK:        key: "dtype"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     type: DT_INT32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: attr {
  // CHECK-NEXT:   key: "value"
  // CHECK-NEXT:   value {
  // CHECK-NEXT:     tensor {
  // CHECK-NEXT:       dtype: DT_INT32
  // CHECK-NEXT:       tensor_shape {
  // CHECK-NEXT:         dim {
  // CHECK-NEXT:           size: 2
  // CHECK-NEXT:         }
  // CHECK-NEXT:       }
  // CHECK-NEXT:       tensor_content: "\200\000\000\000\200\000\000\000"
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", dtype = "tfdtype$DT_INT32", value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<2xi32>} : () -> tensor<2xi32> loc("Empty/shape")
    tf_executor.fetch
  }
  func.return
}
