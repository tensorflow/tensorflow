// RUN: tf-mlir-translate -test-only-mlir-to-tf-nodedef %s -o - | FileCheck %s

func @main() {
^bb0:
  // CHECK: name: "node_name"
  // CHECK-NEXT: op: "Const"
  // CHECK-NEXT: attr {
  // CHECK:      key: "dtype"
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
  %0 = "tf.Const"() {value = opaque<"tf", "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<2xi32>} : () -> (tensor<2xi32>)
  return
}


