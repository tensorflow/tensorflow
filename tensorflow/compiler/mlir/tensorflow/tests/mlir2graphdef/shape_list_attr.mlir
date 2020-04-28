// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s


// CHECK:      attr {
// CHECK:        key: "dtypes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:       type: DT_FLOAT
// CHECK-NEXT:       type: DT_INT16

// CHECK:      attr {
// CHECK-NEXT:   key: "shapes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 3
// CHECK:            shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 4
// CHECK-NEXT:         }
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: -1
// CHECK:            shape {
// CHECK-NEXT:         unknown_rank: true


func @main() {
  tf_executor.graph {
    %0:4 = tf_executor.island wraps "tf.InfeedDequeueTuple"() : () -> (tensor<3xi32>, tensor<4x?xf32>, tensor<*xi16>)
    tf_executor.fetch
  }
  return
}
