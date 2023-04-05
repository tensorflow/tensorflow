// RUN: tf-mlir-translate -mlir-tf-to-hlo-text %s -tf-input-shapes=: | FileCheck %s

module attributes {tf.versions = {producer = 179 : i32}} {
  func.func @main() -> (tensor<300x?xf32>, tensor<300x?xf32>) {
    %elem_shape = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    %size = "tf.Const"() {value = dense<300> : tensor<i32>} : () -> tensor<i32>
    %tl = "tf.TensorListReserve"(%elem_shape, %size) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>

    %idx = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %elem_1 = "tf.Const"() {value = dense<10.0> : tensor<8xf32>} : () -> tensor<8xf32>
    %tl_set_item = "tf.TensorListSetItem"(%tl, %idx, %elem_1) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<8xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
    %elem_shape_2 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
    %tls = "tf.TensorListStack"(%tl_set_item, %elem_shape_2) {num_elements = 300 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<300x?xf32>

    %elem_2 = "tf.Const"() {value = dense<10.0> : tensor<9xf32>} : () -> tensor<9xf32>
    %tl_set_item_2 = "tf.TensorListSetItem"(%tl, %idx, %elem_2) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<9xf32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
    %elem_shape_3 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
    %tls_2 = "tf.TensorListStack"(%tl_set_item_2, %elem_shape_3) {num_elements = 300 : i64} : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<300x?xf32>
    func.return %tls, %tls_2 : tensor<300x?xf32>, tensor<300x?xf32>
  }
}

// CHECK-LABEL: HloModule main
// CHECK:       ENTRY %main.{{[0-9]+}} () -> (f32[300,8], f32[300,9]) {
// CHECK:       %tuple.{{[0-9]+}} = (f32[300,8]{1,0}, f32[300,9]{1,0})
// CHECK:       }