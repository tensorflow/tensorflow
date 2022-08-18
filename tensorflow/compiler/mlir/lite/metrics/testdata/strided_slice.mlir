func.func @strided_slice(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %18 = "tf.Const"() {value = dense<1> : tensor<4xi32>} : () -> tensor<4xi32>
  %57 = "tf.Const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  %534 = "tf.StridedSlice"(%arg0, %57, %57, %18) {begin_mask = 11 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 11 : i64, new_axis_mask = 4 : i64, shrink_axis_mask = 0 : i64} : (tensor<*xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<*xf32>
  "func.return"(%534) : (tensor<*xf32>) -> ()
}
