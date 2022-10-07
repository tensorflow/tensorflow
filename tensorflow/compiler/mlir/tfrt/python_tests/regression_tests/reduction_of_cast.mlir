func.func @compute(%arg0: tensor<?xi64> { python_test_attrs.static_type = tensor<2xi64> }) -> tensor<i32> {
  %cst = "tf.Const"() { value = dense<0> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Cast"(%arg0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi64>) -> tensor<?xi32>
  %1 = "tf.Prod"(%0, %cst) { keep_dims = false, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}
