func.func @test(%V__0: tensor<?x?x?xi64>
    { python_test_attrs.static_type = tensor<9x32x26xi64> })
    -> tensor<?x?xi64> {
  %dims1 = "tf.Const"() {
        value = dense<[0]> : tensor<1xi32>,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : () -> tensor<1xi32>
  %1 = "tf.Mean"(%V__0, %dims1) {
        keep_dims = false,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : (tensor<?x?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  %dims2 = "tf.Const"() {
        value = dense<[1]> : tensor<1xi32>,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : () -> tensor<1xi32>
  %2 = "tf.Min"(%1, %dims2) {
        keep_dims = true,
        device = "/job:localhost/replica:0/task:0/device:CPU:0"
      } : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?x?xi64>
  func.return %2 : tensor<?x?xi64>
}
