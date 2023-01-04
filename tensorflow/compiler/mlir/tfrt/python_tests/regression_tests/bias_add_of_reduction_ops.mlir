func.func @test(%V__0: tensor<?x?x?xf32> { python_test_attrs.static_type = tensor<8x9x10xf32> }, %V__1: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }) -> tensor<?x?x?xf32> {
  %dims0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %0 = "tf.Min"(%V__0, %dims0) {keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<1xi32>) -> tensor<?x?x?xf32>
  %dims1 = "tf.Const"() { value = dense<[0, 2]> : tensor<2xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<2xi32>
  %1 = "tf.Sum"(%0, %dims1) {keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?x?xf32>
  %2 = "tf.BiasAdd"(%1, %V__1) {data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?xf32>
  func.return %2 : tensor<?x?x?xf32>
}
