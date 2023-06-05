func.func @test(%V__0 : tensor<?x?x?x?xui8> { python_test_attrs.static_type = tensor<67x0x8x1xui8> }, %V__1 : tensor<?xui8> { python_test_attrs.static_type = tensor<0xui8> }) -> tensor<?x?x?x?xui8> {
  %begin0 = "tf.Const"() { value = dense<[0, 0, 0, 0]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %size0 = "tf.Const"() { value = dense<[67, 0, 8, 1]> : tensor<4xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<4xi32>
  %0 = "tf.Slice"(%V__0, %begin0, %size0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xui8>, tensor<4xi32>, tensor<4xi32>) -> tensor<?x?x?x?xui8>
  %1 = "tf.BiasAdd"(%0, %V__1) { data_format = "NCHW", device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xui8>, tensor<?xui8>) -> tensor<?x?x?x?xui8>
  %2 = "tf.ClipByValue"(%V__0, %V__0, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x?x?xui8>, tensor<?x?x?x?xui8>, tensor<?x?x?x?xui8>) -> tensor<?x?x?x?xui8>
  func.return %2 : tensor<?x?x?x?xui8>
}
