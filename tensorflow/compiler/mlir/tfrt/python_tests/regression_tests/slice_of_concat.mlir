func.func @test(%V__0 : tensor<i8> { python_test_attrs.static_type = tensor<i8> }, %V__1 : tensor<?xf32> { python_test_attrs.static_type = tensor<6xf32> }) -> tensor<?x?xi8> {
  %0 = "tf.Shape"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?xf32>) -> tensor<1xi32>
  %1 = "tf.Const"() { value = dense<[68, 36, 37]> : tensor<3xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<3xi32>
  %axis2 = "tf.Const"() { value = dense<0> : tensor<i32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<i32>
  %2 = "tf.ConcatV2"(%0, %1, %axis2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<1xi32>, tensor<3xi32>, tensor<i32>) -> tensor<4xi32>
  %begin3 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %size3 = "tf.Const"() { value = dense<[2]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %3 = "tf.Slice"(%2, %begin3, %size3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %4 = "tf.BroadcastTo"(%V__0, %3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<i8>, tensor<2xi32>) -> tensor<?x?xi8>
  %dims5 = "tf.Const"() { value = dense<[1]> : tensor<1xi32>, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : () -> tensor<1xi32>
  %5 = "tf.Prod"(%4, %dims5) { keep_dims = true, device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?xi8>, tensor<1xi32>) -> tensor<?x?xi8>
  func.return %5 : tensor<?x?xi8>
}
