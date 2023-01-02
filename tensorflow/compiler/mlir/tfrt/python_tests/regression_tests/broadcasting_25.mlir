func.func @test(
    %V__0: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> },
    %V__1: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> },
    %V__2: tensor<?xf32> { python_test_attrs.static_type = tensor<1xf32> }
  ) -> tensor<?xf32> {
  %1 = "tf.AddV2"(%V__0, %V__0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" }
    : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %2 = "tf.Rint"(%V__1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" }
    : (tensor<?xf32>) -> tensor<?xf32>
  %3 = "tf.Exp"(%2) { device = "/job:localhost/replica:0/task:0/device:CPU:0" }
    : (tensor<?xf32>) -> tensor<?xf32>
  %4 = "tf.AddV2"(%V__2, %3) { device = "/job:localhost/replica:0/task:0/device:CPU:0" }
    : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %4 : tensor<?xf32>
}
