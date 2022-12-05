func.func @test(
    %arg0: tensor<?x?x1xf32> { python_test_attrs.static_type = tensor<1x1x1xf32> },
    %arg1: tensor<?x?x20xf32> { python_test_attrs.static_type = tensor<12x13x20xf32> },
    %arg2: tensor<?x?x40xf32> { python_test_attrs.static_type = tensor<14x15x40xf32> },
    %arg3: tensor<?x?x40xf32> { python_test_attrs.static_type = tensor<16x17x40xf32> },
    %arg4: tensor<?x?x33xf32> { python_test_attrs.static_type = tensor<18x19x33xf32> },
    %arg5: tensor<?x?x13xf32> { python_test_attrs.static_type = tensor<20x21x13xf32> }
  ) -> (tensor<?x?x20xf32>, tensor<?x?x40xf32>, tensor<?x?x40xf32>, tensor<?x?x33xf32>, tensor<?x?x13xf32>) {
  %0 = "tf.Const"() { value = dense<9.99999997E-7>: tensor<f32> } : () -> tensor<f32>
  %1 = "tf.AddV2"(%arg0, %0) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x1xf32>, tensor<f32>) -> tensor<?x?x1xf32>
  %2 = "tf.RealDiv"(%arg1, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x20xf32>, tensor<?x?x1xf32>) -> tensor<?x?x20xf32>
  %3 = "tf.RealDiv"(%arg2, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x40xf32>, tensor<?x?x1xf32>) -> tensor<?x?x40xf32>
  %4 = "tf.RealDiv"(%arg3, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x40xf32>, tensor<?x?x1xf32>) -> tensor<?x?x40xf32>
  %5 = "tf.RealDiv"(%arg4, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x33xf32>, tensor<?x?x1xf32>) -> tensor<?x?x33xf32>
  %6 = "tf.RealDiv"(%arg5, %1) { device = "/job:localhost/replica:0/task:0/device:CPU:0" } : (tensor<?x?x13xf32>, tensor<?x?x1xf32>) -> tensor<?x?x13xf32>
  "func.return"(%2, %3, %4, %5, %6) : (tensor<?x?x20xf32>, tensor<?x?x40xf32>, tensor<?x?x40xf32>, tensor<?x?x33xf32>, tensor<?x?x13xf32>) -> ()
}
