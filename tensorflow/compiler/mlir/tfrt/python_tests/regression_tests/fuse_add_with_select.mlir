builtin.func @test(
    %V__0 : tensor<i1> { python_test_attrs.static_type = tensor<i1> },
    %V__1 : tensor<?xi64> { python_test_attrs.static_type = tensor<1xi64> },
    %V__2 : tensor<1xi64> { python_test_attrs.static_type = tensor<1xi64> },
    %V__3 : tensor<1x1xi64> { python_test_attrs.static_type = tensor<1x1xi64> })
    -> tensor<1xi64> {
  %0 = "tf.Select"(%V__0, %V__1, %V__2) :
      (tensor<i1>, tensor<?xi64>, tensor<1xi64>) -> tensor<1xi64>
  %1 = "tf.Squeeze"(%V__3) { squeeze_dims = [ 1 : i64 ] } :
      (tensor<1x1xi64>) -> tensor<1xi64>
  %2 = "tf.AddV2"(%0, %1) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  return %2 : tensor<1xi64>
}
