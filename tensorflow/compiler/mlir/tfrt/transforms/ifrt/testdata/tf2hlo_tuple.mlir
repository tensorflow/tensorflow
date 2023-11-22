module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> (tensor<1x1xf32>, tensor<1x3xf32>) {
    %0 = "tf.MatMul"(%arg0, %arg1): (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
    func.return %0, %arg0: tensor<1x1xf32>, tensor<1x3xf32>
  }
}