module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.MatMul"(%arg0, %arg1): (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %0 : tensor<*xi32>
  }
}