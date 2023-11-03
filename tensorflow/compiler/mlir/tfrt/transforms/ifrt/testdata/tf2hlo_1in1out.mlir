module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<1x3xi32>) -> (tensor<1x3xi32>) {
    func.return %arg0: tensor<1x3xi32>
  }
}