module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
     %s0 = "tf.Const"() {value = dense<[4, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    func.return
  }
}