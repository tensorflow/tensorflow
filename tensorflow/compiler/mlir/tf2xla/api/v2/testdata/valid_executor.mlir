
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() {
    tf_executor.graph {
      %0:2 = tf_executor.island wraps "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:TPU:0", dtype = "tfdtype$DT_INT32", value = #tf_type<tensor_proto : "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<2xi32>} : () -> tensor<2xi32> loc("Empty/shape")
      tf_executor.fetch
    }
    func.return
  }
}