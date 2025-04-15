module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
    func.return
  }

  module @submodule attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0"}, tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1647 : i32}} {
  }

}