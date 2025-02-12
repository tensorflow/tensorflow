module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() {
    tf_executor.graph {
      %control = tf_executor.island {
        tf_executor.yield
      }
      tf_executor.fetch %control : !tf_executor.control
    }
    tf_executor.graph {
      %control = tf_executor.island {
        tf_executor.yield
      }
      tf_executor.fetch %control : !tf_executor.control
    }
    return
  }
}