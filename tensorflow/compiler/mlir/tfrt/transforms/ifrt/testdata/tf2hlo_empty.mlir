module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () attributes {__tpu_compile_metadata_text = "num_replicas: 1 num_cores_per_replica: 1"} {
    func.return
  }
}