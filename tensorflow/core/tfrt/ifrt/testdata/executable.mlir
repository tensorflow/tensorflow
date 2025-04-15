module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>) -> tensor<*xi32> attributes {__tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT32 kind: PARAMETER } retvals { }  num_replicas: 1 num_cores_per_replica: 1"} {
    %0 = "tf.MatMul"(%arg0, %arg1): (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %0 : tensor<*xi32>
  }
}
