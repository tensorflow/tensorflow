module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>, %arg4: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) attributes {__tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT32 kind: PARAMETER } retvals { }  retvals { } retvals { } num_replicas: 1 num_cores_per_replica: 1"} {
    %0 = "tf.MatMul"(%arg0, %arg1): (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %1 = "tf.MatMul"(%arg2, %arg3): (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %2 = "tf.MatMul"(%arg4, %arg3): (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %0, %1, %2 : tensor<*xi32>, tensor<*xi32>, tensor<*xi32>
  }
}
