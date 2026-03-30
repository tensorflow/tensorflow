module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<?x?xi32> {tf._static_shape_arg_idx = 2 : i32}, %arg1: tensor<?x?xi32> {tf._static_shape_arg_idx = 2 : i32}, %arg2: tensor<2xi64>) -> tensor<?x?xi32> attributes {__tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT32 kind: PARAMETER } args { dtype: DT_INT64 kind: PARAMETER } retvals { }  num_replicas: 1 num_cores_per_replica: 1"} {
    %0 = "tf.MatMul"(%arg0, %arg1) {transpose_b = true} : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    func.return %0 : tensor<?x?xi32>
  }
}
