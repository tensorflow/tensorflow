module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
func.func @main(%arg0: tensor<4x64xf32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg1: tensor<64x10xf32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg2: tensor<*xf32> {mhlo.sharding = ""}) -> (tensor<?x10xf32> {mhlo.sharding = ""}) attributes {tfrt_ifrt_serving.program_id = -5304490761103885474 : i64, __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { unknown_rank: true } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_FLOAT shape { unknown_rank: true } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_FLOAT shape { unknown_rank: true } kind: PARAMETER is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 use_spmd_for_xla_partitioning: true compile_options { }"} {
    %cst = "tf.Const"() <{value = dense<[-1, 4]> : tensor<2xi32>}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> tensor<2xi32>
    %0 = "tf.XlaSharding"(%arg0) <{_XlaSharding = "{devices=[2,1]0,1}", sharding = "{devices=[2,1]0,1}"}> : (tensor<4x64xf32>) -> tensor<4x64xf32>
    %1 = "tf.XlaSharding"(%arg1) <{_XlaSharding = "{devices=[2,1]0,1}", sharding = "{devices=[2,1]0,1}"}> : (tensor<64x10xf32>) -> tensor<64x10xf32>
    %2 = "tf.Reshape"(%arg2, %cst) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x4xf32>
    %3 = "tf.MatMul"(%2, %0) <{transpose_a = false, transpose_b = false}> : (tensor<?x4xf32>, tensor<4x64xf32>) -> tensor<?x64xf32>
    %4 = "tf.Relu"(%3) : (tensor<?x64xf32>) -> tensor<?x64xf32>
    %5 = "tf.MatMul"(%4, %1) <{transpose_a = false, transpose_b = false}> : (tensor<?x64xf32>, tensor<64x10xf32>) -> tensor<?x10xf32>
    return %5 : tensor<?x10xf32>
  }
}