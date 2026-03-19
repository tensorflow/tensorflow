module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
 func.func @main(%arg0: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg1: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg2: tensor<*xi32> {mhlo.sharding = ""}) -> (tensor<*xi32> {mhlo.sharding = ""}) attributes {tfrt_ifrt_serving.program_id = -6498204404019954638 : i64, __tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_INT32 kind: PARAMETER is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 use_spmd_for_xla_partitioning: true compile_options { } "} {
    %0 = "tf.XlaSharding"(%arg0) <{_XlaSharding = "{devices=[2,1]0,1}", sharding = "{devices=[2,1]0,1}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %1 = "tf.XlaSharding"(%arg1) <{_XlaSharding = "{devices=[2,1]0,1}", sharding = "{devices=[2,1]0,1}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %2 = "tf.Add"(%arg0, %arg1) : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.Add"(%2, %arg2) : (tensor<4x2xi32>, tensor<*xi32>) -> tensor<*xi32>
    return %3 : tensor<*xi32>
  }
}
