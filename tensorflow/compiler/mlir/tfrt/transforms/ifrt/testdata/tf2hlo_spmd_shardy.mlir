module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
func.func @main(%arg0: tensor<4x64xf32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x64xf32> {mhlo.sharding = ""}) attributes {tfrt_ifrt_serving.program_id = -5304490761103885474 : i64, __tpu_compile_metadata_text = "args { dtype: DT_FLOAT shape { dim { size: 4 } dim { size: 64 } } kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 iota_reshape_dims: 2 iota_transpose_perm: 0 } is_bounded_dynamic_dim: false } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 device_assignment { replica_count: 1 computation_count: 2 computation_devices { replica_device_ids: 0 } computation_devices { replica_device_ids: 1 } } use_spmd_for_xla_partitioning: true use_shardy_partitioner: true compile_options { }"} {
    %0 = "tf.Relu"(%arg0) : (tensor<4x64xf32>) -> tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}