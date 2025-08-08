module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 2246 : i32}} {
  func.func private @_stablehlo_main_add(%dummy: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {jax.result_info = "", mhlo.sharding = "{devices=[2,1]<=[2]}"}) attributes {_from_xla_call_module} {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4x2xi32>
    return %0 : tensor<4x2xi32>
  }

  func.func private @_stablehlo_main_sub(%dummy: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {jax.result_info = "", mhlo.sharding = "{devices=[2,1]<=[2]}"}) attributes {_from_xla_call_module} {
    %0 = stablehlo.negate %arg1 : tensor<4x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<4x2xi32>
    return %1 : tensor<4x2xi32>
  }

  func.func @main(%arg0: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {mhlo.sharding = ""}, tensor<4x2xi32> {mhlo.sharding = ""})  attributes {tfrt_ifrt_serving.program_id = -6498204404019954638 : i64, __tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 tile_assignment_devices: 0 tile_assignment_devices: 1 } is_bounded_dynamic_dim: false } retvals { sharding { } } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 use_spmd_for_xla_partitioning: true compile_options { } "} {
    %0 = "tf.XlaSharding"(%arg0) <{_XlaSharding = "{devices=[2,1]<=[2]}", sharding = "{devices=[2,1]<=[2]}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %1 = "tf.XlaSharding"(%arg1) <{_XlaSharding = "{devices=[2,1]<=[2]}", sharding = "{devices=[2,1]<=[2]}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %2 = "tf.XlaCallModule"(%0, %1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_sub, _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, _stablehlo_version = "1.11.0", module = "", platforms = ["TPU", "CUDA"], version = 10 : i64, use_shardy_partitioner = false} : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.XlaCallModule"(%0, %1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_add, _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, _stablehlo_version = "1.11.0", module = "", platforms = ["TPU", "CUDA"], version = 10 : i64, use_shardy_partitioner = false} : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    return %2, %3 : tensor<4x2xi32>, tensor<4x2xi32>
  }
}


