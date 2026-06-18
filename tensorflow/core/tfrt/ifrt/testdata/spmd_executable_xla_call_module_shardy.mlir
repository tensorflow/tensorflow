// Generate the actual module by running
// tf-opt -tf-xla-call-module-serialization
// on this module:
//
// module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 2246 : i32}} {
//   func.func private @_stablehlo_main_add(%dummy: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {jax.result_info = "", mhlo.sharding = "{devices=[2,1]<=[2]}"}) attributes {_from_xla_call_module} {
//     %0 = stablehlo.add %arg0, %arg1 : tensor<4x2xi32>
//     return %0 : tensor<4x2xi32>
//   }

//   func.func private @_stablehlo_main_sub(%dummy: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {jax.arg_info = "x", mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {jax.result_info = "", mhlo.sharding = "{devices=[2,1]<=[2]}"}) attributes {_from_xla_call_module} {
//     %0 = stablehlo.negate %arg1 : tensor<4x2xi32>
//     %1 = stablehlo.add %arg0, %0 : tensor<4x2xi32>
//     return %1 : tensor<4x2xi32>
//   }

//   func.func @main(%arg0: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {mhlo.sharding = ""}, tensor<4x2xi32> {mhlo.sharding = ""})  attributes {tfrt_ifrt_serving.program_id = -6498204404019954638 : i64, __tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 iota_reshape_dims: 2 iota_transpose_perm: 0 } is_bounded_dynamic_dim: false } args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 iota_reshape_dims: 2 iota_transpose_perm: 0 } is_bounded_dynamic_dim: false } retvals { sharding { } } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 use_spmd_for_xla_partitioning: true  use_shardy_partitioner: true compile_options { } "} {
//     %0 = "tf.XlaSharding"(%arg0) <{_XlaSharding = "{devices=[2,1]<=[2]}", sharding = "{devices=[2,1]<=[2]}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
//     %1 = "tf.XlaSharding"(%arg1) <{_XlaSharding = "{devices=[2,1]<=[2]}", sharding = "{devices=[2,1]<=[2]}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
//     %2 = "tf.XlaCallModule"(%0, %1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_sub, _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, _stablehlo_version = "1.11.0", module = "", platforms = ["TPU", "CUDA"], version = 10 : i64, use_shardy_partitioner = true} : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
//     %3 = "tf.XlaCallModule"(%0, %1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_add, _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, _stablehlo_version = "1.11.0", module = "", platforms = ["TPU", "CUDA"], version = 10 : i64, use_shardy_partitioner = true} : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
//     return %2, %3 : tensor<4x2xi32>, tensor<4x2xi32>
//   }
// }

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 2246 : i32}} {
  func.func @main(%arg0: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}, %arg1: tensor<4x2xi32> {mhlo.sharding = "{devices=[2,1]<=[2]}"}) -> (tensor<4x2xi32> {mhlo.sharding = ""}, tensor<4x2xi32> {mhlo.sharding = ""}) attributes {__tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 iota_reshape_dims: 2 iota_transpose_perm: 0  } is_bounded_dynamic_dim: false } args { dtype: DT_INT32 kind: PARAMETER sharding { type: OTHER tile_assignment_dimensions: 2 tile_assignment_dimensions: 1 iota_reshape_dims: 2 iota_transpose_perm: 0 } is_bounded_dynamic_dim: false } retvals { sharding { } } retvals { sharding { } } num_replicas: 1 num_cores_per_replica: 2 use_spmd_for_xla_partitioning: true compile_options { } ", tfrt_ifrt_serving.program_id = -6498204404019954638 : i64} {
    %0 = "tf.XlaSharding"(%arg0) <{_XlaSharding = "{devices=[2,1]<=[2]}", sharding = "{devices=[2,1]<=[2]}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %1 = "tf.XlaSharding"(%arg1) <{_XlaSharding = "{devices=[2,1]<=[2]}", sharding = "{devices=[2,1]<=[2]}"}> : (tensor<4x2xi32>) -> tensor<4x2xi32>
    %2 = "tf.XlaCallModule"(%0, %1) <{Sout = [#tf_type.shape<?>], dim_args_spec = [], function_list = [], module = "ML\EFR\0DStableHLO_v1.11.0\00\01\1B\05\01\05\0B\01\03\0B\03\09\0F\13\17\1B\03U7\0B\01\19\0B\13\0B\0F\13\13\13\13\17\13\13\13\03\1F\1B\0B\0B\0B\17\13\0B\0B\0B\0B\0B\0F\1B\0B\0B\01\03\0F\03\09\17\0F\07\1F\02\D7\05\0F\03\03\05\07\05\11\11\01\05\17\01!\15\17\01\0F\07\17\01\0FU\17\01\0F\D3\17\01\0F\FA\02\17\01\11\15\17\01\13\15\17\01\15\0B\0D\05)+\1B\1D\1D\13\1D\15\1D\17\03\07#\19\19\0D\03%'\1D\19\1D\1B\1D\1D\1D\1F#\09\03\031\0D\053\1F\1B\1D\1D!\1D#\01\02\04)\05\11\09\07)\01\07\1B\11\07\05\03\03\03\03\04e\05\01Q\09\03\01\07\04S\03\01\05\03P\0B\03\07\04?\03\0B\0F\07\0B\0D\07\0F\07\11\00\05\06\13\03\03\03\05\07\06\15\03\03\05\03\07\09\04\17\03\09\06\03\01\05\01\00\1E\04%\0B!\05\1B!)\03+\1D){\15\0F\15\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00negate_v1\00add_v1\00return_v1\00third_party/tensorflow/core/tfrt/ifrt/testdata/m_shardy.mlir\00mhlo.num_partitions\00mhlo.sharding\00{devices=[2,1]<=[2]}\00\00jax.global_constant\00_platform_index\00jax.arg_info\00x\00jax.result_info\00main\00\08\15\05\05\01\01\0B!-/5\1F", platforms = ["TPU", "CUDA"], use_shardy_partitioner = true, version = 10 : i64}> : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    %3 = "tf.XlaCallModule"(%0, %1) <{Sout = [#tf_type.shape<?>], dim_args_spec = [], function_list = [], module = "ML\EFR\0DStableHLO_v1.11.0\00\01\19\05\01\05\09\01\03\0B\03\07\0F\13\17\03S5\0B\01\17\0B\13\0B\0F\13\13\13\13\17\13\13\03\1F\1B\0B\0B\0B\17\13\0B\0B\0B\0B\0B\0F\1B\0B\0B\01\03\0F\03\09\17\0F\07\1F\02\CF\05\0D\03\03\05\07\05\0F\11\01\05\17\01#\15\17\01\05\07\17\01\05U\17\01\05\D3\17\01\05\FA\02\17\01\07\15\17\01\09\0B\0D\05')\19\1B\1D\11\1D\13\1D\15\03\07!\17\17\0D\03#%\1D\17\1D\19\1D\1B\1D\1D#\09\03\03/\0D\051\1D\19\1B\1D\1F\1D!\01\02\04)\05\11\09\07)\01\07\1B\11\07\05\03\03\03\03\04W\05\01Q\09\03\01\07\04E\03\01\05\03P\0B\03\07\041\03\09\0B\07\0B\0D\07\0F\07\11\00\05\06\13\03\03\05\03\05\07\04\15\03\07\06\03\01\05\01\00\F2\03#\0B!\05\1B!)\03+\1D){\15\0F\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00add_v1\00return_v1\00third_party/tensorflow/core/tfrt/ifrt/testdata/m_shardy.mlir\00mhlo.num_partitions\00mhlo.sharding\00{devices=[2,1]<=[2]}\00\00jax.global_constant\00_platform_index\00jax.arg_info\00x\00jax.result_info\00main\00\08\15\05\05\01\01\0B\1F+-3\1D", platforms = ["TPU", "CUDA"], use_shardy_partitioner = true, version = 10 : i64}> : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
    return %2, %3 : tensor<4x2xi32>, tensor<4x2xi32>
  }
}




