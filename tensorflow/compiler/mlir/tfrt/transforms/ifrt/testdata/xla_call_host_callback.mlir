module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1758 : i32}} {

  func.func private @callee(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) {
    "tf.XlaHostCompute"(%arg0, %arg1) <{ancestors = [], key = "@test_callee", recv_key = "", send_key = "", shapes = []}> {_xla_original_oc_node_name = "hcb0", _xla_token_input_nodes = ["_xla_token_arg_node"]} : (tensor<?xi32>, tensor<*xi32>) -> ()
    return
  }

  // The mlir module in XlaCallModule is serialized from:
  //
  // func.func private @_stablehlo_main_0(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) -> () attributes {_from_xla_call_module} {
  //   stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @callee}} : (tensor<?xi32>, tensor<*xi32>) -> ()
  //   return
  // }
  //
  // func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) -> () {
  //   "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], _entry_function = @_stablehlo_main_0, _stablehlo_module_attrs = { mhlo.num_partitions = 1 }, module = "", platforms = [], version = 5 : i64} : (tensor<?xi32>, tensor<*xi32>) -> ()
  //   func.return
  // }
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) attributes {tfrt_ifrt_serving.program_id = -2372940092539171444 : i64, __tpu_compile_metadata_text = "args { dtype: DT_INT32 kind: PARAMETER sharding { } }  args { dtype: DT_INT32 kind: PARAMETER sharding { } } num_replicas: 1 num_cores_per_replica: 1 use_spmd_for_xla_partitioning: true compile_options { }"} {
    "tf.XlaCallModule"(%arg0, %arg1) <{Sout = [], dim_args_spec = [], function_list = [@callee], module = "ML\EFR\0DStableHLO_v0.17.6\00\01\19\05\01\05\09\01\03\0B\03\07\0F\13\17\03M-\0D\01\19\0B\13\0B\0F\13\13\13\13\13\0B\13\13\03\15\0B\0B\0B\0B\13\0B\0F\0B\0B\0B\01\03\0F\03\0B3\07\0B\17\07\02\B1\05\0D\03\03\05\07\05\0F\11\01\05\17\01A\0B\17\01!\07\17\01!Q\17\01!}\03\03\13!\05\11\17\01#\0B\17\01%\0B\03\01\1D\13#\09\1D\15\0D\03#%\1D\17\13\0B\01\0B\05\1D\19\05\03\01\02\04)\03\00\FF\FF\FF\FF\FF\FF\FF\FF\05\1B3\05\11\05\03\07\01\1D\04O\05\01Q\09\03\01\07\04=\03\01\05\03P\0B\03\07\04)\03\05\0B\05\07\0D\0F\0F\00\05E\15\11\05\05\01\03\07\00\17\06\03\01\05\01\00j\03\1B)\1B\0B\03%)\95\15\1F\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00custom_call_v1\00return_v1\00experimental/users/deqiangc/mira/testdata/xla_call_module_serialized.mlir\00mhlo.num_partitions\00tf.backend_config\00\00main\00called_index\00tf.call_tf_function\00\08'\07\05\01\01\0B\19\1D\19\1F\1B\11'\1B)\19+\19\19\19", platforms = [], version = 5 : i64}> : (tensor<?xi32>, tensor<*xi32>) -> ()
    return
  }
}
