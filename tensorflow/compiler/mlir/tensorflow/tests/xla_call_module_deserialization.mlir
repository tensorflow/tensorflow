// RUN: tf-opt %s -split-input-file -tf-xla-call-module-deserialization | FileCheck %s

// Tests that `tf.XlaCallModule` with both StableHLO module and TF function
// calls can be deserialized.

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func private @_tf_func
  func.func private @_tf_func(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) {
    // CHECK: tf.StreamResults

    // StreamResults is a pseudo op in this test.
    "tf.StreamResults"(%arg0, %arg1) : (tensor<?xi32>, tensor<*xi32>) -> ()
    func.return
  }

  // CHECK-LABEL: func @main
  // CHECK-SAME:    %[[ARG0:.*]]: tensor<10xi32>, %[[ARG1:.*]]: tensor<10xi32>
  func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    // CHECK:      %[[RESULT:.*]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-NOT:    function_list
    // CHECK-SAME:   module = ""
    // CHECK-SAME:   _entry_function = @main_0,

    // `module` is stablehlo bytecode for:
    //  func.func @main(%arg0: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) {
    //    stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @_tf_func}} : (tensor<?xi32>, tensor<*xi32>) -> ()
    //    return %arg0 : tensor<?xi32>
    //  }
    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], function_list = [@_tf_func], module = "ML\EFR\07StableHLO_v0.12.0\00\01\19\05\01\05\01\03\05\03\09\07\09\0B\0D\03\8Fm\0F\01?\0B\07\0B\0B\0B\0B\0B\13\0B\0F\133\133\13\13S\0B\0B\0B\0B\0B\0B\0B\0B\0B\13\13\0B\13\13\03/\0B\0B\0B\13\1B\0B\0B\0B\0B\0B\0B\0F\13\0B\0B\0B\0B\0B\0B\0B\13\0B\0F\01\03\0F\03\0D3\07\0B\1B\17\07\02:\03\05\0F\1F\05\11\05\13\05\15\05\17\05\19\03\03\11\13\05\1B\11\01\05\17\01S\15\03\0B\05E\07S\09U\0B[\0DA\17\011\07\03\0B\05?\07]\09?\0BC\0D_\17\01'\07\17\01)\0B\03\13#a%A'c)?+e-?/?1?3g\05\1D\05\1F\05!\05#\05%\05'\05)\05+\05-\17\013\0B\03\039C\05/\17\015\1B\17\017\0B\03\01\1D1\1D3\03\05GQ\0D\05IKMO\1D5\1D7\1D9\1D;\0D\01#\09\03\03W\0D\03YA\1D=\1D?#\0B\1DA\0B\05\1DC\05\03\0D\03ik\1DE\13\0D\01\01\02\04)\03\00\FF\FF\FF\FF\FF\FF\FF\FF\05\1B3\05\11\05\03\07\03\03\11\03\03\03\03\1D\04}\05\01\11\15\0F\07\04m\03\01\09\03\11\19\17\05\03\07\0F\05\03\03\07\03\00\07\055!\05\01\03\09\07;7\03\03\03\01\05\04=\03\05\03\11\1D\1B\05\03\03\07\03\03\03\00\05\04\1F\03\01\06\03\01\05\01\00\9E\07G\1B)\11\0B!\1B\1D\05\1B\1B\03\0F%\1F/!!)#\1F\19)\1F\13\15\1D\15G\11\1F\15\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00return_v1\00custom_call_v1\00call_v1\00xla_call_module_serialization.mlir\00arg_attrs\00function_type\00res_attrs\00sym_name\00sym_visibility\00mhlo.num_partitions\00api_version\00backend_config\00call_target_name\00called_computations\00has_side_effect\00operand_layouts\00output_operand_aliases\00result_layouts\00tf.backend_config\00callee\00\00_stablehlo_f\00jax.arg_info\00x\00mhlo.sharding\00{replicated}\00jax.result_info\00main\00private\00tf.call_tf_function\00called_index\00", platforms = [], version = 5 : i64} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    // CHECK:     return %[[RESULT]]
    func.return %0 : tensor<10xi32>
  }

  // CHECK-LABEL: func @foo
  // CHECK-SAME:    %[[ARG0:.*]]: tensor<10xi32>, %[[ARG1:.*]]: tensor<10xi32>
  func.func @foo(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    // CHECK:      %[[RESULT:.*]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-NOT:    function_list
    // CHECK-SAME:   module = ""
    // CHECK-SAME:   _entry_function = @main_1,

    // `module` is stablehlo bytecode for:
    //  func.func @main(%arg0: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) {
    //    stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @_tf_func}} : (tensor<?xi32>, tensor<*xi32>) -> ()
    //    return %arg0 : tensor<?xi32>
    //  }
    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], function_list = [@_tf_func], module = "ML\EFR\07StableHLO_v0.12.0\00\01\19\05\01\05\01\03\05\03\09\07\09\0B\0D\03\8Fm\0F\01?\0B\07\0B\0B\0B\0B\0B\13\0B\0F\133\133\13\13S\0B\0B\0B\0B\0B\0B\0B\0B\0B\13\13\0B\13\13\03/\0B\0B\0B\13\1B\0B\0B\0B\0B\0B\0B\0F\13\0B\0B\0B\0B\0B\0B\0B\13\0B\0F\01\03\0F\03\0D3\07\0B\1B\17\07\02:\03\05\0F\1F\05\11\05\13\05\15\05\17\05\19\03\03\11\13\05\1B\11\01\05\17\01S\15\03\0B\05E\07S\09U\0B[\0DA\17\011\07\03\0B\05?\07]\09?\0BC\0D_\17\01'\07\17\01)\0B\03\13#a%A'c)?+e-?/?1?3g\05\1D\05\1F\05!\05#\05%\05'\05)\05+\05-\17\013\0B\03\039C\05/\17\015\1B\17\017\0B\03\01\1D1\1D3\03\05GQ\0D\05IKMO\1D5\1D7\1D9\1D;\0D\01#\09\03\03W\0D\03YA\1D=\1D?#\0B\1DA\0B\05\1DC\05\03\0D\03ik\1DE\13\0D\01\01\02\04)\03\00\FF\FF\FF\FF\FF\FF\FF\FF\05\1B3\05\11\05\03\07\03\03\11\03\03\03\03\1D\04}\05\01\11\15\0F\07\04m\03\01\09\03\11\19\17\05\03\07\0F\05\03\03\07\03\00\07\055!\05\01\03\09\07;7\03\03\03\01\05\04=\03\05\03\11\1D\1B\05\03\03\07\03\03\03\00\05\04\1F\03\01\06\03\01\05\01\00\9E\07G\1B)\11\0B!\1B\1D\05\1B\1B\03\0F%\1F/!!)#\1F\19)\1F\13\15\1D\15G\11\1F\15\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00return_v1\00custom_call_v1\00call_v1\00xla_call_module_serialization.mlir\00arg_attrs\00function_type\00res_attrs\00sym_name\00sym_visibility\00mhlo.num_partitions\00api_version\00backend_config\00call_target_name\00called_computations\00has_side_effect\00operand_layouts\00output_operand_aliases\00result_layouts\00tf.backend_config\00callee\00\00_stablehlo_f\00jax.arg_info\00x\00mhlo.sharding\00{replicated}\00jax.result_info\00main\00private\00tf.call_tf_function\00called_index\00", platforms = [], version = 5 : i64} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    // CHECK:     return %[[RESULT]]
    func.return %0 : tensor<10xi32>
  }

  // CHECK-LABEL: func private @main_0
  // CHECK-SAME:    (%[[ARG0:.*]]: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %[[ARG1:.*]]: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) attributes {_from_xla_call_module} {
  // CHECK:         stablehlo.custom_call @tf.call_tf_function(%[[ARG0]], %[[ARG1]]) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @_tf_func}} : (tensor<?xi32>, tensor<*xi32>) -> ()
  // CHECK:         return %arg0 : tensor<?xi32>
  // CHECK:       }

  // CHECK-LABEL: func private @main_1
  // CHECK-SAME:    (%[[ARG0:.*]]: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %[[ARG1:.*]]: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) attributes {_from_xla_call_module} {
  // CHECK:         stablehlo.custom_call @tf.call_tf_function(%[[ARG0]], %[[ARG1]]) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {called_func = @_tf_func}} : (tensor<?xi32>, tensor<*xi32>) -> ()
  // CHECK:         return %arg0 : tensor<?xi32>
  // CHECK:       }
}
