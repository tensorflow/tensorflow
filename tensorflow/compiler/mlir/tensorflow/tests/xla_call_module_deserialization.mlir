// RUN: tf-opt %s -split-input-file -tf-xla-call-module-deserialization | FileCheck %s

// Tests that `tf.XlaCallModule` with both StableHLO module and TF host
// callbacks can be deserialized.

// CHECK-LABEL: module
module {
  // CHECK-LABEL: func private @__tf_host_callback
  func.func private @__tf_host_callback(%arg0: tensor<?xi32>, %arg1: tensor<*xi32>) {
    // CHECK: tf.StreamResults

    // StreamResults is a pseudo op in this test.
    "tf.StreamResults"(%arg0, %arg1) : (tensor<?xi32>, tensor<*xi32>) -> ()
    func.return
  }

  // CHECK-LABEL: func @main
  // CHECK-SAME:    %[[ARG0:.*]]: tensor<10xi32>, %[[ARG1:.*]]: tensor<10xi32>
  func.func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32> {
    // CHECK:      %[[RESULT:.*]] = "tf.XlaCallModule"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME:   _entry_function = @_stablehlo_main_0
    // CHECK-SAME:   function_list = [@__tf_host_callback]
    // CHECK-SAME:   module = ""

    // `module` is stablehlo bytecode for:
    //  func.func @main(%arg0: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %arg1: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) {
    //    stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {caller_name = "__tf_host_callback"}} : (tensor<?xi32>, tensor<*xi32>) -> ()
    //    return %arg0 : tensor<?xi32>
    //  }
    %0 = "tf.XlaCallModule"(%arg0, %arg1) {Sout = [#tf_type.shape<?>], dim_args_spec = [], function_list = [@__tf_host_callback], module = "ML\EFR\03MLIRtrunk\00\01\17\05\01\05\01\03\05\03\07\07\09\0B\03\95}\09\01U\07\0F3\0B\0B\0B\0B\0B\0F\0F\0F\13\0F\0B\0F\0B\13\0B\0F\0B\13\0F\0B\0F\0B\13\0F\0BS\0B\0B\0B\0B\0B\0B\0B\0B\0B\0F\0B\17\0B\03)\0B\0B\13\1B\0B\0B\0B\0B\0B\0B\0F\13\0B\0B\0B\0B\0B\13\0B\0B\03\093\07\0B\1B\026\03\1F\1D#\01\03\0B\07Y\09g\0Bi\0Do\0FW\05\0D\05\0F\05\11\05\13\05\15\15\133\15\15)\15\17!\19\05\19\1D\1D\1B\01\05\17\1D\1F\01\05\19\19\05\03%\05\1B\1D'\01\05\1D\19\05+/\1D-\01\05\1F\1D1\01\05!\19\05\035\1D7\01\05#\03\13;q=W?sAUCuEUGUIUKw\05%\05'\05)\05+\05-\05/\051\053\055\1DOQ\057\17S\CE\05\01\059\03\01\1D;\03\05[e\0D\05]_ac\1D=\1D?\1DA\1DC\0D\01#\07\03\03k\0D\03mW\1DE\1DG\0B\05\1DI\05\03\0D\03y{\1DK\1DM)\03\00\FF\FF\FF\FF\FF\FF\FF\FF\03\1B3\03\11\05\01\05\03\01\04C\05\01\10\01\07\03\01\05\03\11\11\05\05\03\05\0B\05\01\01\05\01\05\05M9\05\01\03\07\04\01\03\01\06\03\01\05\01\00v\13O'\19)\0B!\1B\1D\05\1B\03i\D2\06%\1F/!!)#\1F\19\85\A5)\8B3{\1F\1F\13\15\1D\15\15\1F\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00custom_call_v1\00return_v1\00arg_attrs\00function_type\00res_attrs\00sym_name\00sym_visibility\00XlaCallModule:\00XlaCallModule@__inference_converted_fun_tf_29_cycles_removed\00StatefulPartitionedCall:\00func_call@tpu_fn_icv2___inference_converted_fun_tf_29_cycles_removed\00TPUPartitionedCall:\00tpu_partitioned_call@tpu_call_icv2___inference_converted_fun_tf_29_cycles_removed\00StatefulPartitionedCall@__inference_computation_32_cycles_removed\00api_version\00backend_config\00call_target_name\00called_computations\00has_side_effect\00operand_layouts\00output_operand_aliases\00result_layouts\00tf.backend_config\00jit(tpu_func)/jit(main)/call_tf[callable_flat_tf=<function outer_factory.<locals>.inner_factory.<locals>.tf__make_call.<locals>.callable_flat_tf at 0x7fcb4f790d30> function_flat_tf=<xxxxxxx.xxxxxxxxxxx.tensorflow.python.eager.polymorphic_function.polymorphic_function.Function object at 0x7fcb54454520> args_flat_sig_tf=(TensorSpec(shape=(None,), dtype=tf.int32, name=None),) output_avals=None has_side_effects=True call_tf_graph=True]\00third_party/tensorflow/python/autograph/impl/api.py\00\00jax.arg_info\00x\00mhlo.sharding\00{replicated}\00jax.result_info\00main\00tf.call_tf_function\00caller_name\00__tf_host_callback\00", platforms = [], version = 5 : i64} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    // CHECK:     return %[[RESULT]]
    func.return %0 : tensor<10xi32>
  }

  // CHECK-LABEL: func private @_stablehlo_main_0
  // CHECK-SAME:    (%[[ARG0:.*]]: tensor<?xi32> {jax.arg_info = "x", mhlo.sharding = "{replicated}"}, %[[ARG1:.*]]: tensor<*xi32>) -> (tensor<?xi32> {jax.result_info = ""}) attributes {_from_xla_call_module} {
  // CHECK:         stablehlo.custom_call @tf.call_tf_function(%[[ARG0]], %[[ARG1]]) {api_version = 2 : i32, has_side_effect = true, tf.backend_config = {caller_name = "__tf_host_callback"}} : (tensor<?xi32>, tensor<*xi32>) -> ()
  // CHECK:         return %arg0 : tensor<?xi32>
  // CHECK:       }
}
