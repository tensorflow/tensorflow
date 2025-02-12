//RUN: tf_tfl_translate --enable-stablehlo-conversion --input-mlir %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s


module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1660 : i32}} {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> attributes {tf.entry_function = {control_outputs = "", inputs = "args_tf_0", outputs = "Identity"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Identity"(%arg0) {device = ""} : (tensor<2x3xi32>) -> tensor<2x3xi32>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.XlaSharding"(%outputs) {_XlaSharding = "", device = "", sharding = "", unspecified_dims = []} : (tensor<2x3xi32>) -> tensor<2x3xi32>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.XlaCallModule"(%outputs_0) {Sout = [#tf_type.shape<2x3>], device = "", dim_args_spec = [], disabled_checks = [], function_list = [], has_token_input_output = false, module = "ML\EFR\01StableHLO_v0.9.0\00\01\17\05\01\03\01\03\05\03\07\07\09\0B\03]?\0B\01)\07\0F\0B+\0B\0F\0B\0B\0B3\0B\0B\0B\0B\0F\0B\0F\0B\13\0B\03\17\0F\13\0B\0B\0B\0F\13\0B\0B\0B\0B\01\05\0B\0F\03\07\17\17\07\02\D7\1F\11\03\05\05\0D\03\09\09\0B\0D\03\0F\03\05\11\05\0F\11\01\00\05\11\05\13\05\15\03\0B\15)\171\193\05;\1B=\05\17\05\19\05\1B\05\1D\1D\1F\01\05\1F\1D#%\05!\17'\A9\01\05#\03\03+\0D\03-/\1D%\1D'#\07\03\035\0D\0379\1D)\1D+\1D-\1D/\01\09\01\02\02)\05\09\0D\09\11\03\05\03\05\1B\04C\05\01\11\01\07\07\03\01\05\03\11\01\13\07\03\05\0B\03\05\1D\05\06!\03\05\05\01\01\07\04\01\03\03\06\03\01\05\01\00f\051\0F\0B\03!\1B\1D[;\05\1F\15\1D\15\1D%)9\13\15\19\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00multiply_v1\00return_v1\00sym_name\00jax.uses_shape_polymorphism\00mhlo.num_partitions\00mhlo.num_replicas\00jit_jax_model\00arg_attrs\00function_type\00res_attrs\00sym_visibility\00x\00jit(jax_model)/jit(main)/mul\00experimental/users/ypang/lite/convert_ulm.py\00mhlo.sharding\00{replicated}\00jax.result_info\00\00main\00public\00", platforms = ["CPU"], version = 8 : i64} : (tensor<2x3xi32>) -> tensor<2x3xi32>
      %control_4 = tf_executor.island(%control_3) wraps "tf.NoOp"() {device = ""} : () -> ()
      %outputs_5, %control_6 = tf_executor.island wraps "tf.PreventGradient"(%outputs_2) {device = "", message = "The jax2tf-converted function does not support gradients. Use `with_gradient` parameter to enable gradients"} : (tensor<2x3xi32>) -> tensor<2x3xi32>
      %outputs_7, %control_8 = tf_executor.island wraps "tf.Identity"(%outputs_5) {device = ""} : (tensor<2x3xi32>) -> tensor<2x3xi32>
      %outputs_9, %control_10 = tf_executor.island(%control_4) wraps "tf.Identity"(%outputs_7) {device = ""} : (tensor<2x3xi32>) -> tensor<2x3xi32>
      tf_executor.fetch %outputs_9 : tensor<2x3xi32>
    }
    return %0 : tensor<2x3xi32>
  }
}

// CHECK: module attributes
// CHECK-SAME:  tfl.metadata = {{{.*}}keep_stablehlo_constant = "true"{{.*}}}
// CHECK-NEXT:  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> attributes {tf.entry_function = {inputs = "args_tf_0", outputs = "Identity"}} {
// CHECK-NEXT:    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = ""} : (tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<2x3xi32>
// CHECK-NEXT:    return %1 : tensor<2x3xi32>
// CHECK-NEXT:  }
// CHECK-NEXT: }