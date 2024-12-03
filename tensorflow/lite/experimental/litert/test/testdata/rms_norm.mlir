module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {CONVERSION_METADATA = "\10\00\00\00\00\00\00\00\08\00\0E\00\08\00\04\00\08\00\00\00\10\00\00\00$\00\00\00\00\00\06\00\08\00\04\00\06\00\00\00\04\00\00\00\00\00\00\00\0C\00\18\00\14\00\10\00\0C\00\04\00\0C\00\00\00zs\F5|\1F\CE)\0D\01\00\00\00\02\00\00\00\04\00\00\00\06\00\00\002.19.0\00\00", min_runtime_version = "1.10.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<8x128x1024xf32> {tf_saved_model.index_path = ["args_0"]}) -> (tensor<8x128x1024xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_args_0:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<8x128x1024xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tfl.sum"(%0, %1) <{keep_dims = false}> : (tensor<8x128x1024xf32>, tensor<1xi32>) -> tensor<8x128xf32>
    %3 = "tfl.pseudo_const"() <{value = dense<1.024000e+03> : tensor<f32>}> : () -> tensor<f32>
    %4 = tfl.div(%2, %3) <{fused_activation_function = "NONE"}> : (tensor<8x128xf32>, tensor<f32>) -> tensor<8x128xf32>
    %5 = "tfl.pseudo_const"() <{value = dense<9.99999997E-7> : tensor<f32>}> : () -> tensor<f32>
    %6 = tfl.add(%4, %5) <{fused_activation_function = "NONE"}> : (tensor<8x128xf32>, tensor<f32>) -> tensor<8x128xf32>
    %7 = "tfl.pseudo_const"() <{value = dense<[8, 128, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %8 = "tfl.reshape"(%6, %7) : (tensor<8x128xf32>, tensor<3xi32>) -> tensor<8x128x1xf32>
    %9 = "tfl.rsqrt"(%8) : (tensor<8x128x1xf32>) -> tensor<8x128x1xf32>
    %10 = tfl.mul(%arg0, %9) <{fused_activation_function = "NONE"}> : (tensor<8x128x1024xf32>, tensor<8x128x1xf32>) -> tensor<8x128x1024xf32>
    return %10 : tensor<8x128x1024xf32>
  }
}
