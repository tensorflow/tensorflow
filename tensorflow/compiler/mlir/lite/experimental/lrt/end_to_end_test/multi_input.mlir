// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin --model_path=%t --dry_run | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin --model_path=%t --dry_run=false --soc_man=ExampleSocManufacturer | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s --check-prefix=METADATA

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>) -> tensor<2x2xf32> {
    %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %1 = "tfl.mul"(%0, %arg1) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    %2 = tfl.mul %1, %1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %3 = tfl.add %2, %2 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
}

// CHECK: func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>) -> tensor<2x2xf32> {
// CHECK:     %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     %1 = "tfl.custom"(%0, %arg1) <{custom_code = {{.*}}, custom_option = #tfl<const_bytes : "{{.*}}">}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK:     %2 = tfl.add %1, %1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     return %2 : tensor<2x2xf32>

// CHECK: func.func private @fn_1(%arg0: tensor<2x2xf32>, %arg1: tensor<f32>) -> tensor<2x2xf32> {
// CHECK:     %0 = tfl.mul(%arg0, %arg1) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
// CHECK:     %1 = tfl.mul %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     return %1 : tensor<2x2xf32>

// METADATA: ExampleSocManufacturer = "Partition_0_with_2_muls:"
// METADATA: "tfl.custom"(%0, %arg1) <{custom_code = "ExampleSocManufacturer"