// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin --model_path=%t --dry_run | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin --model_path=%t --dry_run=false --soc_man=ExampleSocManufacturer | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s --check-prefix=METADATA

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:   %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = {{.*}}, custom_option = #tfl<const_bytes : "{{.*}}">}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %0 : tensor<4xf32>

// CHECK: func.func private @fn_1(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:   %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
// CHECK:   return %0 : tensor<4xf32>

// METADATA: ExampleSocManufacturer = "Partition_0_with_1_muls:"
// METADATA: "tfl.custom"({{.*}}) <{custom_code = "ExampleSocManufacturer"