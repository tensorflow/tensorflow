// RUN: tf-opt %s -tfl-lift-tflite-flex-ops | FileCheck %s

// CHECK-LABEL: TfAdd
func @TfAdd(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
  %0 = "tfl.custom"(%arg0, %arg1) {
    custom_code = "FlexAdd",
    custom_option = opaque<"tfl", "0x03416464001412034164641A001A002A070A015412023002320000021B171414042801"> : tensor<35xi8>
  } : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>

// CHECK: "tf.Add"(%arg0, %arg1) {T = f64}  : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}



// CHECK-LABEL: TfBatchMatMulV2
func @TfBatchMatMulV2(%arg0: tensor<4x128x2xf32>, %arg1:  tensor<2x1xf32>) -> tensor<4x128x1xf32> {
  %0 = "tfl.custom"(%arg0, %arg1) {
    custom_code = "FlexBatchMatMulV2",
    custom_option = opaque<"tfl", "0x0D42617463684D61744D756C56320038120D42617463684D61744D756C56321A001A002A070A0154120230012A0B0A0561646A5F78120228002A0B0A0561646A5F791202280032000002493B1414042801"> : tensor<81xi8>
  } : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>

// CHECK: "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false} : (tensor<4x128x2xf32>, tensor<2x1xf32>) -> tensor<4x128x1xf32>
  return %0 : tensor<4x128x1xf32>
}


// CHECK-LABEL: TfTensorArrayV3
func @TfTensorArrayV3(%arg0: tensor<i32>) -> tensor<f32> {
  %0:2 = "tfl.custom"(%arg0) {
    custom_code = "FlexTensorArrayV3",
    custom_option = opaque<"tfl", "0x0D54656E736F724172726179563300A8120D54656E736F72417272617956331A002A1E0A186964656E746963616C5F656C656D656E745F736861706573120228012A120A0C64796E616D69635F73697A65120228002A1D0A1174656E736F725F61727261795F6E616D651208120673616D706C652A160A10636C6561725F61667465725F72656164120228012A0B0A056474797065120230012A1B0A0D656C656D656E745F7368617065120A3A08120208081202080132000002B9AB1414042801"> : tensor<193xi8>
  } : (tensor<i32>) -> (tensor<2xi32>, tensor<*xf32>)

// CHECK: "tf.TensorArrayV3"
// CHECK-SAME: : (tensor<i32>) -> (tensor<2x!tf_type.resource>, tensor<f32>)

  %1 = "tfl.cast"(%0#1) : (tensor<*xf32>) -> tensor<f32>
  return %1 : tensor<f32>
}

