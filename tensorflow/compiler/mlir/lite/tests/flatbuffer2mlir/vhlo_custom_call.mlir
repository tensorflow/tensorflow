// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --disable-vhlo-to-stablehlo=true - -o - | FileCheck %s
// test stablehlo roundtrip

//test TF ops wrapped in stablehlo custom_call

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}

//CHECK:func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> attributes {tf.entry_function = {inputs = "arg0", outputs = "arg0"}} {
//CHECK: return %arg0 : tensor<4xi8>
//CHECK:}

func.func @custom_tf_op(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
  %0 = "vhlo.custom_call_v1" (%arg0, %arg1) <{api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>, 
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"tf.ResizeBilinear">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<false>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>}> {align_corners = #vhlo.bool_v1<true>, device = #vhlo.string_v1<"">, half_pixel_centers = #vhlo.bool_v1<false>} : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32>
  return %0 : tensor<1x1600x1x1xf32>
}

//CHECK:func.func private @custom_tf_op(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
//CHECK: %0 = "vhlo.custom_call_v1"(%arg0, %arg1) <{
//CHECK-SAME: api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
//CHECK-SAME: backend_config = #vhlo.string_v1<"">,
//CHECK-SAME: call_target_name = #vhlo.string_v1<"tf.ResizeBilinear">,
//CHECK-SAME: called_computations = #vhlo.array_v1<[]>,
//CHECK-SAME: has_side_effect = #vhlo.bool_v1<false>,
//CHECK-SAME: operand_layouts = #vhlo.array_v1<[]>,
//CHECK-SAME: output_operand_aliases = #vhlo.array_v1<[]>,
//CHECK-SAME: result_layouts = #vhlo.array_v1<[]>}> {align_corners = #vhlo.bool_v1<true>, device = #vhlo.string_v1<"">, half_pixel_centers = #vhlo.bool_v1<false>} : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32> 
//CHECK-NEXT: return %0 : tensor<1x1600x1x1xf32> 
//CHECK-NEXT:}

func.func @custom_op_with_backend(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
  %0 = "vhlo.custom_call_v1" (%arg0, %arg1) <{api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>, 
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"custom_backend">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<true>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>}> : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32>
  return %0 : tensor<1x1600x1x1xf32>
}

//CHECK:func.func private @custom_op_with_backend(%arg0: tensor<1x320x1x1xf32>, %arg1: tensor<2xi32>) -> tensor<1x1600x1x1xf32> {
//CHECK: "vhlo.custom_call_v1"(%arg0, %arg1) <{
//CHECK-SAME:  api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
//CHECK-SAME: backend_config = #vhlo.string_v1<"">,
//CHECK-SAME: call_target_name = #vhlo.string_v1<"custom_backend">,
//CHECK-SAME: called_computations = #vhlo.array_v1<[]>,
//CHECK-SAME: has_side_effect = #vhlo.bool_v1<true>,
//CHECK-SAME: operand_layouts = #vhlo.array_v1<[]>,
//CHECK-SAME: output_operand_aliases = #vhlo.array_v1<[]>,
//CHECK-SAME: result_layouts = #vhlo.array_v1<[]>}> : (tensor<1x320x1x1xf32>, tensor<2xi32>) -> tensor<1x1600x1x1xf32> 
//CHECK-NEXT: return %0 : tensor<1x1600x1x1xf32>
//CHECK-NEXT:}