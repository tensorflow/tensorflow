// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text --with-layouts=true --print-layouts=true %s | FileCheck %s

// Checks exporting layouts

// CHECK:  HloModule
func @main(%arg0: tensor<128x224x224x4xf16>, %arg1: tensor<64x7x7x4xf16>) -> tensor<128x64x112x112xf16> {
  // CHECK: %convolution.{{.*}} = f16[128,64,112,112]{1,3,2,0} convolution{{.*}}op_name="root.42"
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 0,
      kernel_spatial_dimensions = [1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 1,
      output_spatial_dimensions = [2, 3]
    >,
    feature_group_count = 1 : i64,
    lhs_dilations = dense<1> : tensor<2xi64>,
    xla_shape = "f16[128,64,112,112]{1,3,2,0}",
    padding = dense<3> : tensor<2x2xi64>,
    precision_config = [ "DEFAULT", "DEFAULT" ],
    rhs_dilations = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>
  } : (tensor<128x224x224x4xf16>, tensor<64x7x7x4xf16>)-> tensor<128x64x112x112xf16> loc("root.42")

  // CHECK: s32[1,1]{0,1} constant({ {42} })
  %cst_1 = "arith.constant"() {value = dense<[[42]]> : tensor<1x1xi32>, xla_shape = "s32[1,1]{0,1}"} : () -> tensor<1x1xi32>

  return %0 : tensor<128x64x112x112xf16>
}
