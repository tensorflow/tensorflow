// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text-with-layouts %s | FileCheck %s

// Checks exporting layouts

// CHECK:  HloModule
func @main(%arg0: tensor<128x224x224x4xf16>, %arg1: tensor<64x7x7x4xf16>) -> tensor<128x64x112x112xf16> {
  // CHECK: %convolution.{{.*}} = f16[128,64,112,112]{1,3,2,0} convolution{{.*}}op_name="root.42"
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 3 : i64,
      input_spatial_dimensions = dense<[ 1, 2 ]> : tensor<2xi64>,
      kernel_input_feature_dimension = 3 : i64,
      kernel_output_feature_dimension = 0 : i64,
      kernel_spatial_dimensions = dense<[ 1, 2 ]> : tensor<2xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 1 : i64,
      output_spatial_dimensions = dense<[ 2, 3 ]> : tensor<2xi64>
    },
    feature_group_count = 1 : i64,
    lhs_dilations = dense<1> : tensor<2xi64>,
    minor_to_major = dense<[ 1, 3, 2, 0 ]> : tensor<4xindex>,
    padding = dense<3> : tensor<2x2xi64>,
    precision_config = [ "DEFAULT", "DEFAULT" ],
    rhs_dilations = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>
  } : (tensor<128x224x224x4xf16>, tensor<64x7x7x4xf16>)-> tensor<128x64x112x112xf16> loc("root.42")

  // CHECK: s32[1,1]{0,1} constant({ {42} })
  %cst_1 = "std.constant"() {value = dense<[[42]]> : tensor<1x1xi32>, minor_to_major = dense<[0, 1]> : tensor<2xindex>} : () -> tensor<1x1xi32>

  return %0 : tensor<128x64x112x112xf16>
}
