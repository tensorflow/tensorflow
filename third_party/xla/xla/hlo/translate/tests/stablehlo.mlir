// RUN: xla-translate --stablehlo-to-hlo-text -split-input-file %s | FileCheck %s
// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo=convert-xla-supported-stablehlo=false -split-input-file %s | FileCheck %s --check-prefix CHECK-DIRECT

// Tests for all stablehlo ops to validate stablehlo -> hlo conversion.


// CHECK-LABEL: HloModule

// CHECK: %[[ARG0:.*]] = f32[4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[4] parameter(1)
// CHECK: ROOT %add.3 = f32[4] add(%[[ARG0]], %[[ARG1]])
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>  func.return %0 : tensor<4xf32>
}
// CHECK-DIRECT: stablehlo.add

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3,4]{1,0})->s32[1,2]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3,4] parameter(0)
// CHECK-NEXT:  ROOT %[[slice_2:[^ ]+]] = s32[1,2] slice(%[[Arg_0_1]]), slice={[1:2:1], [0:4:2]},
func.func @main(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = stablehlo.slice %arg0 [1:2, 0:4:2] : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}
// CHECK-DIRECT: stablehlo.slice

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3,4]{1,0}, s64[], s64[])->s32[1,4]{1,0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s64[] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s64[] parameter(2)
// CHECK-NEXT:  ROOT %[[dynamic_slice_4:[^ ]+]] = s32[1,4] dynamic-slice(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), dynamic_slice_sizes={1,4},
func.func @main(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [1, 4] : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}
// CHECK-DIRECT: stablehlo.dynamic_slice

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[4]{0})->s32[1,2,3,4]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[4] parameter(0)
// CHECK-NEXT:  ROOT %[[broadcast_2:[^ ]+]] = s32[1,2,3,4] broadcast(%[[Arg_0_1]]), dimensions={3}
func.func @main(%arg0: tensor<4xi32>) -> tensor<1x2x3x4xi32> {
  %0 = stablehlo.broadcast %arg0, sizes = [1, 2, 3] : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %0 : tensor<1x2x3x4xi32>
}
// CHECK-DIRECT: stablehlo.broadcast

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[1]{0})->f32[1,10]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[1] parameter(0)
// CHECK-NEXT:  ROOT %[[broadcast_2:[^ ]+]] = f32[1,10] broadcast(%[[Arg_0_1]]), dimensions={0}
func.func @main(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xf32>) -> tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
}
// CHECK-DIRECT: stablehlo.broadcast_in_dim

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[100,26,26,32]{3,2,1,0}, f32[3,3,1,32]{3,2,1,0})->f32[100,28,28,1]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[100,26,26,32] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[3,3,1,32] parameter(1)
// CHECK-NEXT:  ROOT %[[convolution_3:[^ ]+]] = f32[100,28,28,1] convolution(%[[Arg_0_1]], %[[Arg_1_2]]), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f, metadata=

module {
  func.func @main(%arg0: tensor<100x26x26x32xf32>, %arg1: tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  return %0 : tensor<100x28x28x1xf32>
  }
}
// CHECK-DIRECT: stablehlo.convolution

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s8[100,26,26,32]{3,2,1,0}, s8[3,3,1,32]{3,2,1,0})->s32[100,28,28,1]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s8[100,26,26,32] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s8[3,3,1,32] parameter(1)
// CHECK-NEXT:  ROOT %[[convolution_3:[^ ]+]] = s32[100,28,28,1] convolution(%[[Arg_0_1]], %[[Arg_1_2]]), window={size=3x3 pad=2_2x2_2}, dim_labels=b01f_01oi->b01f, metadata=

module {
  func.func @main(%arg0: tensor<100x26x26x32xi8>, %arg1: tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  return %0 : tensor<100x28x28x1xi32>
  }
}
// CHECK-DIRECT: stablehlo.convolution

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s8[100,26,26,32]{3,2,1,0}, s8[3,3,1,32]{3,2,1,0})->s32[100,28,28,1]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s8[100,26,26,32] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s8[3,3,1,32] parameter(1)
// CHECK-NEXT:  ROOT %[[convolution_3:[^ ]+]] = s32[100,28,28,1] convolution(%[[Arg_0_1]], %[[Arg_1_2]]), window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f, metadata=

module {
  func.func @main(%arg0: tensor<100x26x26x32xi8>, %arg1: tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, o, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[2, 2], [2, 2]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [true, true]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  return %0 : tensor<100x28x28x1xi32>
  }
}
// CHECK-DIRECT: stablehlo.convolution
