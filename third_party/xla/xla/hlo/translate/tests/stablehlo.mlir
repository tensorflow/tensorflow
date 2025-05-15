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

// -----
// Binary elementwise ops

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,2]{1,0}, f32[2,2]{1,0})->f32[2,2]{1,0}}

// CHECK:       ENTRY %[[$main_11:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[2,2] parameter(1)
// CHECK-NEXT:  %[[add_3:[^ ]+]] = f32[2,2] add(%[[Arg_0_1]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[atan2_4:[^ ]+]] = f32[2,2] atan2(%[[add_3]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[divide_5:[^ ]+]] = f32[2,2] divide(%[[atan2_4]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[maximum_6:[^ ]+]] = f32[2,2] maximum(%[[divide_5]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[minimum_7:[^ ]+]] = f32[2,2] minimum(%[[maximum_6]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[multiply_8:[^ ]+]] = f32[2,2] multiply(%[[minimum_7]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[power_9:[^ ]+]] = f32[2,2] power(%[[multiply_8]], %[[Arg_1_2]]),
// CHECK-NEXT:  ROOT %[[subtract_10:[^ ]+]] = f32[2,2] subtract(%[[power_9]], %[[Arg_1_2]]),

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x2xf32>
  %1 = stablehlo.atan2 %0, %arg1 : tensor<2x2xf32>
  %2 = stablehlo.divide %1, %arg1 : tensor<2x2xf32>
  %3 = stablehlo.maximum %2, %arg1 : tensor<2x2xf32>
  %4 = stablehlo.minimum %3, %arg1 : tensor<2x2xf32>
  %5 = stablehlo.multiply %4, %arg1 : tensor<2x2xf32>
  %6 = stablehlo.power %5, %arg1 : tensor<2x2xf32>
  %7 = stablehlo.subtract %6, %arg1 : tensor<2x2xf32>
  return %7 : tensor<2x2xf32>
}
// CHECK-DIRECT: stablehlo.add
// CHECK-DIRECT: stablehlo.atan2
// CHECK-DIRECT: stablehlo.divide
// CHECK-DIRECT: stablehlo.maximum
// CHECK-DIRECT: stablehlo.minimum
// CHECK-DIRECT: stablehlo.multiply
// CHECK-DIRECT: stablehlo.power
// CHECK-DIRECT: stablehlo.subtract

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[128,32]{1,0})->f32[128,128]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[128,32] parameter(0)
// CHECK-NEXT:  ROOT %[[all_gather_2:[^ ]+]] = f32[128,128] all-gather(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}}: replica_groups={{0,2,4,6},{1,3,5,7}},
// CHECK-SAME: dimensions={1},
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
%0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>}> {shard_count = 4 : i64} : (tensor<128x32xf32>) -> tensor<128x128xf32>
return %0 : tensor<128x128xf32>
}
// CHECK-DIRECT: stablehlo.all_gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[128,32]{1,0})->f32[128,128]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[128,32] parameter(0)
// CHECK-NEXT:  ROOT %[[all_gather_2:[^ ]+]] = f32[128,128] all-gather(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}}: replica_groups={{0,2,4,6},{1,3,5,7}},
// CHECK-SAME: dimensions={1}, use_global_device_ids=true,
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> {shard_count = 4 : i64} : (tensor<128x32xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
// CHECK-DIRECT: stablehlo.all_gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8,2]{1,0}, f32[8,4]{1,0})->(f32[8,8]{1,0}, f32[8,16]{1,0})}

// CHECK:       ENTRY %[[$main_10:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[8,4] parameter(1)
// CHECK-NEXT:  %[[tuple_3:[^ ]+]] = (f32[8,2], f32[8,4]) tuple(%[[Arg_0_1]], %[[Arg_1_2]]),
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = f32[8,2] get-tuple-element(%[[tuple_3]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = f32[8,4] get-tuple-element(%[[tuple_3]]), index=1,
// CHECK-NEXT:  %[[all_gather_6:[^ ]+]] = (f32[8,8], f32[8,16]) all-gather(%[[get_tuple_element_4]], %[[get_tuple_element_5]]), channel_id=1,
// CHECK-SAME{{LITERAL}}: replica_groups={{0,2,4,6},{1,3,5,7}},
// CHECK-SAME:  dimensions={1}, use_global_device_ids=true,
// CHECK-NEXT:  %[[get_tuple_element_7:[^ ]+]] = f32[8,8] get-tuple-element(%[[all_gather_6]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_8:[^ ]+]] = f32[8,16] get-tuple-element(%[[all_gather_6]]), index=1,
// CHECK-NEXT:  ROOT %[[tuple_9:[^ ]+]] = (f32[8,8], f32[8,16]) tuple(%[[get_tuple_element_7]], %[[get_tuple_element_8]]),
func.func @main(%arg0: tensor<8x2xf32>, %arg1: tensor<8x4xf32>) -> tuple<tensor<8x8xf32>, tensor<8x16xf32>> {
  %0:2 = "stablehlo.all_gather"(%arg0, %arg1) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<8x2xf32>, tensor<8x4xf32>) -> (tensor<8x8xf32>, tensor<8x16xf32>)
  %1 = stablehlo.tuple %0#0, %0#1 {xla_shape = "(f32[8,8]{0,1}, f32[8,16]{0,1})"} : tuple<tensor<8x8xf32>, tensor<8x16xf32>>
  return %1 : tuple<tensor<8x8xf32>, tensor<8x16xf32>>
}
// CHECK-DIRECT: stablehlo.all_gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[4]{0}, pred[4]{0})->pred[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = pred[4] parameter(1)
// CHECK-NEXT:  ROOT %[[and_3:[^ ]+]] = pred[4] and(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-DIRECT: stablehlo.and

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16,16,16,16]{3,2,1,0}, f32[16]{0}, f32[16]{0}, f32[16]{0}, f32[16]{0})->f32[16,16,16,16]{3,2,1,0}}

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16,16,16,16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[16] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[16] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = f32[16] parameter(3)
// CHECK-NEXT:  %[[Arg_4_5:[^ ]+]] = f32[16] parameter(4)
// CHECK-NEXT:  ROOT %[[batch_norm_inference_6:[^ ]+]] = f32[16,16,16,16] batch-norm-inference(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_3_4]], %[[Arg_4_5]]), epsilon=0.001, feature_index=0,
func.func @main(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
  epsilon = 0.001 : f32,
  feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}
// CHECK-DIRECT: stablehlo.batch_norm_inference

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[3,3]{1,0})->f32[3,3]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[3,3] parameter(0)
// CHECK-NEXT:  ROOT %[[cholesky_2:[^ ]+]] = f32[3,3] cholesky(%[[Arg_0_1]]), lower=true,
func.func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = stablehlo.cholesky %arg0 {lower = true} : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
// CHECK-DIRECT: stablehlo.cholesky

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16,8]{1,0})->f32[16,8]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16,8] parameter(0)
// CHECK-NEXT:  ROOT %[[collective_permute_2:[^ ]+]] = f32[16,8] collective-permute(%[[Arg_0_1]]),
// CHECK-SAME{{LITERAL}}: source_target_pairs={{0,1},{1,2},{2,3}},
func.func @main(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = "stablehlo.collective_permute"(%arg0) {
  source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}
// CHECK-DIRECT: stablehlo.collective_permute

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8]{0}, f32[8]{0})->f32[16]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[8] parameter(1)
// CHECK-NEXT:  ROOT %[[concatenate_3:[^ ]+]] = f32[16] concatenate(%[[Arg_0_1]], %[[Arg_1_2]]), dimensions={0},

func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<16xf32> {
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {
  dimension = 0 : i64
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.concatenate

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[])->f32[]}

// CHECK:       %[[$sum_2:[^ ]+]]
// CHECK-NEXT:  %[[x_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[y_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_5:[^ ]+]] = f32[] add(%[[x_3]], %[[y_4]])

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  ROOT %[[all_reduce_6:[^ ]+]] = f32[] all-reduce(%[[Arg_0_1]]),
// CHECK-SAME{{LITERAL}}: replica_groups={{0},{1}},
// CHECK-SAME: to_apply=%[[$sum_2]],
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.cross-replica-sum"(%arg0) {
  replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.cross-replica-sum

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0}, f32[4]{0}, s64[])->f32[16]{0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[4] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s64[] parameter(2)
// CHECK-NEXT:  ROOT %[[dynamic_update_slice_4:[^ ]+]] = f32[16] dynamic-update-slice(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]),
func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<4xf32>, %arg2: tensor<i64>) -> tensor<16xf32> {
  %0 = "stablehlo.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.dynamic_update_slice

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8,16]{1,0}, f32[16,8]{1,0})->f32[8,8]{1,0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8,16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[16,8] parameter(1)
// CHECK-NEXT:  %[[dot_3:[^ ]+]] = f32[8,8] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={grad_x="false",grad_y="false"},
// CHECK-NEXT:  ROOT %[[transpose_4:[^ ]+]] = f32[8,8] transpose(%[[dot_3]]), dimensions={0,1},
func.func @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {
  einsum_config = "ab,bc->ac"
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-DIRECT: stablehlo.einsum

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(c64[16]{0})->c64[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = c64[16] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = c64[16] fft(%[[Arg_0_1]]), fft_type=FFT, fft_length={16},
func.func @main(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type FFT>,
  fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(c64[16]{0})->c64[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = c64[16] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = c64[16] fft(%[[Arg_0_1]]), fft_type=IFFT, fft_length={16},
func.func @main(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type IFFT>,
  fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0})->c64[9]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = c64[9] fft(%[[Arg_0_1]]), fft_type=RFFT, fft_length={16},
func.func @main(%arg0: tensor<16xf32>) -> tensor<9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type RFFT>,
  fft_length = array<i64: 16>
  } : (tensor<16xf32>) -> tensor<9xcomplex<f32>>
  func.return %0 : tensor<9xcomplex<f32>>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(c64[9]{0})->f32[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = c64[9] parameter(0)
// CHECK-NEXT:  ROOT %[[fft_2:[^ ]+]] = f32[16] fft(%[[Arg_0_1]]), fft_type=IRFFT, fft_length={16},
func.func @main(%arg0: tensor<9xcomplex<f32>>) -> tensor<16xf32> {
  %0 = "stablehlo.fft"(%arg0) {
  fft_type = #stablehlo<fft_type IRFFT>,
  fft_length = array<i64: 16>
  } : (tensor<9xcomplex<f32>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.fft

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[?]{0})->s32[]}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[?] parameter(0)
// CHECK-NEXT:  ROOT %[[get_dimension_size_2:[^ ]+]] = s32[] get-dimension-size(%[[Arg_0_1]]), dimensions={0},

func.func @main(%arg0: tensor<?xf32>) -> tensor<i32> {
  %0 = "stablehlo.get_dimension_size"(%arg0) {
  dimension = 0 : i64
  } : (tensor<?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-DIRECT: stablehlo.get_dimension_size

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[4]{0}, pred[4]{0})->pred[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = pred[4] parameter(1)
// CHECK-NEXT:  ROOT %[[or_3:[^ ]+]] = pred[4] or(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-DIRECT: stablehlo.or

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[4]{0}, s32[4]{0})->s32[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[4] parameter(1)
// CHECK-NEXT:  ROOT %[[or_3:[^ ]+]] = s32[4] or(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}
// CHECK-DIRECT: stablehlo.or

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={()->u32[]}

// CHECK:       ENTRY %[[$main_2:[^ ]+]]
// CHECK-NEXT:  ROOT %[[replica_id_1:[^ ]+]] = u32[] replica-id(),
func.func @main() -> tensor<ui32> {
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-DIRECT: stablehlo.replica_id

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  ROOT %[[reverse_2:[^ ]+]] = f32[16] reverse(%[[Arg_0_1]]), dimensions={0},
func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = "stablehlo.reverse"(%arg0) {
  dimensions = array<i64: 0>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.reverse

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[], s32[2,3]{1,0}, s32[2,3]{1,0})->s32[2,3]{1,0}}

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[] parameter(0)
// CHECK-NEXT:  %[[broadcast_4:[^ ]+]] = pred[2,3] broadcast(%[[Arg_0_1]]), dimensions={},
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[2,3] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s32[2,3] parameter(2)
// CHECK-NEXT:  ROOT %[[select_5:[^ ]+]] = s32[2,3] select(%[[broadcast_4]], %[[Arg_1_2]], %[[Arg_2_3]]),
func.func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}
// CHECK-DIRECT: stablehlo.select

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[5,1,5]{2,1,0}, s32[2]{0})->f32[2,1,5]{2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[5,1,5] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[2] parameter(1)
// CHECK-NEXT:  ROOT %[[gather_3:[^ ]+]] = f32[2,1,5] gather(%[[Arg_0_1]], %[[Arg_1_2]]), offset_dims={1,2}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,1,5},
func.func @main(%arg0: tensor<5x1x5xf32>, %arg1: tensor<2xi32>) ->  tensor<2x1x5xf32> {
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
  dim = 0 : i64,
  batch_dims = 0 : i64
  } : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  func.return %0 : tensor<2x1x5xf32>
}
// CHECK-DIRECT: stablehlo.torch_index_select

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[1,2,3,4]{3,2,1,0})->s32[2,1,4,3]{2,3,0,1}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[1,2,3,4] parameter(0)
// CHECK-NEXT:  ROOT %[[transpose_2:[^ ]+]] = s32[2,1,4,3] transpose(%[[Arg_0_1]]), dimensions={1,0,3,2},
func.func @main(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2] : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0 : tensor<2x1x4x3xi32>
}
// CHECK-DIRECT: stablehlo.transpose

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,4]{1,0}, f32[3,4]{1,0})->f32[3,4]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[3,4] parameter(1)
// CHECK-NEXT:  ROOT %[[triangular_solve_3:[^ ]+]] = f32[3,4] triangular-solve(%[[Arg_0_1]], %[[Arg_1_2]]), lower=true, transpose_a=NO_TRANSPOSE,
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-DIRECT: stablehlo.triangular_solve

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,4]{1,0}, f32[3,4]{1,0})->f32[3,4]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[3,4] parameter(1)
// CHECK-NEXT:  ROOT %[[triangular_solve_3:[^ ]+]] = f32[3,4] triangular-solve(%[[Arg_0_1]], %[[Arg_1_2]]), lower=true, transpose_a=NO_TRANSPOSE,
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{left_side = false, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
// CHECK-DIRECT: stablehlo.triangular_solve

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[4]{0}, pred[4]{0})->pred[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = pred[4] parameter(1)
// CHECK-NEXT:  ROOT %[[xor_3:[^ ]+]] = pred[4] xor(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-DIRECT: stablehlo.xor

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[4]{0}, s32[4]{0})->s32[4]{0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[4] parameter(1)
// CHECK-NEXT:  ROOT %[[xor_3:[^ ]+]] = s32[4] xor(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}
// CHECK-DIRECT: stablehlo.xor

// -----

func.func @main() {
  // CHECK: token[]
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return
}
// CHECK-DIRECT: stablehlo.create_token

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[], s32[])->(f32[], s32[])}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  ROOT %[[tuple_3:[^ ]+]] = (f32[], s32[]) tuple(%[[Arg_0_1]], %[[Arg_1_2]]),
func.func @main(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tuple<tensor<f32>, tensor<i32>> {
  %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<f32>, tensor<i32>>
  return %0 : tuple<tensor<f32>, tensor<i32>>
  }
// CHECK-DIRECT: stablehlo.tuple

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[], token[])->token[]}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = token[] parameter(1)
// CHECK-NEXT:  %[[send_3:[^ ]+]] = (f32[], u32[], token[]) send(%[[Arg_0_1]], %[[Arg_1_2]]), is_host_transfer=true, metadata
// CHECK-NEXT:  ROOT %[[send_done_4:[^ ]+]] = token[] send-done(%[[send_3]]), is_host_transfer=true, metadata=

func.func @main(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.send"(%arg0, %arg1) {
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 2>,
  is_host_transfer = true
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.send

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(token[])->(f32[], token[])}

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = token[] parameter(0)
// CHECK-NEXT:  %[[recv_2:[^ ]+]] = (f32[], u32[], token[]) recv(%[[Arg_0_1]]), is_host_transfer=true,
// CHECK-NEXT:  %[[recv_done_3:[^ ]+]] = (f32[], token[]) recv-done(%[[recv_2]]), is_host_transfer=true,
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = f32[] get-tuple-element(%[[recv_done_3]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = token[] get-tuple-element(%[[recv_done_3]]), index=1,
// CHECK-NEXT:  ROOT %[[tuple_6:[^ ]+]] = (f32[], token[]) tuple(%[[get_tuple_element_4]], %[[get_tuple_element_5]])

func.func @main(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  %0:2 = "stablehlo.recv"(%arg0) {
  channel_handle = #stablehlo.channel_handle<handle = 0, type = 3>,
  is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}
// CHECK-DIRECT: stablehlo.recv

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(token[])->((s32[3,3]{1,0}, pred[]), token[])}

// CHECK:       ENTRY %[[$main_9:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = token[] parameter(0)
// CHECK-NEXT:  %[[infeed_2:[^ ]+]] = ((s32[3,3], pred[]), token[]) infeed(%[[Arg_0_1]]), infeed_config="foobar",
// CHECK-NEXT:  %[[get_tuple_element_3:[^ ]+]] = (s32[3,3], pred[]) get-tuple-element(%[[infeed_2]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = s32[3,3] get-tuple-element(%[[get_tuple_element_3]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = pred[] get-tuple-element(%[[get_tuple_element_3]]), index=1, metadata=
// CHECK-NEXT:  %[[tuple_7:[^ ]+]] = (s32[3,3], pred[]) tuple(%[[get_tuple_element_4]], %[[get_tuple_element_5]]), metadata=
// CHECK-NEXT:  %[[get_tuple_element_6:[^ ]+]] = token[] get-tuple-element(%[[infeed_2]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_8:[^ ]+]] = ((s32[3,3], pred[]), token[]) tuple(%[[tuple_7]], %[[get_tuple_element_6]]), metadata=

func.func @main(%arg0: !stablehlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token> {
  %0:3 = "stablehlo.infeed"(%arg0) <{infeed_config = "foobar", layout = [[0, 1], [0]]}> : (!stablehlo.token) -> (tensor<3x3xi32>, tensor<i1>, !stablehlo.token)
  %1 = stablehlo.tuple %0#0, %0#1 : tuple<tensor<3x3xi32>, tensor<i1>>
  %2 = stablehlo.tuple %1, %0#2 : tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
  return %2 : tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !stablehlo.token>
}
// CHECK-DIRECT: stablehlo.infeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(token[])->s32[3,3]{0,1}}

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = token[] parameter(0)
// CHECK-NEXT:  %[[infeed_2:[^ ]+]] = ((s32[3,3]), token[]) infeed(%[[Arg_0_1]]), infeed_config="foobar", metadata=
// CHECK-NEXT:  %[[get_tuple_element_3:[^ ]+]] = (s32[3,3]) get-tuple-element(%[[infeed_2]]), index=0, metadata=
// CHECK-NEXT:  ROOT %[[get_tuple_element_4:[^ ]+]] = s32[3,3] get-tuple-element(%[[get_tuple_element_3]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = token[] get-tuple-element(%[[infeed_2]]), index=1, metadata=
func.func @main(%arg0: !stablehlo.token) -> tensor<3x3xi32> {
  %0:2 = "stablehlo.infeed"(%arg0) <{infeed_config = "foobar", layout = [[0, 1]]}> : (!stablehlo.token) -> (tensor<3x3xi32>, !stablehlo.token)
  return %0#0 : tensor<3x3xi32>
}
// CHECK-DIRECT: stablehlo.infeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(token[])->token[]}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = token[] parameter(0)
// CHECK-NEXT:  %[[infeed_2:[^ ]+]] = ((), token[]) infeed(%[[Arg_0_1]]), infeed_config="foobar", metadata=
// CHECK-NEXT:  ROOT %[[get_tuple_element_3:[^ ]+]] = token[] get-tuple-element(%[[infeed_2]]), index=1, metadata=

func.func @main(%arg0: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.infeed"(%arg0) <{infeed_config = "foobar", layout = []}> : (!stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.infeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[], token[])->token[]}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[tuple_3:[^ ]+]] = (f32[]) tuple(%[[Arg_0_1]]), metadata=
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = token[] parameter(1)
// CHECK-NEXT:  ROOT %[[outfeed_4:[^ ]+]] = token[] outfeed(%[[tuple_3]], %[[Arg_1_2]]), outfeed_shape=(f32[]), metadata=

func.func @main(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%arg0, %arg1) {
  outfeed_config = ""
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.outfeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3]{0}, token[])->token[]}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3] parameter(0)
// CHECK-NEXT:  %[[tuple_3:[^ ]+]] = (s32[3]) tuple(%[[Arg_0_1]]), metadata=
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = token[] parameter(1)
// CHECK-NEXT:  ROOT %[[outfeed_4:[^ ]+]] = token[] outfeed(%[[tuple_3]], %[[Arg_1_2]]), outfeed_shape=(s32[3]{0}), outfeed_config="foobar", metadata=

func.func @main(%arg0: tensor<3xi32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%arg0, %arg1) <{outfeed_config = "foobar"}> : (tensor<3xi32>, !stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.outfeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3,2]{1,0}, token[])->token[]}

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3,2] parameter(0)
// CHECK-NEXT:  %[[custom_call_3:[^ ]+]] = s32[3,2] custom-call(%[[Arg_0_1]]), custom_call_target="Sharding", sharding={devices=[1,2]0,1}, metadata=
// CHECK-NEXT:  %[[custom_call_4:[^ ]+]] = s32[6,2] custom-call(%[[custom_call_3]]), custom_call_target="SPMDShardToFullShape", sharding={devices=[1,2]0,1}, metadata=
// CHECK-NEXT:  %[[tuple_5:[^ ]+]] = (s32[6,2]) tuple(%[[custom_call_4]]), metadata=
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = token[] parameter(1)
// CHECK-NEXT:  ROOT %[[outfeed_6:[^ ]+]] = token[] outfeed(%[[tuple_5]], %[[Arg_1_2]]), outfeed_shape=(s32[6,2]{1,0}), outfeed_config="foobar",
// CHECK-SAME{{LITERAL}} : sharding={{devices=[2,1]0,1}, {maximal device=0}},
func.func @main(%arg0: tensor<3x2xi32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<3x2xi32>) -> tensor<3x2xi32>
  %1 = stablehlo.custom_call @SPMDShardToFullShape(%0) {backend_config = "", mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<3x2xi32>) -> tensor<6x2xi32>
  %2 = "stablehlo.outfeed"(%1, %arg1) <{outfeed_config = "foobar"}> {mhlo.sharding = "\08\02*\0A\08\03\1A\02\02\01\22\02\00\01*\08\08\01\1A\01\01\22\01\00"} : (tensor<6x2xi32>, !stablehlo.token) -> !stablehlo.token
  return %2 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.outfeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[3]{0}, s32[3]{0}, token[])->token[]}

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[3] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[3] parameter(1)
// CHECK-NEXT:  %[[tuple_4:[^ ]+]] = (s32[3], s32[3]) tuple(%[[Arg_0_1]], %[[Arg_1_2]]), metadata=
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = token[] parameter(2)
// CHECK-NEXT:  ROOT %[[outfeed_5:[^ ]+]] = token[] outfeed(%[[tuple_4]], %[[Arg_2_3]]), outfeed_shape=(s32[3]{0}, s32[3]{0}), outfeed_config="foobar", metadata=

func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>, %arg2: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%arg0, %arg1, %arg2) <{outfeed_config = "foobar"}> : (tensor<3xi32>, tensor<3xi32>, !stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.outfeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(token[])->token[]}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[tuple_2:[^ ]+]] = () tuple(), metadata=
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = token[] parameter(0)
// CHECK-NEXT:  ROOT %[[outfeed_3:[^ ]+]] = token[] outfeed(%[[tuple_2]], %[[Arg_0_1]]), outfeed_shape=(), outfeed_config="foobar", metadata=

func.func @main(%arg0: !stablehlo.token) -> !stablehlo.token {
  %0 = "stablehlo.outfeed"(%arg0) <{outfeed_config = "foobar"}> : (!stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}
// CHECK-DIRECT: stablehlo.outfeed

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={((f32[], f32[], f32[], f32[], f32[]))->f32[]}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = (f32[], f32[], f32[], f32[], f32[]) parameter(0)
// CHECK-NEXT:  ROOT %[[get_tuple_element_2:[^ ]+]] = f32[] get-tuple-element(%[[Arg_0_1]]), index=4, metadata=

func.func @main(%arg0: tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32> {
  %0 = "stablehlo.get_tuple_element"(%arg0) {
  index = 4 : i32
  } : (tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.get_tuple_element

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[])->f32[]}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  ROOT %[[opt_barrier_2:[^ ]+]] = f32[] opt-barrier(%[[Arg_0_1]]), metadata=

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.optimization_barrier

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(pred[], f32[], f32[])->f32[]}

// CHECK:       %[[$region_0_4:[^ ]+]]
// CHECK-NEXT:  ROOT %[[Arg__5:[^ ]+]] = f32[] parameter(0)

// CHECK:       %[[$region_1_6:[^ ]+]]
// CHECK-NEXT:  ROOT %[[Arg__7:[^ ]+]] = f32[] parameter(0)

// CHECK:       ENTRY %[[$main_9:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = pred[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  ROOT %[[conditional_8:[^ ]+]] = f32[] conditional(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), true_computation=%[[$region_0_4]], false_computation=%[[$region_1_6]], metadata=

func.func @main(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.if"(%arg0) ({
  "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }, {
  "stablehlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.if

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[], f32[])->f32[]}

// CHECK:       %[[$region_0_3:[^ ]+]]
// CHECK-NEXT:  ROOT %[[Arg__4:[^ ]+]] = f32[] parameter(0)

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[conditional_5:[^ ]+]] = f32[] conditional(%[[Arg_0_1]], %[[Arg_1_2]]), branch_computations={%[[$region_0_3]]}, metadata=

func.func @main(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.case"(%arg0) ({
  "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.case

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10]{0})->f32[10]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[maximum_5:[^ ]+]] = f32[] maximum(%[[Arg_0_3]], %[[Arg_1_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10] parameter(0)
// CHECK-NEXT:  ROOT %[[all_reduce_6:[^ ]+]] = f32[10] all-reduce(%[[Arg_0_1]]), channel_id=5,
// CHECK-SAME{{LITERAL}}:  replica_groups={{0,2,4,6},{1,3,5,7}}, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
  }
}
// CHECK-DIRECT: stablehlo.all_reduce

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10]{0})->f32[10]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[maximum_5:[^ ]+]] = f32[] maximum(%[[Arg_0_3]], %[[Arg_1_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10] parameter(0)
// CHECK-NEXT:  ROOT %[[all_reduce_6:[^ ]+]] = f32[10] all-reduce(%[[Arg_0_1]]), channel_id=5,
// CHECK-SAME{{LITERAL}}:  replica_groups={{0,2,4},{1,3,5,6}}, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 2, 4, -1], [1, 3, 5, 6]]> : tensor<2x4xi64>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
  }
}
// CHECK-DIRECT: stablehlo.all_reduce

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10]{0})->f32[10]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[maximum_5:[^ ]+]] = f32[] maximum(%[[Arg_0_3]], %[[Arg_1_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10] parameter(0)
// CHECK-NEXT:  ROOT %[[all_reduce_6:[^ ]+]] = f32[10] all-reduce(%[[Arg_0_1]]), channel_id=5,
// CHECK-SAME{{LITERAL}}:  replica_groups={{0,2,4,6},{1,3,5,7}}, use_global_device_ids=true, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "stablehlo.all_reduce"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
  }
}
// CHECK-DIRECT: stablehlo.all_reduce

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8]{0}, f32[])->(f32[8]{0}, f32[])}

// CHECK:       %[[$region_0_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_7:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_8:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_9:[^ ]+]] = f32[] add(%[[Arg_0_7]], %[[Arg_1_8]]), metadata=

// CHECK:       ENTRY %[[$main_14:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  %[[tuple_3:[^ ]+]] = (f32[8], f32[]) tuple(%[[Arg_0_1]], %[[Arg_1_2]]), metadata=
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = f32[8] get-tuple-element(%[[tuple_3]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = f32[] get-tuple-element(%[[tuple_3]]), index=1, metadata=
// CHECK-NEXT:  %[[all_reduce_10:[^ ]+]] = (f32[8], f32[]) all-reduce(%[[get_tuple_element_4]], %[[get_tuple_element_5]]), replica_groups={}, to_apply=%[[$region_0_6]], metadata=
// CHECK-NEXT:  %[[get_tuple_element_11:[^ ]+]] = f32[8] get-tuple-element(%[[all_reduce_10]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_12:[^ ]+]] = f32[] get-tuple-element(%[[all_reduce_10]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_13:[^ ]+]] = (f32[8], f32[]) tuple(%[[get_tuple_element_11]], %[[get_tuple_element_12]]), metadata=

module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tuple<tensor<8xf32>, tensor<f32>> {
  %0:2 = "stablehlo.all_reduce"(%arg0, %arg1) <{replica_groups = dense<> : tensor<0x0xi64>}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
  stablehlo.return %2 : tensor<f32>
  }) : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  %1 = stablehlo.tuple %0#0, %0#1 {xla_shape = "(f32[8]{0}, f32[])"} : tuple<tensor<8xf32>, tensor<f32>>
  return %1 : tuple<tensor<8xf32>, tensor<f32>>
  }
}
// CHECK-DIRECT: stablehlo.all_reduce

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[1,10]{1,0}, s32[1,10]{1,0}, f32[], s32[])->(f32[1]{0}, s32[1]{0})}

// CHECK:       %[[$region_0_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_6:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_2_8:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  %[[maximum_10:[^ ]+]] = f32[] maximum(%[[Arg_0_6]], %[[Arg_2_8]]),
// CHECK-NEXT:  %[[Arg_1_7:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  %[[Arg_3_9:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[maximum_11:[^ ]+]] = s32[] maximum(%[[Arg_1_7]], %[[Arg_3_9]]),
// CHECK-NEXT:  ROOT %[[tuple_12:[^ ]+]] = (f32[], s32[]) tuple(%[[maximum_10]], %[[maximum_11]])

// CHECK:  ENTRY %[[$main_17:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[1,10] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[1,10] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[reduce_13:[^ ]+]] = (f32[1], s32[1]) reduce(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_3_4]]), dimensions={1}, to_apply=%[[$region_0_5]],
// CHECK-NEXT:  %[[get_tuple_element_14:[^ ]+]] = f32[1] get-tuple-element(%[[reduce_13]]), index=0,
// CHECK-NEXT:  %[[get_tuple_element_15:[^ ]+]] = s32[1] get-tuple-element(%[[reduce_13]]), index=1,
// CHECK-NEXT:  ROOT %[[tuple_16:[^ ]+]] = (f32[1], s32[1]) tuple(%[[get_tuple_element_14]], %[[get_tuple_element_15]])
func.func @main(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0:2 = stablehlo.reduce(%arg0 init: %arg2), (%arg1 init: %arg3) across dimensions = [1] : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
    reducer(%arg4: tensor<f32>, %arg6: tensor<f32>) (%arg5: tensor<i32>, %arg7: tensor<i32>)  {
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<1xf32>, tensor<1xi32>
}
// CHECK-DIRECT: stablehlo.reduce

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4]{0}, f32[4]{0})->f32[4]{0}}

// CHECK:       %[[$region_0_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_4:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_5:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_6:[^ ]+]] = f32[] add(%[[Arg_0_4]], %[[Arg_1_5]]), metadata=

// CHECK:       ENTRY %[[$main_8:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[4] parameter(1)
// CHECK-NEXT:  ROOT %[[map_7:[^ ]+]] = f32[4] map(%[[Arg_0_1]], %[[Arg_1_2]]), dimensions={0}, to_apply=%[[$region_0_3]], metadata=

module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "stablehlo.map"(%arg0, %arg1) <{dimensions = array<i64: 0>}> ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
  %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
  }
}
// CHECK-DIRECT: stablehlo.map

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[2,2]{1,0})->s32[2,2]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[2,2] parameter(0)
// CHECK-NEXT:  ROOT %[[all_to_all_2:[^ ]+]] = s32[2,2] all-to-all(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}}:  replica_groups={{1,2},{0,3}}, dimensions={1}, metadata=

func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "stablehlo.all_to_all"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[1, 2], [0, 3]]> : tensor<2x2xi64>, split_count = 2 : i64, split_dimension = 1 : i64}> : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
// CHECK-DIRECT: stablehlo.all_to_all

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2,2,2,2]{3,2,1,0})->(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})}

// CHECK:       ENTRY %[[$main_11:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,2,2,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[2] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[2] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = f32[2] parameter(3)
// CHECK-NEXT:  %[[Arg_4_5:[^ ]+]] = f32[2,2,2,2] parameter(4)
// CHECK-NEXT:  %[[batch_norm_grad_6:[^ ]+]] = (f32[2,2,2,2], f32[2], f32[2]) batch-norm-grad(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_3_4]], %[[Arg_4_5]]), epsilon=0.001, feature_index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_7:[^ ]+]] = f32[2,2,2,2] get-tuple-element(%[[batch_norm_grad_6]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_8:[^ ]+]] = f32[2] get-tuple-element(%[[batch_norm_grad_6]]), index=1, metadata=
// CHECK-NEXT:  %[[get_tuple_element_9:[^ ]+]] = f32[2] get-tuple-element(%[[batch_norm_grad_6]]), index=2, metadata=
// CHECK-NEXT:  ROOT %[[tuple_10:[^ ]+]] = (f32[2,2,2,2], f32[2], f32[2]) tuple(%[[get_tuple_element_7]], %[[get_tuple_element_8]], %[[get_tuple_element_9]]), metadata=

func.func @main(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2x2x2x2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 0 : i64}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %0 = stablehlo.tuple %grad_operand, %grad_scale, %grad_offset : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  return %0 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}
// CHECK-DIRECT: stablehlo.batch_norm_grad

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})->(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})}

// CHECK:       ENTRY %[[$main_9:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,2,2,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[2] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[2] parameter(2)
// CHECK-NEXT:  %[[batch_norm_training_4:[^ ]+]] = (f32[2,2,2,2], f32[2], f32[2]) batch-norm-training(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), epsilon=0.001, feature_index=3, metadata=
// CHECK-NEXT:  %[[get_tuple_element_5:[^ ]+]] = f32[2,2,2,2] get-tuple-element(%[[batch_norm_training_4]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_6:[^ ]+]] = f32[2] get-tuple-element(%[[batch_norm_training_4]]), index=1, metadata=
// CHECK-NEXT:  %[[get_tuple_element_7:[^ ]+]] = f32[2] get-tuple-element(%[[batch_norm_training_4]]), index=2, metadata=
// CHECK-NEXT:  ROOT %[[tuple_8:[^ ]+]] = (f32[2,2,2,2], f32[2], f32[2]) tuple(%[[get_tuple_element_5]], %[[get_tuple_element_6]], %[[get_tuple_element_7]]), metadata=

func.func @main(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %0 = stablehlo.tuple %output, %batch_mean, %batch_var : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  return %0 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}
// CHECK-DIRECT: stablehlo.batch_norm_training

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s32[2]{0})->f32[2]{0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s32[2] parameter(0)
// CHECK-NEXT:  ROOT %[[bitcast_convert_2:[^ ]+]] = f32[2] bitcast-convert(%[[Arg_0_1]]), metadata=

func.func @main(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = stablehlo.bitcast_convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-DIRECT: stablehlo.bitcast_convert

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[], f32[], f32[])->f32[]}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  ROOT %[[clamp_4:[^ ]+]] = f32[] clamp(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), metadata=

func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-DIRECT: stablehlo.clamp

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[128,32]{1,0})->f32[128,32]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[128,32] parameter(0)
// CHECK-NEXT:  ROOT %[[collective_broadcast_2:[^ ]+]] = f32[128,32] collective-broadcast(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}} : replica_groups={{0,1},{2,3}}, metadata=

func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}
// CHECK-DIRECT: stablehlo.collective_broadcast

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[16]{0})->f32[16]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[compare_5:[^ ]+]] = pred[] compare(%[[Arg_0_3]], %[[Arg_1_4]]), direction=GT, metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[16] parameter(0)
// CHECK-NEXT:  ROOT %[[sort_6:[^ ]+]] = f32[16] sort(%[[Arg_0_1]]), dimensions={0}, is_stable=true, to_apply=%[[$region_0_2]], metadata=

func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  %0 = "stablehlo.sort"(%arg0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }) {
  dimension = 0 : i64,
  is_stable = true
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-DIRECT: stablehlo.sort
// CHECK-DIRECT: stablehlo.compare

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s64[])->s64[]}

// CHECK:       %[[$add_n_impl_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = s64[] parameter(0)
// CHECK-NEXT:  %[[constant_4:[^ ]+]] = s64[] constant(2)
// CHECK-NEXT:  ROOT %[[add_5:[^ ]+]] = s64[] add(%[[Arg_0_3]], %[[constant_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s64[] parameter(0)
// CHECK-NEXT:  ROOT %[[call_6:[^ ]+]] = s64[] call(%[[Arg_0_1]]), to_apply=%[[$add_n_impl_2]], is_composite=true, frontend_attributes={composite.attributes={n = 2 : i64},composite.name="stablehlo.add_n",composite.version="0"}

func.func @main(%arg0 : tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.composite "stablehlo.add_n" %arg0 {
  composite_attributes = { n = 2 : i64 },
  decomposition = @add_n.impl
  } : (tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

func.func @add_n.impl(%arg0: tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.add %arg0, %0 : tensor<i64>
  func.return %1 : tensor<i64>
}
// CHECK-DIRECT: stablehlo.composite

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,3]{1,0})->f32[2,3]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,3] parameter(0)
// CHECK-NEXT:  ROOT %[[custom_call_2:[^ ]+]] = f32[2,3] custom-call(%[[Arg_0_1]]), custom_call_target="SetBound", literal=s32[] 1, metadata=

module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = stablehlo.custom_call @SetBound(%arg0) {mhlo.literal = dense<1> : tensor<i32>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[6]{0}, f32[6]{0}, s32[3]{0}, s32[3]{0}, s32[3]{0}, /*index=5*/s32[3]{0})->f32[6]{0}}

// CHECK:       ENTRY %[[$main_8:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[6] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[6] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s32[3] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = s32[3] parameter(3)
// CHECK-NEXT:  %[[Arg_4_5:[^ ]+]] = s32[3] parameter(4)
// CHECK-NEXT:  %[[Arg_5_6:[^ ]+]] = s32[3] parameter(5)
// CHECK-NEXT:  ROOT %[[ragged_all_to_all_7:[^ ]+]] = f32[6] ragged-all-to-all(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_3_4]], %[[Arg_4_5]],

module {
  func.func @main(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>, %arg2: tensor<3xi32>, %arg3: tensor<3xi32>, %arg4: tensor<3xi32>, %arg5: tensor<3xi32>) -> tensor<6xf32> {
  %0 = stablehlo.custom_call @ragged_all_to_all(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {api_version = 4 : i32, backend_config = {channel_id = 1 : i64, replica_groups = dense<[[0, 1, 2]]> : tensor<1x3xi64>}} : (tensor<6xf32>, tensor<6xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(bf16[16,256]{1,0}, s32[], s32[16,256]{1,0}, bf16[])->(bf16[16,4]{1,0}, s32[16,4]{1,0})}

// CHECK:       %[[$top_k_gt_comparator_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_2_8:[^ ]+]] = s32[] parameter(2)
// CHECK-NEXT:  %[[Arg_3_9:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[Arg_0_6:[^ ]+]] = bf16[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_7:[^ ]+]] = bf16[] parameter(1)
// CHECK-NEXT:  ROOT %[[compare_10:[^ ]+]] = pred[] compare(%[[Arg_0_6]], %[[Arg_1_7]]), direction=GT, metadata=

// CHECK:       ENTRY %[[$main_20:[^ ]+]]
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = bf16[] parameter(3)
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = bf16[16,256] parameter(0)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s32[16,256] parameter(2)
// CHECK-NEXT:  %[[sort_11:[^ ]+]] = (bf16[16,256], s32[16,256]) sort(%[[Arg_0_1]], %[[Arg_2_3]]), dimensions={1}, to_apply=%[[$top_k_gt_comparator_5]], metadata=
// CHECK-NEXT:  %[[get_tuple_element_12:[^ ]+]] = bf16[16,256] get-tuple-element(%[[sort_11]]), index=0, metadata=
// CHECK-NEXT:  %[[slice_13:[^ ]+]] = bf16[16,4] slice(%[[get_tuple_element_12]]), slice={[0:16], [0:4]}, metadata=
// CHECK-NEXT:  %[[get_tuple_element_14:[^ ]+]] = s32[16,256] get-tuple-element(%[[sort_11]]), index=1, metadata=
// CHECK-NEXT:  %[[slice_15:[^ ]+]] = s32[16,4] slice(%[[get_tuple_element_14]]), slice={[0:16], [0:4]}, metadata=
// CHECK-NEXT:  %[[tuple_16:[^ ]+]] = (bf16[16,4], s32[16,4]) tuple(%[[slice_13]], %[[slice_15]]), metadata=
// CHECK-NEXT:  %[[get_tuple_element_17:[^ ]+]] = bf16[16,4] get-tuple-element(%[[tuple_16]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_18:[^ ]+]] = s32[16,4] get-tuple-element(%[[tuple_16]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_19:[^ ]+]] = (bf16[16,4], s32[16,4]) tuple(%[[get_tuple_element_17]], %[[get_tuple_element_18]])

module {
  func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
  }
  func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {api_version = 4 : i32, backend_config = {aggregate_to_topk = true, is_fallback = true, recall_target = 0.949217975 : f32, reduction_dim = 1 : i64, reduction_input_size_override = -1 : i64, top_k = 4 : i64}, called_computations = [@top_k_gt_comparator]} : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %0#0, %0#1 : tensor<16x4xbf16>, tensor<16x4xi32>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(bf16[16,256]{1,0}, s32[], s32[16,256]{1,0}, bf16[])->(bf16[16,4]{1,0}, s32[16,4]{1,0})}

// CHECK:       %[[$top_k_gt_comparator_5:[^ ]+]]
// CHECK-NEXT:  %[[Arg_2_8:[^ ]+]] = s32[] parameter(2)
// CHECK-NEXT:  %[[Arg_3_9:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[Arg_0_6:[^ ]+]] = bf16[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_7:[^ ]+]] = bf16[] parameter(1)
// CHECK-NEXT:  ROOT %[[compare_10:[^ ]+]] = pred[] compare(%[[Arg_0_6]], %[[Arg_1_7]]), direction=GT, metadata=

// CHECK:       %[[$top_k_gt_comparator_14:[^ ]+]]
// CHECK-NEXT:  %[[Arg_2_17:[^ ]+]] = s32[] parameter(2)
// CHECK-NEXT:  %[[Arg_3_18:[^ ]+]] = s32[] parameter(3)
// CHECK-NEXT:  %[[Arg_0_15:[^ ]+]] = bf16[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_16:[^ ]+]] = bf16[] parameter(1)
// CHECK-NEXT:  ROOT %[[compare_19:[^ ]+]] = pred[] compare(%[[Arg_0_15]], %[[Arg_1_16]]), direction=GT, metadata=

// CHECK:       ENTRY %[[$main_29:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = bf16[16,256] parameter(0)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = s32[16,256] parameter(2)
// CHECK-NEXT:  %[[Arg_3_4:[^ ]+]] = bf16[] parameter(3)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  %[[custom_call_11:[^ ]+]] = (bf16[16,128], s32[16,128]) custom-call(%[[Arg_0_1]], %[[Arg_2_3]], %[[Arg_3_4]], %[[Arg_1_2]]), custom_call_target="PartialReduce", called_computations={%[[$top_k_gt_comparator_5]]}, metadata=
// CHECK-NEXT:  %[[get_tuple_element_12:[^ ]+]] = bf16[16,128] get-tuple-element(%[[custom_call_11]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_13:[^ ]+]] = s32[16,128] get-tuple-element(%[[custom_call_11]]), index=1, metadata=
// CHECK-NEXT:  %[[sort_20:[^ ]+]] = (bf16[16,128], s32[16,128]) sort(%[[get_tuple_element_12]], %[[get_tuple_element_13]]), dimensions={1}, to_apply=%[[$top_k_gt_comparator_14]], metadata=
// CHECK-NEXT:  %[[get_tuple_element_21:[^ ]+]] = bf16[16,128] get-tuple-element(%[[sort_20]]), index=0, metadata=
// CHECK-NEXT:  %[[slice_22:[^ ]+]] = bf16[16,4] slice(%[[get_tuple_element_21]]), slice={[0:16], [0:4]}, metadata=
// CHECK-NEXT:  %[[get_tuple_element_23:[^ ]+]] = s32[16,128] get-tuple-element(%[[sort_20]]), index=1, metadata=
// CHECK-NEXT:  %[[slice_24:[^ ]+]] = s32[16,4] slice(%[[get_tuple_element_23]]), slice={[0:16], [0:4]}, metadata=
// CHECK-NEXT:  %[[tuple_25:[^ ]+]] = (bf16[16,4], s32[16,4]) tuple(%[[slice_22]], %[[slice_24]]), metadata=
// CHECK-NEXT:  %[[get_tuple_element_26:[^ ]+]] = bf16[16,4] get-tuple-element(%[[tuple_25]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_27:[^ ]+]] = s32[16,4] get-tuple-element(%[[tuple_25]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_28:[^ ]+]] = (bf16[16,4], s32[16,4]) tuple(%[[get_tuple_element_26]], %[[get_tuple_element_27]])

module {
  func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
  }
  func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {api_version = 4 : i32, backend_config = {aggregate_to_topk = true, is_fallback = false, recall_target = 0.949217975 : f32, reduction_dim = 1 : i64, reduction_input_size_override = -1 : i64, top_k = 4 : i64}, called_computations = [@top_k_gt_comparator]} : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %0#0, %0#1 : tensor<16x4xbf16>, tensor<16x4xi32>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,3]{1,0})->(f32[2,3]{1,0})}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,3] parameter(0)
// CHECK-NEXT:  ROOT %[[custom_call_2:[^ ]+]] = (f32[2,3]) custom-call(%[[Arg_0_1]]), custom_call_target="foo", metadata=

module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tuple<tensor<2x3xf32>> {
  %0 = stablehlo.custom_call @foo(%arg0) : (tensor<2x3xf32>) -> tuple<tensor<2x3xf32>>
  return %0 : tuple<tensor<2x3xf32>>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,3]{1,0})->(f32[2,3]{1,0}, f16[4,5]{1,0})}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,3] parameter(0)
// CHECK-NEXT:  ROOT %[[custom_call_2:[^ ]+]] = (f32[2,3], f16[4,5]) custom-call(%[[Arg_0_1]]), custom_call_target="foo", metadata=

module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tuple<tensor<2x3xf32>, tensor<4x5xf16>> {
  %0 = stablehlo.custom_call @foo(%arg0) : (tensor<2x3xf32>) -> tuple<tensor<2x3xf32>, tensor<4x5xf16>>
  return %0 : tuple<tensor<2x3xf32>, tensor<4x5xf16>>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2,3]{1,0})->(f32[2,3]{1,0}, f16[4,5]{1,0})}

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2,3] parameter(0)
// CHECK-NEXT:  %[[custom_call_2:[^ ]+]] = (f32[2,3], f16[4,5]) custom-call(%[[Arg_0_1]]), custom_call_target="foo", metadata=
// CHECK-NEXT:  %[[get_tuple_element_3:[^ ]+]] = f32[2,3] get-tuple-element(%[[custom_call_2]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = f16[4,5] get-tuple-element(%[[custom_call_2]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_5:[^ ]+]] = (f32[2,3], f16[4,5]) tuple(%[[get_tuple_element_3]], %[[get_tuple_element_4]])

module {
  func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<4x5xf16>) {
  %0:2 = stablehlo.custom_call @foo(%arg0) : (tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<4x5xf16>)
  return %0#0, %0#1 : tensor<2x3xf32>, tensor<4x5xf16>
  }
}
// CHECK-DIRECT: stablehlo.custom_call

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s8[2,2,2]{2,1,0}, s8[2,2,3]{2,1,0})->s32[2,2,3]{2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s8[2,2,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s8[2,2,3] parameter(1)
// CHECK-NEXT:  ROOT %[[dot_3:[^ ]+]] = s32[2,2,3] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, metadata=

module {
  func.func @main(%arg0: tensor<2x2x2xi8>, %arg1: tensor<2x2x3xi8>) -> tensor<2x2x3xi32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [] : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  return %0 : tensor<2x2x3xi32>
  }
}
// CHECK-DIRECT: stablehlo.dot_general

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8,8,16]{2,1,0}, f32[8,16,8]{2,1,0})->f32[8,8,8]{2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8,8,16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[8,16,8] parameter(1)
// CHECK-NEXT:  ROOT %[[dot_3:[^ ]+]] = f32[8,8,8] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, metadata=

func.func @main(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
  dot_dimension_numbers = #stablehlo.dot<
  lhs_batching_dimensions = [0],
  lhs_contracting_dimensions = [2],
  rhs_batching_dimensions = [0],
  rhs_contracting_dimensions = [1]
  >,
  precision_config = []
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}
// CHECK-DIRECT: stablehlo.dot_general

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[8,8,16]{2,1,0}, f32[8,16,8]{2,1,0})->f32[8,8,8]{2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[8,8,16] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[8,16,8] parameter(1)
// CHECK-NEXT:  ROOT %[[dot_3:[^ ]+]] = f32[8,8,8] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}, algorithm=dot_tf32_tf32_f32, metadata=

func.func @main(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
  dot_dimension_numbers = #stablehlo.dot<
  lhs_batching_dimensions = [0],
  lhs_contracting_dimensions = [2],
  rhs_batching_dimensions = [0],
  rhs_contracting_dimensions = [1]
  >,
  algorithm = #stablehlo.dot_algorithm<
  lhs_precision_type = tf32,
  rhs_precision_type = tf32,
  accumulation_type = f32,
  lhs_component_count = 1,
  rhs_component_count = 1,
  num_primitive_operations = 1,
  allow_imprecise_accumulation = false
  >
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}
// CHECK-DIRECT: stablehlo.dot_general

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s8[3]{0}, s8[3]{0})->s64[]}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s8[3] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s8[3] parameter(1)
// CHECK-NEXT:  ROOT %[[dot_3:[^ ]+]] = s64[] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata=

module {
  func.func @main(%arg0: tensor<3xi8>, %arg1: tensor<3xi8>) -> tensor<i64> {
  %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<3xi8>, tensor<3xi8>) -> tensor<i64>
  return %0 : tensor<i64>
  }
}
// CHECK-DIRECT: stablehlo.dot

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(s4[3]{0}, s4[3]{0})->s8[]}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = s4[3] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s4[3] parameter(1)
// CHECK-NEXT:  ROOT %[[dot_3:[^ ]+]] = s8[] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata=

module {
  func.func @main(%arg0: tensor<3xi4>, %arg1: tensor<3xi4>) -> tensor<i8> {
  %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<3xi4>, tensor<3xi4>) -> tensor<i8>
  return %0 : tensor<i8>
  }
}
// CHECK-DIRECT: stablehlo.dot

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(u4[3]{0}, u4[3]{0})->u8[]}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = u4[3] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = u4[3] parameter(1)
// CHECK-NEXT:  ROOT %[[dot_3:[^ ]+]] = u8[] dot(%[[Arg_0_1]], %[[Arg_1_2]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata=

module {
  func.func @main(%arg0: tensor<3xui4>, %arg1: tensor<3xui4>) -> tensor<ui8> {
  %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<3xui4>, tensor<3xui4>) -> tensor<ui8>
  return %0 : tensor<ui8>
  }
}
// CHECK-DIRECT: stablehlo.dot

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[2]{0})->f32[1,2]{1,0}}

// CHECK:       ENTRY %[[$main_3:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[2] parameter(0)
// CHECK-NEXT:  ROOT %[[reshape_2:[^ ]+]] = f32[1,2] reshape(%[[Arg_0_1]]), metadata=

module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<1x2xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
  }
}
// CHECK-DIRECT: stablehlo.reshape

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[200,100,300]{2,1,0}, s32[10,2]{1,0})->f32[10,300]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[200,100,300] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[10,2] parameter(1)
// CHECK-NEXT:  ROOT %[[gather_3:[^ ]+]] = f32[10,300] gather(%[[Arg_0_1]], %[[Arg_1_2]]), offset_dims={1}, collapsed_slice_dims={0,1}, start_index_map={0,1}, index_vector_dim=1, slice_sizes={1,1,300}, indices_are_sorted=true, metadata=

module {
  func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<10x300xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 300>}> : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
  return %0 : tensor<10x300xf32>
  }
}
// CHECK-DIRECT: stablehlo.gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[200,100,300]{2,1,0}, s32[100,200,1]{2,1,0})->f32[100,200,300]{2,1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[200,100,300] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[100,200,1] parameter(1)
// CHECK-NEXT:  ROOT %[[gather_3:[^ ]+]] = f32[100,200,300] gather(%[[Arg_0_1]], %[[Arg_1_2]]), offset_dims={2}, collapsed_slice_dims={}, start_index_map={2}, operand_batching_dims={0,1}, start_indices_batching_dims={1,0}, index_vector_dim=2, slice_sizes={1,1,300}, indices_are_sorted=true, metadata=

module {
  func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<100x200x1xi32>) -> tensor<100x200x300xf32> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], operand_batching_dims = [0, 1], start_indices_batching_dims = [1, 0], start_index_map = [2], index_vector_dim = 2>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1, 300>}> : (tensor<200x100x300xf32>, tensor<100x200x1xi32>) -> tensor<100x200x300xf32>
  return %0 : tensor<100x200x300xf32>
  }
}
// CHECK-DIRECT: stablehlo.gather

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={()->f32[1,10]{1,0}}

// CHECK:       ENTRY %[[$main_2:[^ ]+]]
// CHECK-NEXT:  ROOT %[[iota_1:[^ ]+]] = f32[1,10] iota(), iota_dimension=1, metadata=

module {
  func.func @main() -> tensor<1x10xf32> {
  %0 = stablehlo.iota dim = 1 : tensor<1x10xf32>
  return %0 : tensor<1x10xf32>
  }
}
// CHECK-DIRECT: stablehlo.iota

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,6]{1,0}, f32[])->f32[13,19]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,6] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[pad_3:[^ ]+]] = f32[13,19] pad(%[[Arg_0_1]], %[[Arg_1_2]]), padding=2_4_1x3_5_1, metadata=

module {
  func.func @main(%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<13x19xf32> {
  %0 = stablehlo.pad %arg0, %arg1, low = [2, 3], high = [4, 5], interior = [1, 1] : (tensor<4x6xf32>, tensor<f32>) -> tensor<13x19xf32>
  return %0 : tensor<13x19xf32>
  }
}
// CHECK-DIRECT: stablehlo.pad

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={()->u32[]}

// CHECK:       ENTRY %[[$main_2:[^ ]+]]
// CHECK-NEXT:  ROOT %[[partition_id_1:[^ ]+]] = u32[] partition-id(), metadata=

module {
  func.func @main() -> tensor<ui32> {
  %0 = stablehlo.partition_id : tensor<ui32>
  return %0 : tensor<ui32>
  }
}
// CHECK-DIRECT: stablehlo.partition_id

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10]{0})->f32[5]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[maximum_5:[^ ]+]] = f32[] maximum(%[[Arg_0_3]], %[[Arg_1_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10] parameter(0)
// CHECK-NEXT:  ROOT %[[reduce_scatter_6:[^ ]+]] = f32[5] reduce-scatter(%[[Arg_0_1]]), channel_id=5,
// CHECK-SAME{{LITERAL}} : replica_groups={{0,2},{1,3}}, dimensions={0}, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, scatter_dimension = 0 : i64}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<10xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
  }
}
// CHECK-DIRECT: stablehlo.reduce_scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10]{0})->f32[5]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[maximum_5:[^ ]+]] = f32[] maximum(%[[Arg_0_3]], %[[Arg_1_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10] parameter(0)
// CHECK-NEXT:  ROOT %[[reduce_scatter_6:[^ ]+]] = f32[5] reduce-scatter(%[[Arg_0_1]]), channel_id=5,
// CHECK-SAME{{LITERAL}} : replica_groups={{0,2},{1,3}}, dimensions={0}, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, scatter_dimension = 0 : i64}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<10xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
  }
}
// CHECK-DIRECT: stablehlo.reduce_scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10]{0})->f32[5]{0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[maximum_5:[^ ]+]] = f32[] maximum(%[[Arg_0_3]], %[[Arg_1_4]]), metadata=

// CHECK:       ENTRY %[[$main_7:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10] parameter(0)
// CHECK-NEXT:  ROOT %[[reduce_scatter_6:[^ ]+]] = f32[5] reduce-scatter(%[[Arg_0_1]]), channel_id=5,
// CHECK-SAME{{LITERAL}} : replica_groups={{0,2},{1,3}}, use_global_device_ids=true, dimensions={0}, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 2>, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<10xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
  }
}
// CHECK-DIRECT: stablehlo.reduce_scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,16]{1,0})->f32[4,4]{1,0}}

// CHECK:       %[[$region_0_2:[^ ]+]]
// CHECK-NEXT:  ROOT %[[Arg_0_3:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_4:[^ ]+]] = f32[] parameter(1)

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK:  %[[Arg_0_1:[^ ]+]] = f32[4,16] parameter(0)
// CHECK-NEXT:  ROOT %[[reduce_scatter_5:[^ ]+]] = f32[4,4] reduce-scatter(%[[Arg_0_1]]), channel_id=1,
// CHECK-SAME{{LITERAL}} : replica_groups={{0,1,2,3}}, use_global_device_ids=true, dimensions={1}, to_apply=%[[$region_0_2]], metadata=

module {
  func.func @main(%arg0: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  stablehlo.return %arg1 : tensor<f32>
  }) : (tensor<4x16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  }
}
// CHECK-DIRECT: stablehlo.reduce_scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[], f32[])->f32[2,3,5]{2,1,0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[constant_3:[^ ]+]] = s64[3] constant({2, 3, 5})
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[rng_4:[^ ]+]] = f32[2,3,5] rng(%[[Arg_0_1]], %[[Arg_1_2]]), distribution=rng_normal, metadata=

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<2x3x5xf32> {
  %c = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = stablehlo.rng %arg0, %arg1, %c, distribution =  NORMAL : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
  }
}
// CHECK-DIRECT: stablehlo.rng

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={()->f32[2,3,5]{2,1,0}}

// CHECK:       ENTRY %[[$main_5:[^ ]+]]
// CHECK-NEXT:  %[[constant_3:[^ ]+]] = s64[3] constant({2, 3, 5})
// CHECK-NEXT:  %[[constant_1:[^ ]+]] = f32[] constant(0)
// CHECK-NEXT:  %[[constant_2:[^ ]+]] = f32[] constant(1)
// CHECK-NEXT:  ROOT %[[rng_4:[^ ]+]] = f32[2,3,5] rng(%[[constant_1]], %[[constant_2]]), distribution=rng_uniform, metadata=

module {
  func.func @main() -> tensor<2x3x5xf32> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %c = stablehlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = stablehlo.rng %cst, %cst_0, %c, distribution =  UNIFORM : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
  }
}
// CHECK-DIRECT: stablehlo.rng

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(u64[3]{0})->(u64[3]{0}, u32[2,2]{1,0})}

// CHECK:       ENTRY %[[$main_6:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = u64[3] parameter(0)
// CHECK-NEXT:  %[[rng_bit_generator_2:[^ ]+]] = (u64[3], u32[2,2]) rng-bit-generator(%[[Arg_0_1]]), algorithm=rng_philox, metadata=
// CHECK-NEXT:  %[[get_tuple_element_3:[^ ]+]] = u64[3] get-tuple-element(%[[rng_bit_generator_2]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_4:[^ ]+]] = u32[2,2] get-tuple-element(%[[rng_bit_generator_2]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_5:[^ ]+]] = (u64[3], u32[2,2]) tuple(%[[get_tuple_element_3]], %[[get_tuple_element_4]]), metadata=

module {
  func.func @main(%arg0: tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<2x2xui32>> {
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  PHILOX : (tensor<3xui64>) -> (tensor<3xui64>, tensor<2x2xui32>)
  %0 = stablehlo.tuple %output_state, %output : tuple<tensor<3xui64>, tensor<2x2xui32>>
  return %0 : tuple<tensor<3xui64>, tensor<2x2xui32>>
  }
}
// CHECK-DIRECT: stablehlo.rng_bit_generator

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[200,100,300]{2,1,0}, s32[10,2]{1,0}, f32[10,300]{1,0})->f32[200,100,300]{2,1,0}}

// CHECK:       %[[$region_0_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_5:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_6:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_7:[^ ]+]] = f32[] add(%[[Arg_0_5]], %[[Arg_1_6]]), metadata=

// CHECK:       ENTRY %[[$main_9:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[200,100,300] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[10,2] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[10,300] parameter(2)
// CHECK-NEXT:  ROOT %[[scatter_8:[^ ]+]] = f32[200,100,300] scatter(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), update_window_dims={1}, inserted_window_dims={0,1}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true, to_apply=%[[$region_0_4]], metadata=

module {
  func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  return %0 : tensor<200x100x300xf32>
  }
}
// CHECK-DIRECT: stablehlo.scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[200,100,300]{2,1,0}, s32[100,200,1]{2,1,0}, f32[100,200,300]{2,1,0})->f32[200,100,300]{2,1,0}}

// CHECK:       %[[$region_0_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_5:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_6:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_7:[^ ]+]] = f32[] add(%[[Arg_0_5]], %[[Arg_1_6]]), metadata=

// CHECK:       ENTRY %[[$main_9:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[200,100,300] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[100,200,1] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[100,200,300] parameter(2)
// CHECK-NEXT:  ROOT %[[scatter_8:[^ ]+]] = f32[200,100,300] scatter(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), update_window_dims={2}, inserted_window_dims={}, scatter_dims_to_operand_dims={2}, input_batching_dims={0,1}, scatter_indices_batching_dims={1,0}, index_vector_dim=2, indices_are_sorted=true, unique_indices=true, to_apply=%[[$region_0_4]], metadata=

module {
  func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<100x200x1xi32>, %arg2: tensor<100x200x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], input_batching_dims = [0, 1], scatter_indices_batching_dims = [1, 0], scatter_dims_to_operand_dims = [2], index_vector_dim = 2>, unique_indices = true}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
  %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
  stablehlo.return %1 : tensor<f32>
  }) : (tensor<200x100x300xf32>, tensor<100x200x1xi32>, tensor<100x200x300xf32>) -> tensor<200x100x300xf32>
  return %0 : tensor<200x100x300xf32>
  }
}
// CHECK-DIRECT: stablehlo.scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[200,100,300]{2,1,0}, s64[10,2]{1,0}, f32[10,300]{1,0})->(f32[200,100,300]{2,1,0}, f32[200,100,300]{2,1,0})}

// CHECK:       %[[$region_0_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_5:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_6:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  %[[add_9:[^ ]+]] = f32[] add(%[[Arg_0_5]], %[[Arg_1_6]]), metadata=
// CHECK-NEXT:  %[[Arg_2_7:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  %[[Arg_3_8:[^ ]+]] = f32[] parameter(3)
// CHECK-NEXT:  %[[add_10:[^ ]+]] = f32[] add(%[[Arg_2_7]], %[[Arg_3_8]]), metadata=
// CHECK-NEXT:  ROOT %[[tuple_11:[^ ]+]] = (f32[], f32[]) tuple(%[[add_9]], %[[add_10]])

// CHECK:       ENTRY %[[$main_16:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[200,100,300] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s64[10,2] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[10,300] parameter(2)
// CHECK-NEXT:  %[[scatter_12:[^ ]+]] = (f32[200,100,300], f32[200,100,300]) scatter(%[[Arg_0_1]], %[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]], %[[Arg_2_3]]), update_window_dims={1}, inserted_window_dims={0,1}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=%[[$region_0_4]], metadata=
// CHECK-NEXT:  %[[get_tuple_element_13:[^ ]+]] = f32[200,100,300] get-tuple-element(%[[scatter_12]]), index=0, metadata=
// CHECK-NEXT:  %[[get_tuple_element_14:[^ ]+]] = f32[200,100,300] get-tuple-element(%[[scatter_12]]), index=1, metadata=
// CHECK-NEXT:  ROOT %[[tuple_15:[^ ]+]] = (f32[200,100,300], f32[200,100,300]) tuple(%[[get_tuple_element_13]], %[[get_tuple_element_14]])

module {
  func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi64>, %arg2: tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
  %0:2 = "stablehlo.scatter"(%arg0, %arg0, %arg1, %arg2, %arg2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
  %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
  %2 = stablehlo.add %arg5, %arg6 : tensor<f32>
  stablehlo.return %1, %2 : tensor<f32>, tensor<f32>
  }) : (tensor<200x100x300xf32>, tensor<200x100x300xf32>, tensor<10x2xi64>, tensor<10x300xf32>, tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>)
  return %0#0, %0#1 : tensor<200x100x300xf32>, tensor<200x100x300xf32>
  }
}
// CHECK-DIRECT: stablehlo.scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[10,24,24,64]{3,2,1,0}, f32[10,12,12,64]{3,2,1,0}, f32[])->f32[10,24,24,64]{3,2,1,0}}

// CHECK:       %[[$region_0_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_5:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_6:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[compare_7:[^ ]+]] = pred[] compare(%[[Arg_0_5]], %[[Arg_1_6]]), direction=GE, type=TOTALORDER, metadata=

// CHECK:       %[[$region_1_8:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_9:[^ ]+]] = f32[] parameter(0)
// CHECK-NEXT:  %[[Arg_1_10:[^ ]+]] = f32[] parameter(1)
// CHECK-NEXT:  ROOT %[[add_11:[^ ]+]] = f32[] add(%[[Arg_0_9]], %[[Arg_1_10]]), metadata=

// CHECK:       ENTRY %[[$main_13:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[10,24,24,64] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = f32[10,12,12,64] parameter(1)
// CHECK-NEXT:  %[[Arg_2_3:[^ ]+]] = f32[] parameter(2)
// CHECK-NEXT:  ROOT %[[select_and_scatter_12:[^ ]+]] = f32[10,24,24,64] select-and-scatter(%[[Arg_0_1]], %[[Arg_1_2]], %[[Arg_2_3]]), window={size=1x2x2x1 stride=1x2x2x1}, select=%[[$region_0_4]], scatter=%[[$region_1_8]], metadata=

func.func @main(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
  window_dimensions = array<i64: 1, 2, 2, 1>,
  window_strides = array<i64: 1, 2, 2, 1>,
  padding = dense<0> : tensor<4x2xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %0 : tensor<10x24x24x64xf32>
}
// CHECK-DIRECT: stablehlo.select_and_scatter

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,2]{1,0}, s32[])->f32[4,<=2]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,2] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  ROOT %[[set_dimension_size_3:[^ ]+]] = f32[4,<=2] set-dimension-size(%[[Arg_0_1]], %[[Arg_1_2]]), dimensions={1}, metadata=

module {
  func.func @main(%arg0: tensor<4x2xf32>, %arg1: tensor<i32>) -> tensor<4x2xf32> {
  %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
  }
}
// CHECK-DIRECT: stablehlo.set_dimension_size

// -----

// CHECK-LABEL: HloModule main, entry_computation_layout={(f32[4,4]{1,0}, s32[])->f32[4,<=4]{1,0}}

// CHECK:       ENTRY %[[$main_4:[^ ]+]]
// CHECK-NEXT:  %[[Arg_0_1:[^ ]+]] = f32[4,4] parameter(0)
// CHECK-NEXT:  %[[Arg_1_2:[^ ]+]] = s32[] parameter(1)
// CHECK-NEXT:  ROOT %[[set_dimension_size_3:[^ ]+]] = f32[4,<=4] set-dimension-size(%[[Arg_0_1]], %[[Arg_1_2]]), dimensions={1}, metadata=

module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<i32>) -> tensor<4x?xf32, #stablehlo.bounds<?, 4>> {
  %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x4xf32>, tensor<i32>) -> tensor<4x?xf32, #stablehlo.bounds<?, 4>>
  return %0 : tensor<4x?xf32, #stablehlo.bounds<?, 4>>
  }
}
// CHECK-DIRECT: stablehlo.set_dimension_size
