// RUN: xla-cpu-opt %s -xla-legalize-collective-ops | FileCheck %s

func.func @max_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
   func.return %0 : tensor<10xf32>
}

// CHECK-LABEL: @max_reduce
//  CHECK-SAME: %[[ARG0:.*]]: tensor<10xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<10xf32>
//       CHECK: %[[RET:.*]] = "xla_cpu.all_reduce"(%[[ARG0]], %[[DST]]) {
//  CHECK-SAME:   channel_handle = 5 : i64,
//  CHECK-SAME:   reduction_kind = 3 : i32,
//  CHECK-SAME:   replica_groups = dense<{{\[}}[0, 2, 4, 6], [1, 3, 5, 7]]>
//  CHECK-SAME:   use_global_device_ids = 1
//       CHECK: return %[[RET]]

func.func @and_reduce(%arg0: tensor<1xi1>) -> tensor<1xi1> {
  %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<i1>, %rhs: tensor<i1>):
    %1 = mhlo.and %lhs, %rhs : tensor<i1>
    mhlo.return %1 : tensor<i1>
  }) {
    replica_groups = dense<> : tensor<0x0xi64>
  } : (tensor<1xi1>) -> tensor<1xi1>
  func.return %0 : tensor<1xi1>
}

// CHECK-LABEL: @and_reduce
//       CHECK:   reduction_kind = 2 : i32,

func.func @or_reduce(%arg0: tensor<1xi1>) -> tensor<1xi1> {
  %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<i1>, %rhs: tensor<i1>):
    %1 = mhlo.or %lhs, %rhs : tensor<i1>
    mhlo.return %1 : tensor<i1>
  }) {
    replica_groups = dense<> : tensor<0x0xi64>
  } : (tensor<1xi1>) -> tensor<1xi1>
  func.return %0 : tensor<1xi1>
}

// CHECK-LABEL: @or_reduce
//       CHECK:   reduction_kind = 3 : i32,

func.func @min_reduce_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.minimum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<> : tensor<0x0xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >
  } : (tensor<?xf32>) -> tensor<?xf32>
   func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @min_reduce
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[DIM]])
//       CHECK: "xla_cpu.all_reduce"(%[[ARG0]], %[[DST]])
//  CHECK-SAME:   reduction_kind = 2
//  CHECK-SAME:   use_global_device_ids = 0

func.func @partition_id() -> tensor<ui32> {
  %0 = "mhlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// CHECK-LABEL: @partition_id
// CHECK: %[[ID:.*]] = "xla_cpu.partition_id"() : () -> i32
// CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ID]] : tensor<i32>
// CHECK: %[[CAST:.*]] = mhlo.convert %[[TENSOR]] : (tensor<i32>) -> tensor<ui32>
// CHECK: return %[[CAST]]

func.func @replica_id() -> tensor<ui32> {
  %0 = "mhlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// CHECK-LABEL: @replica_id
// CHECK: %[[ID:.*]] = "xla_cpu.replica_id"() : () -> i32
// CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ID]] : tensor<i32>
// CHECK: %[[CAST:.*]] = mhlo.convert %[[TENSOR]] : (tensor<i32>) -> tensor<ui32>
// CHECK: return %[[CAST]]

func.func @collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: @collective_permute
//  CHECK-SAME: %[[ARG0:.*]]: tensor<16x8xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<16x8xf32>
//       CHECK: %[[RET:.*]] = "xla_cpu.collective_permute"(%[[ARG0]], %[[DST]]) {
//  CHECK-SAME:    channel_handle = 1
//  CHECK-SAME:    source_target_pairs = dense<
//       CHECK: return %[[RET]]

func.func @collective_permute_dynamic(%arg0: tensor<16x?xf32>)
    -> tensor<16x?xf32> {
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<16x?xf32>) -> tensor<16x?xf32>
  func.return %0 : tensor<16x?xf32>
}

// CHECK-LABEL: @collective_permute_dynamic
//  CHECK-SAME: %[[ARG0:.*]]: tensor<16x?xf32>
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[DIM]]) : tensor<16x?xf32>
//       CHECK: "xla_cpu.collective_permute"(%[[ARG0]], %[[DST]]) {

func.func @all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    channel_handle = #mhlo.channel_handle<handle = 2, type = 0>,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// CHECK-LABEL: @all_to_all
//  CHECK-SAME: %[[ARG0:.*]]: tensor<4x16xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<16x4xf32>
//       CHECK: %[[RET:.*]] = "xla_cpu.all_to_all"(%[[ARG0]], %[[DST]]) {
//  CHECK-SAME:    channel_id_present = 1
//  CHECK-SAME:    concat_dimension = 0
//  CHECK-SAME:    op_id = 2
//  CHECK-SAME:    replica_groups = dense<
//  CHECK-SAME:    split_count = 4
//  CHECK-SAME:    split_dimension = 1
//       CHECK: return %[[RET]]

func.func @all_to_all_dynamic_concat_dim(%arg0: tensor<?x16xf32>)
    -> tensor<?x4xf32> {
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<?x16xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: @all_to_all_dynamic_concat_dim
//  CHECK-SAME: %[[ARG0:.*]]: tensor<?x16xf32>
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK: %[[CONCAT_DIM:.*]] = arith.muli %[[DIM]], %[[C4]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[CONCAT_DIM]]) : tensor<?x4xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0]], %[[DST]]) {

func.func @all_to_all_dynamic_split_dim(%arg0: tensor<4x?xf32>)
    -> tensor<16x?xf32> {
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x?xf32>) -> tensor<16x?xf32>
  func.return %0 : tensor<16x?xf32>
}

// CHECK-LABEL: @all_to_all_dynamic_split_dim
//  CHECK-SAME: %[[ARG0:.*]]: tensor<4x?xf32>
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1
//   CHECK-DAG: %[[C4:.*]] = arith.constant 4
//       CHECK: %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK: %[[CONCAT_DIM:.*]] = arith.divui %[[DIM]], %[[C4]]
//       CHECK: %[[DST:.*]] = tensor.empty(%[[CONCAT_DIM]]) : tensor<16x?xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0]], %[[DST]]) {

func.func @all_to_all_tuple(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>)
    -> (tensor<128x4xf32>, tensor<128x4xf32>) {
  %0:2 = "mhlo.all_to_all"(%arg0, %arg1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  } : (tensor<128x4xf32>, tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>)
  return %0#0, %0#1 : tensor<128x4xf32>, tensor<128x4xf32>
}

// CHECK-LABEL: @all_to_all_tuple
//  CHECK-SAME: %[[ARG0:.*]]: tensor<128x4xf32>,
//  CHECK-SAME: %[[ARG1:.*]]: tensor<128x4xf32>
//       CHECK: %[[DST0:.*]] = tensor.empty() : tensor<128x4xf32>
//       CHECK: %[[DST1:.*]] = tensor.empty() : tensor<128x4xf32>
//       CHECK: "xla_cpu.all_to_all"(%[[ARG0]], %[[ARG1]], %[[DST0]], %[[DST1]])

func.func @outfeed_0_input(%token: !mhlo.token) -> !mhlo.token {
  %res = "mhlo.outfeed"(%token) {outfeed_config = "foobar"} : (!mhlo.token) -> !mhlo.token
  func.return %res : !mhlo.token
}

// CHECK-LABEL: @outfeed_0_input
//       CHECK: "xla_cpu.outfeed"() {config = "foobar", result_type = []} : () -> ()

func.func @outfeed_1_input(%data: tensor<2xui32>, %token: !mhlo.token)
  -> !mhlo.token attributes {xlaframework.result_mapping = 1 : i32} {
    %res = "mhlo.outfeed"(%data, %token) {
      outfeed_config = "", xla_shape = "token[]"
      } : (tensor<2xui32>, !mhlo.token) -> !mhlo.token
    func.return %res : !mhlo.token
}

// CHECK-LABEL: @outfeed_1_input
//  CHECK-SAME: %[[DATA:.*]]: tensor<2xui32>
//  CHECK-SAME: %[[TOKEN:.*]]: !mhlo.token
//       CHECK: "xla_cpu.outfeed"(%[[DATA]]) {config = "", result_type = [ui32]} : (tensor<2xui32>) -> ()
//       CHECK: return %[[TOKEN]] : !mhlo.token

func.func @outfeed_2_input(%data1: tensor<3xui32>, %data2: tensor<3xi32>, %token: !mhlo.token) -> !mhlo.token {
  %res = "mhlo.outfeed"(%data1, %data2,  %token) {outfeed_config = "foobar"}
    : (tensor<3xui32>, tensor<3xi32>, !mhlo.token) -> !mhlo.token
  func.return %res : !mhlo.token
}

// CHECK-LABEL: @outfeed_2_input
//  CHECK-SAME: %[[ARG0:.*]]: tensor<3xui32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<3xi32>
//       CHECK: "xla_cpu.outfeed"(%[[ARG0]], %[[ARG1]]) {config = "foobar", result_type = [ui32, i32]}
//  CHECK-SAME: (tensor<3xui32>, tensor<3xi32>)

func.func @add_dependency(%arg0: tensor<16xf32>, %arg1: !mhlo.token) -> tensor<16xf32> {
  %0 = "mhlo.add_dependency"(%arg0, %arg1) : (tensor<16xf32>, !mhlo.token) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// CHECK-LABEL: @add_dependency
//  CHECK-SAME: %[[ARG0:.*]]: tensor<16xf32>
//  CHECK-SAME: %[[ARG1:.*]]: !mhlo.token
//       CHECK: %[[RES:.*]] = "xla_cpu.add_dependency"
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<16xf32>

func.func @conv_i4(%arg0: tensor<64x8x8x8xi4>, %arg1: tensor<4x4x8x32xi4>)
  -> tensor<64x3x3x32xi8> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [2, 2]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64} :
       (tensor<64x8x8x8xi4>, tensor<4x4x8x32xi4>) -> tensor<64x3x3x32xi8>
  func.return %0 : tensor<64x3x3x32xi8>
}

// CHECK-LABEL: @conv_i4
//  CHECK-SAME: %[[ARG0:.*]]: tensor<64x8x8x8xi4>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<4x4x8x32xi4>
//       CHECK: %[[RES:.*]] = mhlo.convolution
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<64x3x3x32xi8>

func.func @conv_0d_nc(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>)
  -> tensor<3x3xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, f]x[i, o]->[b, f],
     window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [],
               reverse = []}
     {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
   : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @conv_0d_nc
//  CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<2x3xf32>
//       CHECK: %[[RES:.*]] = mhlo.convolution
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<3x3xf32>

func.func @conv_1d_nwc_dyn(%arg0: tensor<?x8x?xf32>, %arg1: tensor<2x?x?xf32>)
  -> tensor<?x7x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 2,
      input_spatial_dimensions = [1],
      kernel_input_feature_dimension = 1,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0],
      output_batch_dimension = 0,
      output_feature_dimension = 2,
      output_spatial_dimensions = [1]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0]]> : tensor<1x2xi64>,
    rhs_dilation = dense<1> : tensor<1xi64>,
    window_strides = dense<1> : tensor<1xi64>,
    someattr
  } : (tensor<?x8x?xf32>, tensor<2x?x?xf32>) -> tensor<?x7x?xf32>
  func.return %0 : tensor<?x7x?xf32>
}

// CHECK-LABEL: @conv_1d_nwc_dyn
//  CHECK-SAME: %[[ARG0:.*]]: tensor<?x8x?xf32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<2x?x?xf32>
//       CHECK: %[[RES:.*]] = mhlo.convolution
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<?x7x?xf32>

func.func @depthwise_conv1d(%arg0: tensor<1x10x8xf32>,
                            %arg1: tensor<3x1x16xf32>) -> tensor<1x10x16xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [1],
      pad = [[1, 1]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 8 : i64,
    someattr} : (tensor<1x10x8xf32>, tensor<3x1x16xf32>) -> tensor<1x10x16xf32>
  func.return %0 : tensor<1x10x16xf32>
}

// CHECK-LABEL: @depthwise_conv1d
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x10x8xf32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<3x1x16xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<1x10x16xf32>
//       CHECK: %[[RES:.*]] = "xla_cpu.convolution"
//  CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[DST]]
//       CHECK: return %[[RES]] : tensor<1x10x16xf32>

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x4x5x1xf32>, %arg1: tensor<3x2x1x1xf32>)
  -> tensor<1x2x4x1xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x4x5x1xf32>, tensor<3x2x1x1xf32>) -> tensor<1x2x4x1xf32>
  func.return %0 : tensor<1x2x4x1xf32>
}

// CHECK-LABEL: @conv_2d_nhwc_hwcf
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x4x5x1xf32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<3x2x1x1xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<1x2x4x1xf32>
//       CHECK: %[[RES:.*]] = "xla_cpu.convolution"
//  CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[DST]]
//       CHECK: return %[[RES]] : tensor<1x2x4x1xf32>

func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<1x8x8x8x1xf32>,
                               %arg1: tensor<2x2x2x1x1xf32>)
  -> tensor<1x7x7x7x1xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 4,
      input_spatial_dimensions = [1, 2, 3],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 4,
      kernel_spatial_dimensions = [0, 1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 4,
      output_spatial_dimensions = [1, 2, 3]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
    lhs_dilation = dense<1> : tensor<3xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    window_strides = dense<1> : tensor<3xi64>
  } : (tensor<1x8x8x8x1xf32>, tensor<2x2x2x1x1xf32>) -> tensor<1x7x7x7x1xf32>
  func.return %0 : tensor<1x7x7x7x1xf32>
}

// CHECK-LABEL: @conv_3d_ndhwc_dhwcf
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x8x8x8x1xf32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<2x2x2x1x1xf32>
//       CHECK: %[[DST:.*]] = tensor.empty() : tensor<1x7x7x7x1xf32>
//       CHECK: %[[RES:.*]] = "xla_cpu.convolution"
//  CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[DST]]
//       CHECK: return %[[RES]] : tensor<1x7x7x7x1xf32>

func.func @normal_convolution_with_reversal(%arg0: tensor<1x3x3x3xf32>,
    %arg1: tensor<3x3x3x1xf32>) -> tensor<1x1x1x1xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [0, 0]],
        lhs_dilate = [1, 1],
        rhs_dilate = [1, 1],
        reverse = [1, 1]
      } {
        batch_group_count = 1 : i64,
        feature_group_count = 1 : i64, precision_config = [
          #mhlo<precision DEFAULT>,
          #mhlo<precision DEFAULT>]
      } : (tensor<1x3x3x3xf32>, tensor<3x3x3x1xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// CHECK-LABEL: @normal_convolution_with_reversal
//  CHECK-SAME: %[[ARG0:.*]]: tensor<1x3x3x3xf32>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<3x3x3x1xf32>
//       CHECK: %[[RES:.*]] = mhlo.convolution
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<1x1x1x1xf32>

func.func @general_convolution_with_zero_sized_dimension_in_output(
  %arg0: tensor<2x4x9x0xi64> {bufferization.writable = false,
                              xla_framework.input_mapping = 2 : i32},
  %arg1: tensor<4x5x2x4xi64> {bufferization.writable = false,
                              xla_framework.input_mapping = 0 : i32})
  -> tensor<2x5x0x4xi64> attributes {xla_framework.result_mapping = 1 : i32} {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 1], pad = [[1, 2], [2, 0]], lhs_dilate = [1, 4],
              rhs_dilate = [1, 1], reverse = [0, 0]}
    {batch_group_count = 1 : i64, feature_group_count = 2 : i64,
     precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
    : (tensor<2x4x9x0xi64>, tensor<4x5x2x4xi64>) -> tensor<2x5x0x4xi64>
  return %0 : tensor<2x5x0x4xi64>
}

// CHECK-LABEL: @general_convolution_with_zero_sized_dimension_in_output
//  CHECK-SAME: %[[ARG0:.*]]: tensor<2x4x9x0xi64>
//  CHECK-SAME: %[[ARG1:.*]]: tensor<4x5x2x4xi64>
//       CHECK: %[[RES:.*]] = mhlo.convolution
//  CHECK-SAME: %[[ARG0]], %[[ARG1]]
//       CHECK: return %[[RES]] : tensor<2x5x0x4xi64>

func.func @foo(%0: tensor<3x9x9x8xf32>, %1: tensor<1x7x8x8xf32>) -> tensor<3x9x9x8xf32> {
  %2 = mhlo.convolution(%0, %1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3x9x9x8xf32>, tensor<1x7x8x8xf32>) -> tensor<3x9x9x8xf32>
  return %2 : tensor<3x9x9x8xf32>
}
