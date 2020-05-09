// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK:  HloModule
func @main(%arg0: !xla_hlo.token, %arg1: !xla_hlo.token) -> !xla_hlo.token {
  %0 = "xla_hlo.after_all"(%arg0, %arg1) : (!xla_hlo.token, !xla_hlo.token) -> !xla_hlo.token
  return %0 : !xla_hlo.token
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = token[] parameter(0)
// CHECK:  %[[ARG1:.*]] = token[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = token[] after-all(token[] %[[ARG0]], token[] %[[ARG1]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "xla_hlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = xla_hlo.maximum %lhs, %rhs : tensor<f32>
    "xla_hlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_id = {
      handle = 5 : i64,
      type = 2 : i64
    }
  } : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME:  replica_groups={{[{][{]}}0,2,4,6},{1,3,5,7{{[}][}]}}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----

// CHECK:  HloModule
func @main(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0 = "xla_hlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  return %0 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,2,2,2] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[2] parameter(2)
// CHECK:  [[VAL_4:%.*]] = f32[2] parameter(3)
// CHECK:  [[VAL_5:%.*]] = f32[2,2,2,2] parameter(4)
// CHECK:  ROOT
// CHECK-SAME:  (f32[2,2,2,2], f32[2], f32[2]) batch-norm-grad(f32[2,2,2,2] [[VAL_1]], f32[2] [[VAL_2]], f32[2] [[VAL_3]], f32[2] [[VAL_4]], f32[2,2,2,2] [[VAL_5]]), epsilon=0.001, feature_index=0

// -----

// CHECK:  HloModule
func @main(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0 = "xla_hlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 3 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  return %0 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,2,2,2] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[2] parameter(2)
// CHECK:  ROOT
// CHECK-SAME:  (f32[2,2,2,2], f32[2], f32[2]) batch-norm-training(f32[2,2,2,2] [[VAL_1]], f32[2] [[VAL_2]], f32[2] [[VAL_3]]), epsilon=0.001, feature_index=3

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK:  [[VAL_1:%.*]] = s32[4] parameter(0)
  // CHECK:  [[VAL_2:%.*]] = s32[4] parameter(1)
  // CHECK:  [[ATAN2:%.*]] = s32[4] atan2(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
  %0 = xla_hlo.atan2 %arg0, %arg1 : tensor<4xi32>

  // CHECK:  [[SHL:%.*]] = s32[4] shift-left(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
  %1 = xla_hlo.shift_left %arg0, %arg1 : tensor<4xi32>

  // CHECK:  [[SHRA:%.*]] = s32[4] shift-right-arithmetic(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
  %2 = xla_hlo.shift_right_arithmetic %arg0, %arg1 : tensor<4xi32>

  // CHECK:  [[SHRL:%.*]] = s32[4] shift-right-logical(s32[4] [[VAL_1]], s32[4] [[VAL_2]])
  %3 = xla_hlo.shift_right_logical %arg0, %arg1 : tensor<4xi32>

  // CHECK:  ROOT
  // CHECK-SAME:  [[VAL_7:%.*]] = (s32[4], s32[4], s32[4], s32[4]) tuple(s32[4] [[ATAN2]], s32[4] [[SHL]], s32[4] [[SHRA]], s32[4] [[SHRL]])
  return %0, %1, %2, %3 : tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<1x4xi32>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3x4xi32>) -> tensor<2x3x4xi32> {
  // Same rank degenerate broadcast
  // CHECK:  [[ARG_0:%.*]] = s32[1,4] parameter(0)
  // CHECK-NEXT:  [[RESHAPE_1:%.*]] = s32[4] reshape(s32[1,4] [[ARG_0]])
  // CHECK-NEXT:  [[BROADCAST_1:%.*]] = s32[2,4] broadcast(s32[4] [[RESHAPE_1]])
  // CHECK-NEXT:  [[ARG_1:%.*]] = s32[2,4] parameter(1)
  // CHECK-NEXT:  s32[2,4] add(s32[2,4] [[BROADCAST_1]], s32[2,4] [[ARG_1]])
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<1x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>

  // Broadcast up rank
  // CHECK-NEXT:  [[BROADCAST_2:%.*]] = s32[2,3,4] broadcast(s32[2,4] [[ARG_1]]), dimensions={0,2}
  // CHECK-NEXT:  [[ARG_2:%.*]] = s32[2,3,4] parameter(2)
  // CHECK-NEXT:  s32[2,3,4] add(s32[2,3,4] [[BROADCAST_2]], s32[2,3,4] [[ARG_2]])
  %1 = "xla_hlo.add"(%arg1, %arg2) {broadcast_dimensions = dense<[0,2]> : tensor<2xi64>} : (tensor<2x4xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>

  // Broadcast up rank + degenerate broadcast
  // CHECK-NEXT:  [[BROADCAST_3:%.*]] = s32[2,1,4] broadcast(s32[1,4] [[ARG_0]]), dimensions={1,2}
  // CHECK-NEXT:  [[RESHAPE_2:%.*]] = s32[2,4] reshape(s32[2,1,4] [[BROADCAST_3]])
  // CHECK-NEXT:  [[BROADCAST_4:%.*]] = s32[2,3,4] broadcast(s32[2,4] [[RESHAPE_2]]), dimensions={0,2}
  // CHECK:  ROOT
  // CHECK-SAME:  s32[2,3,4] add(s32[2,3,4] [[BROADCAST_4]], s32[2,3,4] [[ARG_2]])
  %2 = "xla_hlo.add"(%arg0, %arg2) {broadcast_dimensions = dense<[1,2]> : tensor<2xi64>} : (tensor<1x4xi32>, tensor<2x3x4xi32>) -> tensor<2x3x4xi32>
  return %2 : tensor<2x3x4xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "xla_hlo.bitcast_convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = s32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2] bitcast-convert(s32[2] %[[ARG]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xi32>) -> tensor<1x2x3x4xi32> {
  // CHECK:  [[ARG:%.*]] = s32[4] parameter(0)
  // CHECK-NEXT:  ROOT %broadcast.2 = s32[1,2,3,4] broadcast(s32[4] [[ARG]]), dimensions={3}
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[1,2,3]> : tensor<3xi64>} : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  return %0 : tensor<1x2x3x4xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %result = "xla_hlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>) -> tensor<1x10xf32>
  return %result : tensor<1x10xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[1] parameter(0)
// CHECK:  ROOT %broadcast.2 = f32[1,10] broadcast(f32[1] [[ARG]]), dimensions={0}

// -----

// CHECK:  HloModule
func @main() -> !xla_hlo.token {
  %0 = "xla_hlo.create_token"() : () -> !xla_hlo.token
  return %0 : !xla_hlo.token
}

// CHECK:  ROOT [[TOKEN:%.*]] = token[] after-all()

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = call @callee(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = call @callee(%0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

func @callee(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK:  [[CALLEE_1:%.*]] ([[ARG_1:.*]]: s32[4], [[ARG_2:.*]]: s32[4]) -> s32[4] {
// CHECK:  %[[ARG_1]] = s32[4] parameter(0)
// CHECK:  %[[ARG_2]] = s32[4] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  s32[4] add(s32[4] %[[ARG_1]], s32[4] %[[ARG_2]])

// CHECK:  [[CALLEE_2:%.*]] ([[ARG_3:.*]]: s32[4], [[ARG_4:.*]]: s32[4]) -> s32[4] {
// CHECK:  %[[ARG_3]] = s32[4] parameter(0)
// CHECK:  %[[ARG_4]] = s32[4] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  s32[4] add(s32[4] %[[ARG_3]], s32[4] %[[ARG_4]])

// CHECK:  ENTRY [[MAIN:%.*]] ([[ARG:.*]]: s32[4]) -> s32[4] {
// CHECK:  %[[ARG]] = s32[4] parameter(0)
// CHECK:  [[CALL_OUT:%.*]] = s32[4] call(s32[4] %[[ARG]], s32[4] %[[ARG]]), to_apply=[[CALLEE_1]]
// CHECK:  ROOT
// CHECK-SAME:  s32[4] call(s32[4] [[CALL_OUT]], s32[4] [[CALL_OUT]]), to_apply=[[CALLEE_2]]

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0:2 = call @callee(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
  return %0#0, %0#1 : tensor<4xi32>, tensor<4xi32>
}

func @callee(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "xla_hlo.multiply"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0, %1 : tensor<4xi32>, tensor<4xi32>
}

// Get name of callee computation
// CHECK:  [[CALLEE:%.*]] ({{.*}}) -> ({{.*}}) {

// CHECK:  ENTRY
// CHECK-SAME:  [[MAIN:%.*]] ([[ARG:.*]]: s32[4]) -> (s32[4], s32[4]) {
// CHECK:  %[[ARG]] = s32[4] parameter(0)
// CHECK:  [[CALL_OUT:%.*]] = (s32[4], s32[4]) call(s32[4] %[[ARG]], s32[4] %[[ARG]]), to_apply=[[CALLEE]]
// CHECK:  [[OUT_0:%.*]] = s32[4] get-tuple-element((s32[4], s32[4]) [[CALL_OUT]]), index=0
// CHECK:  [[OUT_1:%.*]] = s32[4] get-tuple-element((s32[4], s32[4]) [[CALL_OUT]]), index=1
// CHECK:  ROOT
// CHECK-SAME:  (s32[4], s32[4]) tuple(s32[4] [[OUT_0]], s32[4] [[OUT_1]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "xla_hlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}
// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[128,32] parameter(0)
// CHECK:  ROOT [[RESULT:%.*]] = f32[128,32] collective-permute(f32[128,32] [[ARG]]), source_target_pairs={{\{\{}}0,1},{1,2},{2,3}}

// -----

// CHECK:  HloModule
func @main(%arg0 : tensor<5x2xf32>,
           %arg1 : tensor<5x5xf32>,
           %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  %result = "xla_hlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 1 : i64
  } : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  return %result : tensor<5x14xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[5,2] parameter(0)
// CHECK:  %[[ARG1:.*]] = f32[5,5] parameter(1)
// CHECK:  %[[ARG2:.*]] = f32[5,7] parameter(2)
// CHECK:  ROOT %[[RESULT:.*]] = f32[5,14] concatenate(f32[5,2] %[[ARG0]], f32[5,5] %[[ARG1]], f32[5,7] %[[ARG2]]), dimensions={1}

// -----

// CHECK:  HloModule
func @main() {
  // CHECK:  constant.{{.*}} = s64[] constant(1)
  %cst = constant dense<1> : tensor<i64>
  // CHECK:  constant.{{.*}} = f32[2,2,1,1]
  // CHECK-SAME:  { { /*i0=0*/ { /*i1=0*/ {1} }, { /*i1=1*/ {2} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {4} } } }
  %cst_0 = constant dense<
    [[[[1.000000e+00]], [[2.000000e+00]]], [[[3.000000e+00]], [[4.000000e+00]]]]
  > : tensor<2x2x1x1xf32>

  // CHECK:  s32[1] constant({1})
  %cst_1 = constant dense<1> : tensor<1xi32>

  // CHECK:  %[[C:.*]] = s32[] constant(1)
  // CHECK:  s32[10] broadcast(s32[] %[[C]])
  %cst_2 = constant dense<1> : tensor<10xi32>

  // CHECK:  s32[4] constant({1, 2, 3, 4})
  %cst_3 = constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  // CHECK:  s32[2,2] constant({ { 1, 2 }, { 3, 4 } })
  %cst_4 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>

  // CHECK:  s32[2,2] constant({ { 3, 2 }, { 1, 4 } })
  %cst_5 = constant dense<[[3, 2], [1, 4]]> : tensor<2x2xi32>

  // CHECK:  u32[2,2] constant({ { 1, 2 }, { 4, 8 } })
  %cst_6 = constant dense<[[1, 2], [4, 8]]> : tensor<2x2xui32>

  // CHECK: bf16[4] constant({1, 2, 3, 4})
  %cst_7 = constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xbf16>

  // CHECK: f16[4] constant({1, -4, -65504, 0.015625}
  %cst_8 = constant dense<[1.0e+00, -4.0e+00, -65504.0e+00, 1.5625e-02]> : tensor<4xf16>

  // CHECK: c64[] constant((1, 0))
  %cst_9 = constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>

  // CHECK: c128[] constant((1, 0))
  %cst_10 = constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>

  return
}

// -----

// CHECK:  HloModule
func @main(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  %result = "xla_hlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 3 : i64,
      input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
      kernel_input_feature_dimension = 3 : i64,
      kernel_output_feature_dimension = 2 : i64,
      kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 3 : i64,
      output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
    },
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  return %result : tensor<100x28x28x1xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[100,26,26,32] parameter(0)
// CHECK:  %[[ARG1:.*]] = f32[3,3,1,32] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = f32[100,28,28,1] convolution(f32[100,26,26,32] %[[ARG0]], f32[3,3,1,32] %[[ARG1]]),
// CHECK-SAME:  window={size=3x3 pad=2_2x2_2},
// CHECK-SAME:  dim_labels=b01f_01oi->b01f

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "xla_hlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = s32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2] convert(s32[2] %[[ARG]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "xla_hlo.copy"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = s32[2] copy(s32[2] [[ARG]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = xla_hlo.constant dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi32>
  %1 = "xla_hlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>} : (tensor<10xf32>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK:  %[[SUM_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> f32[]
// CHECK:  ROOT %[[RESULT:.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME:  replica_groups={{[{][{]}}0,2,4,6},{1,3,5,7{{[}][}]}}
// CHECK-SAME:  to_apply=%[[SUM_COMPUTATION]]

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
  %0 = "xla_hlo.custom_call"(%arg0, %arg1) {backend_config = "bar", call_target_name = "foo"} : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,3] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[5,5] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[1,2,3] custom-call(f32[2,3] [[VAL_1]], f32[5,5] [[VAL_2]]), custom_call_target="foo", backend_config="bar"

// -----

// CHECK:  HloModule
func @main(%arg: tensor<16x16xi32>) -> tensor<16x64xbf16> {

  %0 = "xla_hlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = false} : (tensor<16x16xi32>) -> tensor<16x64xbf16>
  return %0 : tensor<16x64xbf16>
}

// CHECK: ENTRY
// CHECK:   %[[Arg:.*]] = s32[16,16] parameter(0)
// CHECK:   u32[16,16] convert(s32[16,16] %[[Arg]])
// CHECK:   u32[4] subtract(u32[4] %{{.*}}, u32[4] %{{.*}})
// CHECK:   u32[4] multiply(u32[4] %{{.*}}, u32[4] %{{.*}})
// CHECK:   u32[4,16,16] shift-right-logical(u32[4,16,16] %{{.*}}, u32[4,16,16] %{{.*}})
// CHECK:   bf16[4,16,16] convert(u32[4,16,16] %{{.*}})
// CHECK:   bf16[4,16,16] multiply(bf16[4,16,16] %{{.*}}, bf16[4,16,16] %{{.*}})
// CHECK:   ROOT

// -----

// CHECK:  HloModule
func @main(%arg: tensor<16x16xi32>) -> tensor<16x32xbf16> {

  %0 = "xla_hlo.dequantize"(%arg) {min_range = -0.1 : f32, max_range = 0.1 : f32, mode = "MIN_COMBINED", transpose_output = false, is_16bits = true} : (tensor<16x16xi32>) -> tensor<16x32xbf16>
  return %0 : tensor<16x32xbf16>
}

// CHECK: ENTRY
// CHECK:   %[[Arg:.*]] = s32[16,16] parameter(0)
// CHECK:   u32[16,16] convert(s32[16,16] %[[Arg]])
// CHECK:   u32[2] subtract(u32[2] %{{.*}}, u32[2] %{{.*}})
// CHECK:   u32[2] multiply(u32[2] %{{.*}}, u32[2] %{{.*}})
// CHECK:   u32[2,16,16] shift-right-logical(u32[2,16,16] %{{.*}}, u32[2,16,16] %{{.*}})
// CHECK:   bf16[2,16,16] convert(u32[2,16,16] %{{.*}})
// CHECK:   bf16[2,16,16] multiply(bf16[2,16,16] %{{.*}}, bf16[2,16,16] %{{.*}})
// CHECK:   ROOT

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<3x4xi32>, %arg1: tensor<4x5xi32>) -> tensor<3x5xi32> {
  // Simple einsum is lowered to HLO dot op.
  // CHECK:  dot(s32[3,4] %{{.*}}, s32[4,5] %{{.*}}), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %0 = "xla_hlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<3x4xi32>, tensor<4x5xi32>) -> tensor<3x5xi32>
  return %0 : tensor<3x5xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>> {
  %0 = "xla_hlo.fft"(%arg0) {fft_length = dense<9> : tensor<1xi64>, fft_type = "RFFT"} : (tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>>
  return %0 : tensor<3x5xcomplex<f32>>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[3,9] parameter(0)
// CHECK:  c64[3,5] fft(f32[3,9] [[ARG]]), fft_type=RFFT, fft_length={9}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<10x300xf32> {
  // CHECK:  [[ARG0:%.*]] = f32[200,100,300] parameter(0)
  // CHECK:  [[ARG1:%.*]] = s32[10,2] parameter(1)
  // CHECK:  f32[10,300] gather(f32[200,100,300] [[ARG0]], s32[10,2] [[ARG1]])
  // CHECK-SAME:  offset_dims={1}
  // CHECK-SAME:  collapsed_slice_dims={0,1}
  // CHECK-SAME:  start_index_map={0,1}
  // CHECK-SAME:  index_vector_dim=1
  // CHECK-SAME:  slice_sizes={1,1,300}
  // CHECK-SAME:  indices_are_sorted=true
  %0 = "xla_hlo.gather"(%arg0, %arg1) {dimension_numbers = {collapsed_slice_dims = dense<[0, 1]> : tensor<2xi64>, index_vector_dim = 1 : i64, offset_dims = dense<1> : tensor<1xi64>, start_index_map = dense<[0, 1]> : tensor<2xi64>}, indices_are_sorted = true, name = "gather", slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>} : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
    return %0 : tensor<10x300xf32>
}

// -----

// CHECK:  HloModule
func @main(%arg: tensor<4x2xf32>, %size: tensor<i32>) -> tensor<i32> {
  %0 = "xla_hlo.set_dimension_size"(%arg, %size) {dimension = 1 : i32} : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
  %1 = "xla_hlo.get_dimension_size"(%0) {dimension = 1 : i32} : (tensor<4x2xf32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,2] parameter(0)
// CHECK:  [[SIZE:%.*]] = s32[] parameter(1)
// CHECK:  [[DYNAMIC:%.*]] = f32[4,<=2] set-dimension-size(f32[4,2] [[ARG]], s32[] [[SIZE]]), dimensions={1}
// CHECK:  ROOT %[[RESULT:.*]] = s32[] get-dimension-size(f32[4,<=2] [[DYNAMIC]]), dimensions={1}


// -----

// CHECK:  HloModule
func @main(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "xla_hlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = (f32[], s32[]) parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[] get-tuple-element((f32[], s32[]) %[[ARG0]]), index=0

// -----

// CHECK:  HloModule
func @main(%arg0: !xla_hlo.token) -> tuple<tuple<tensor<3xi32>, tensor<i1>>, !xla_hlo.token> {
  %0 = "xla_hlo.infeed"(%arg0) {infeed_config = "foobar"} : (!xla_hlo.token) -> tuple<tuple<tensor<3xi32>, tensor<i1>>, !xla_hlo.token>
  return %0 : tuple<tuple<tensor<3xi32>, tensor<i1>>, !xla_hlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = ((s32[3], pred[]), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"

// -----

// CHECK:  HloModule
func @main() -> tensor<1x10xf32> {
  %result = "xla_hlo.iota"() {
    iota_dimension = 1 : i64
  } : () -> tensor<1x10xf32>
  return %result : tensor<1x10xf32>
}

// CHECK:  ENTRY
// CHECK:  ROOT %[[RESULT:.*]] = f32[1,10] iota(), iota_dimension=1

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "xla_hlo.map"(%arg0, %arg1) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):       // no predecessors
    %1 = xla_hlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK:  [[COMPUTATION:%.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[] {
// CHECK:  [[ARG_0:%.*]] = f32[] parameter(0)
// CHECK:  [[ARG_1:%.*]] = f32[] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[] add(f32[] [[ARG_0]], f32[] [[ARG_1]])
// CHECK:  }

// CHECK:  ENTRY
// CHECK:  [[ARG_2:%.*]] = f32[4] parameter(0)
// CHECK:  [[ARG_3:%.*]] = f32[4] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[4] map(f32[4] [[ARG_2]], f32[4] [[ARG_3]]), dimensions={0}, to_apply=[[COMPUTATION]]

// -----

// CHECK:  HloModule
func @main(%data: tensor<3xi32>, %token: !xla_hlo.token) -> !xla_hlo.token {
  %0 = "xla_hlo.outfeed"(%data, %token) {outfeed_config = "foobar"} : (tensor<3xi32>, !xla_hlo.token) -> !xla_hlo.token
  return %0 : !xla_hlo.token
}

// CHECK:  ENTRY
// CHECK:  [[DATA:%.*]] = s32[3] parameter(0)
// CHECK:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = token[] outfeed(s32[3] [[DATA]], token[] [[TOKEN]]), outfeed_config="foobar"

// -----

// CHECK:  HloModule
func @main(%arg: tensor<4x6xf32>, %pad: tensor<f32>) -> tensor<13x19xf32> {
  %0 = "xla_hlo.pad"(%arg, %pad) {edge_padding_high = dense<[4,5]> : tensor<2xi64>, edge_padding_low = dense<[2,3]> : tensor<2xi64>, interior_padding = dense<1> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<13x19xf32>
  return %0 : tensor<13x19xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,6] parameter(0)
// CHECK:  [[PADDING_VAL:%.*]] = f32[] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[13,19] pad(f32[4,6] [[ARG]], f32[] [[PADDING_VAL]]), padding=2_4_1x3_5_1

// -----

// CHECK:  HloModule
func @main(%token: !xla_hlo.token) -> tuple<tensor<3x4xi32>, !xla_hlo.token> {
  %0 = "xla_hlo.recv"(%token) {
    channel_id = {
      handle = 5 : i64,
      type = 3 : i64  // Host to device channel
    },
    is_host_transfer = true
  } : (!xla_hlo.token) -> tuple<tensor<3x4xi32>, !xla_hlo.token>
  return %0 : tuple<tensor<3x4xi32>, !xla_hlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[RECV:%.*]] = (s32[3,4], u32[], token[]) recv(token[] [[TOKEN]]), channel_id=5, is_host_transfer=true
// CHECK:  ROOT
// CHECK-SAME:  (s32[3,4], token[]) recv-done((s32[3,4], u32[], token[]) [[RECV]]), channel_id=5, is_host_transfer=true

// -----

// CHECK:  HloModule
func @main(%token: !xla_hlo.token) -> tuple<tensor<3x4xi32>, !xla_hlo.token> {
  %0 = "xla_hlo.recv"(%token) {
    channel_id = {
      handle = 5 : i64,
      type = 1 : i64  // Device to device channel
    },
    is_host_transfer = false
  } : (!xla_hlo.token) -> tuple<tensor<3x4xi32>, !xla_hlo.token>
  return %0 : tuple<tensor<3x4xi32>, !xla_hlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[RECV:%.*]] = (s32[3,4], u32[], token[]) recv(token[] [[TOKEN]]), channel_id=5
// CHECK:  ROOT
// CHECK-SAME:  (s32[3,4], token[]) recv-done((s32[3,4], u32[], token[]) [[RECV]]), channel_id=5


// -----

// CHECK:  HloModule
func @main(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1x10xi32>, %arg2 : tensor<f32>, %arg3 : tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %result0, %result1 = "xla_hlo.reduce"(%arg0, %arg1, %arg2, %arg3) ( {
    ^bb0(%fa: tensor<f32>, %ia : tensor<i32>, %fb: tensor<f32>, %ib: tensor<i32>):   // no predecessors
      %fmax = "xla_hlo.maximum"(%fa, %fb) {} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %imax = "xla_hlo.maximum"(%ia, %ib) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "xla_hlo.return"(%fmax, %imax) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  return %result0, %result1 : tensor<1xf32>, tensor<1xi32>
}

// CHECK:  %[[REGION:region_[0-9]+]]
// CHECK-SAME:  ([[ARG_FA:.*]]: f32[], [[ARG_IA:.*]]: s32[], [[ARG_FB:.*]]: f32[], [[ARG_IB:.*]]: s32[]) -> (f32[], s32[])
// CHECK:  %[[FMAX:.*]] = f32[] maximum(f32[] %[[ARG_FA]], f32[] %[[ARG_FB]])
// CHECK:  %[[IMAX:.*]] = s32[] maximum(s32[] %[[ARG_IA]], s32[] %[[ARG_IB]])
// CHECK:  ROOT %[[RESULT_REGION:.*]] = (f32[], s32[]) tuple(f32[] %[[FMAX]], s32[] %[[IMAX]])

// CHECK:  ENTRY
// CHECK-SAME:  ([[ARG0:.*]]: f32[1,10], [[ARG1:.*]]: s32[1,10], [[ARG2:.*]]: f32[], [[ARG3:.*]]: s32[]) -> (f32[1], s32[1])
// CHECK:  %[[RESULT:.*]] = (f32[1], s32[1]) reduce(f32[1,10] %[[ARG0]], s32[1,10] %[[ARG1]], f32[] %[[ARG2]], s32[] %[[ARG3]]), dimensions={1}, to_apply=%[[REGION]]
// CHECK:  %[[RESULT0:.*]] = f32[1] get-tuple-element((f32[1], s32[1]) %[[RESULT]]), index=0
// CHECK:  %[[RESULT1:.*]] = s32[1] get-tuple-element((f32[1], s32[1]) %[[RESULT]]), index=1
// CHECK:  ROOT %[[RESULT:.*]] = (f32[1], s32[1]) tuple(f32[1] %[[RESULT0]], s32[1] %[[RESULT1]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2x17x31x7xi32>) -> tensor<2x3x5x7xi32> {
  %0 = xla_hlo.constant dense<-2147483648> : tensor<i32>
  %1 = "xla_hlo.reduce_window"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):	// no predecessors
    %2 = xla_hlo.maximum %arg1, %arg2 : tensor<i32>
    "xla_hlo.return"(%2) : (tensor<i32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>,
    base_dilations = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<2x17x31x7xi32>, tensor<i32>) -> tensor<2x3x5x7xi32>
  return %1 : tensor<2x3x5x7xi32>
}

// CHECK:  %[[MAX_COMPUTATION:.*]] ([[ARG0:.*]]: s32[], [[ARG1:.*]]: s32[]) -> s32[]
// CHECK:  ROOT %[[RESULT:.*]] = s32[] maximum(s32[] %[[ARG0]], s32[] %[[ARG1]])

// CHECK:  ENTRY
// CHECK-DAG:  %[[ARG0:.*]] = s32[2,17,31,7] parameter(0)
// CHECK-DAG:  %[[INIT:.*]] = s32[] constant(-2147483648)
// CHECK:  ROOT %[[RESULT:.*]] = s32[2,5,8,7] reduce-window(s32[2,17,31,7] %[[ARG0]], s32[] %constant.2),
// CHECK-SAME:  window={size=1x2x2x1 stride=1x4x4x1 pad=0_0x2_0x0_2x0_0 rhs_dilate=1x2x2x1},
// CHECK-SAME:  to_apply=%[[MAX_COMPUTATION]]

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2xf32>) -> tensor<1x2xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[1,2] reshape(f32[2] %[[ARG0]])

// -----

// CHECK:  HloModule
func @main(%arg0 : tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %result = "xla_hlo.reverse"(%arg0) {
    dimensions = dense<[1,2]> : tensor<2xi64>
  } : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  return %result : tensor<10x11x12x13xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10,11,12,13] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10,11,12,13] reverse(f32[10,11,12,13] %[[ARG0]]), dimensions={1,2}

// -----

// CHECK:  HloModule
func @main(%mu: tensor<f32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = xla_hlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = "xla_hlo.rng_normal"(%mu, %sigma, %shape) : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[MU:.*]] = f32[] parameter(0)
// CHECK:  %[[SIGMA:.*]] = f32[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2,3,5] rng(f32[] %[[MU]], f32[] %[[SIGMA]]), distribution=rng_normal

// -----

// CHECK:  HloModule
func @main() -> tensor<2x3x5xf32> {
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = xla_hlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = xla_hlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %3 = "xla_hlo.rng_uniform"(%0, %1, %2) : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  return %3 : tensor<2x3x5xf32>
}

// CHECK:  ENTRY
// CHECK-DAG:  %[[A:.*]] = f32[] constant(0)
// CHECK-DAG:  %[[B:.*]] = f32[] constant(1)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2,3,5] rng(f32[] %[[A]], f32[] %[[B]]), distribution=rng_uniform

// -----

// CHECK:  HloModule
func @main(%input_tensor: tensor<200x100x300xf32>, %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "xla_hlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>): // no predecessors
    %add = xla_hlo.add %lhs, %rhs : tensor<f32>
    "xla_hlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = {
      update_window_dims = dense<[1]> : tensor<1xi64>,
      inserted_window_dims = dense<[0, 1]> : tensor<2xi64>,
      scatter_dims_to_operand_dims = dense<[0, 1]> : tensor<2xi64>,
      index_vector_dim = 1 : i64
    },
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  return %0 : tensor<200x100x300xf32>
}

// CHECK:  [[COMPUTATION:%.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[200,100,300] parameter(0)
// CHECK:  [[VAL_2:%.*]] = s32[10,2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[10,300] parameter(2)
// CHECK:  ROOT
// CHECK-SAME:  f32[200,100,300] scatter(f32[200,100,300] [[VAL_1]], s32[10,2] [[VAL_2]], f32[10,300] [[VAL_3]]), update_window_dims={1}, inserted_window_dims={0,1}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, indices_are_sorted=true, unique_indices=true, to_apply=[[COMPUTATION]]

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK:  %[[ARG0:.*]] = pred[] parameter(0)
  // CHECK:  %[[COND:.*]] = pred[2,3] broadcast(pred[] %[[ARG0]]), dimensions={}
  // CHECK:  %[[ARG1:.*]] = s32[2,3] parameter(1)
  // CHECK:  %[[ARG2:.*]] = s32[2,3] parameter(2)

  // CHECK:  ROOT %[[RES:.*]] = s32[2,3] select(pred[2,3] %[[COND]], s32[2,3] %[[ARG1]], s32[2,3] %[[ARG2]])
  %0 = "xla_hlo.select"(%arg0, %arg1, %arg2) {name = "select.4"} : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "xla_hlo.select_and_scatter"(%arg0, %arg1, %0) ( {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):	// no predecessors
    %2 = "xla_hlo.compare"(%arg3, %arg4) {comparison_direction = "GE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):	// no predecessors
    %2 = xla_hlo.add %arg3, %arg4 : tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}

// CHECK:  %[[SELECT_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> pred[] {
// CHECK:  ROOT %[[RESULT:.*]] = pred[] compare(f32[] %[[ARG0]], f32[] %[[ARG1]]), direction=GE

// CHECK:  %[[SCATTER_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> f32[] {
// CHECK:  ROOT %[[RESULT:.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10,24,24,64] parameter(0)
// CHECK:  %[[ARG1:.*]] = f32[10,12,12,64] parameter(1)
// CHECK:  %[[INIT:.*]] = f32[] constant(0)

// CHECK:  ROOT %[[RESULT:.*]] = f32[10,24,24,64]
// CHECK-SAME:  select-and-scatter(f32[10,24,24,64] %[[ARG0]], f32[10,12,12,64] %[[ARG1]], f32[] %[[INIT]]),
// CHECK-SAME:  window={size=1x2x2x1 stride=1x2x2x1},
// CHECK-SAME:  select=%[[SELECT_COMPUTATION]], scatter=%[[SCATTER_COMPUTATION]]

// -----

// CHECK:  HloModule
func @main(%arg: tensor<3x4xi32>, %token: !xla_hlo.token) -> !xla_hlo.token {
  %0 = "xla_hlo.send"(%arg, %token) {
    channel_id = {
      handle = 5 : i64,
      type = 2 : i64  // Device to host channel
    },
    is_host_transfer = true
  } : (tensor<3x4xi32>, !xla_hlo.token) -> !xla_hlo.token
  return %0 : !xla_hlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  [[SEND:%.*]] = (s32[3,4], u32[], token[]) send(s32[3,4] [[ARG]], token[] [[TOKEN]]), channel_id=5, is_host_transfer=true
// CHECK:  ROOT
// CHECK-SAME:  token[] send-done((s32[3,4], u32[], token[]) [[SEND]]), channel_id=5, is_host_transfer=true

// -----

// CHECK:  HloModule
func @main(%arg: tensor<3x4xi32>, %token: !xla_hlo.token) -> !xla_hlo.token {
  %0 = "xla_hlo.send"(%arg, %token) {
    channel_id = {
      handle = 5 : i64,
      type = 1 : i64  // Device to device channel
    },
    is_host_transfer = false
  } : (tensor<3x4xi32>, !xla_hlo.token) -> !xla_hlo.token
  return %0 : !xla_hlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  [[SEND:%.*]] = (s32[3,4], u32[], token[]) send(s32[3,4] [[ARG]], token[] [[TOKEN]]), channel_id=5
// CHECK:  ROOT
// CHECK-SAME:  token[] send-done((s32[3,4], u32[], token[]) [[SEND]]), channel_id=5

// -----

// CHECK:  HloModule
func @main(%arg: tensor<4x4xf32>, %size: tensor<i32>) -> tensor<4x4xf32> {
  %0 = "xla_hlo.set_dimension_size"(%arg, %size) {dimension = 1 : i32} : (tensor<4x4xf32>, tensor<i32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,4] parameter(0)
// CHECK:  [[SIZE:%.*]] = s32[] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[4,<=4] set-dimension-size(f32[4,4] [[ARG]], s32[] [[SIZE]]), dimensions={1}

// -----

// CHECK:  HloModule
func @main(%arg: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "xla_hlo.slice"(%arg) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  ROOT
// CHECK-SAME:  s32[1,2] slice(s32[3,4] [[ARG]]), slice={[1:2:1], [0:4:2]}

// -----

// CHECK:  HloModule
func @main(%arg: tensor<3x4xi32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "xla_hlo.dynamic-slice"(%arg, %start1, %start2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  return %0 : tensor<1x4xi32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = s32[3,4] parameter(0)
// CHECK:  %[[ARG1:.*]] = s64[] parameter(1)
// CHECK:  %[[ARG2:.*]] = s64[] parameter(2)
// CHECK:  ROOT
// CHECK-SAME:  s32[1,4] dynamic-slice(s32[3,4] %[[ARG]], s64[] %[[ARG1]], s64[] %[[ARG2]]), dynamic_slice_sizes={1,4}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  "xla_hlo.trace"(%arg0) {tag = "This is a random test"} : (tensor<2xi32>) -> ()
  return %arg0: tensor<2xi32>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = s32[2] parameter(0)
// CHECK:  () trace(s32[2] [[VAL_1]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // CHECK:  [[ARG:%.*]] = s32[1,2,3,4] parameter(0)

  // CHECK-NEXT:  ROOT %transpose.2 = s32[2,1,4,3] transpose(s32[1,2,3,4] [[ARG]]), dimensions={1,0,3,2}
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  return %0 : tensor<2x1x4x3xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %0 = "xla_hlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// CHECK:  [[ARG_A:%.*]] = f32[4,4] parameter(0)
// CHECK:  [[ARG_B:%.*]] = f32[4,3] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[4,3] triangular-solve(f32[4,4] [[ARG_A]], f32[4,3] [[ARG_B]]), left_side=true, lower=true, unit_diagonal=true, transpose_a=NO_TRANSPOSE

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<f32>, %arg1 : tensor<i32>) -> tuple<tensor<f32>, tensor<i32>> {
  %result = "xla_hlo.tuple"(%arg0, %arg1) {} : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
  return %result : tuple<tensor<f32>, tensor<i32>>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[] parameter(0)
// CHECK:  %[[ARG1:.*]] = s32[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = (f32[], s32[]) tuple(f32[] %[[ARG0]], s32[] %[[ARG1]])

// -----

// CHECK:  HloModule
func @main(%arg_f32: tensor<4xf32>, %arg_i32: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK:  [[ARG_F32:%.*]] = f32[4] parameter(0)
  // CHECK:  [[EXPM1:%.*]] = f32[4] exponential-minus-one(f32[4] [[ARG_F32]])
  %expm1 = "xla_hlo.exponential_minus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK:  [[LOG1P:%.*]] = f32[4] log-plus-one(f32[4] [[ARG_F32]])
  %log1p = "xla_hlo.log_plus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK:  [[ARG_I32:%.*]] = s32[4] parameter(1)
  // CHECK:  [[NOT:%.*]] = s32[4] not(s32[4] [[ARG_I32]])
  %not = "xla_hlo.not"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK:  [[POPCNT:%.*]] = s32[4] popcnt(s32[4] [[ARG_I32]])
  %popcnt = "xla_hlo.popcnt"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>

  return %expm1, %log1p, %not, %popcnt : tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK:  [[VAL_1:%.*]] = pred[4] parameter(0)
  // CHECK:  [[VAL_2:%.*]] = pred[4] parameter(1)
  %0 = xla_hlo.xor %arg0, %arg1 : tensor<4xi1>
  // CHECK:  ROOT [[VAL_3:%.*]] = pred[4] xor(pred[4] [[VAL_1]], pred[4] [[VAL_2]])
  return %0 : tensor<4xi1>
}

// -----

// CHECK:  HloModule
func @main(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0 = "xla_hlo.sort"(%input0, %input1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "xla_hlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> tuple<tensor<16x16xf32>, tensor<16x16xi32>>
  return
}

// CHECK: %[[SORT_CMP:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[], {{.*}}: s32[], {{.*}}: s32[]) -> pred[] {
// CHECK:   ROOT %compare.8 = pred[] compare(f32[] %[[ARG0]], f32[] %[[ARG1]]), direction=GT

// CHECK: ENTRY %{{.*}} ([[MAIN_ARG0:.*]]: f32[16,16], [[MAIN_ARG1:.*]]: s32[16,16]) -> (f32[16,16], s32[16,16]) {
// CHECK:   ROOT %{{.*}} = (f32[16,16], s32[16,16]) sort(f32[16,16] %[[MAIN_ARG0]], s32[16,16] %[[MAIN_ARG1]]), dimensions={1}, is_stable=true, to_apply=%[[SORT_CMP]]


// -----

// CHECK:  HloModule
func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = "xla_hlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", xla_hlo.sharding = "type: OTHER\ntile_assignment_dimensions: 1\ntile_assignment_dimensions: 2\ntile_assignment_devices: 0\ntile_assignment_devices: 1"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[16,16] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[16,16] custom-call(f32[16,16] %[[ARG0]]), custom_call_target="Sharding", sharding={devices=[1,2]0,1}

// -----

// Tests that the exported HLO module keeps parameter replication annotation.

// CHECK:  HloModule
func @main(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32> {tf_device.is_same_data_across_replicas = true}) -> tensor<16x16xf32> {
  %0 = "xla_hlo.add"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[16,16] parameter(0)
// CHECK-NOT: parameter_replication={true}
// CHECK:  %[[ARG1:.*]] = f32[16,16] parameter(1), parameter_replication={true}
// CHECK:  ROOT %[[RESULT:.*]] = f32[16,16] add(f32[16,16] %[[ARG0]], f32[16,16] %[[ARG1]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f64>>) -> (tensor<2xf32>, tensor<2xf64>) {
  %0 = "xla_hlo.abs"(%arg0) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %1 = "xla_hlo.abs"(%arg1) : (tensor<2xcomplex<f64>>) -> (tensor<2xf64>)
  return %0, %1 : tensor<2xf32>, tensor<2xf64>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = c64[2] parameter(0)
// CHECK:  %[[ABS0:.*]] = f32[2] abs(c64[2] %[[ARG0]])
// CHECK:  %[[ARG1:.*]] = c128[2] parameter(1)
// CHECK:  %[[ABS1:.*]] = f64[2] abs(c128[2] %[[ARG1]])
// CHECK:  ROOT %[[RESULT:.*]] = (f32[2], f64[2]) tuple(f32[2] %[[ABS0]], f64[2] %[[ABS1]])

// -----

// CHECK:  HloModule
func @main(%arg0: tensor<4xui8>) -> (tensor<4xui8>) {
  %0 = "xla_hlo.not"(%arg0) : (tensor<4xui8>) -> tensor<4xui8>
  return %0 : tensor<4xui8>
}

// CHECK: ENTRY
// CHECK: %[[ARG0:.*]] = u8[4] parameter(0)
//  ROOT %[[RESULT:.*]] = u8[4] not(u8[4] %[[ARG0]])
