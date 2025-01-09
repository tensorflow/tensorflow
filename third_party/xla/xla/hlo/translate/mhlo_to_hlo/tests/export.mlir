// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s
// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics --via-builder=true %s | FileCheck %s

// CHECK: HloModule foo
// CHECK: ENTRY %main
module @foo {
  func.func @main(%arg: tensor<i1>) -> tensor<i1> {
    func.return %arg : tensor<i1>
  }
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2xi1>) -> tensor<2xi1> {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = pred[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = pred[2] xor(pred[2] %[[ARG]], pred[2] %[[ARG]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: !mhlo.token, %arg1: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0, %arg1) : (!mhlo.token, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = token[] parameter(0)
// CHECK:  %[[ARG1:.*]] = token[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = token[] after-all(token[] %[[ARG0]], token[] %[[ARG1]])

// -----


// CHECK:  HloModule
func.func @main() -> !mhlo.token {
  %0 = "mhlo.after_all"() : () -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ROOT [[TOKEN:%.*]] = token[] after-all()

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "mhlo.reduce_scatter"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    scatter_dimension = 0 : i64
  } : (tensor<10xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[5] reduce-scatter(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2},{1,3}}
// CHECK-SAME: dimensions={0}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    shard_count = 4,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x32xf32>) -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[128,32] parameter(0)
// CHECK: ROOT %[[OUTPUT:.*]] = f32[128,128] all-gather(f32[128,32] %[[INPUT]])
// CHECK-SAME: channel_id=1
// CHECK-SAME{LITERAL}: replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME: dimensions={1}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    shard_count = 4,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<128x32xf32>) -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[128,32] parameter(0)
// CHECK: ROOT %[[OUTPUT:.*]] = f32[128,128] all-gather(f32[128,32] %[[INPUT]])
// CHECK-SAME: channel_id=1
// CHECK-SAME{LITERAL}: replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME: dimensions={1}
// CHECK-SAME: use_global_device_ids=true

// -----

func.func private @main(%arg0: tensor<8x2xf32>, %arg1: tensor<8x4xf32>) -> tuple<tensor<8x8xf32>, tensor<8x16xf32>> {
  // CHECK:      %[[ARG0:.*]] = f32[8,2] parameter(0)
  // CHECK-NEXT: %[[ARG1:.*]] = f32[8,4] parameter(1)
  // CHECK-NEXT: %[[TUPLE:.*]] = (f32[8,2], f32[8,4]) tuple
  // CHECK-NEXT: %[[TUPLE_ARG0:.*]] = f32[8,2] get-tuple-element((f32[8,2], f32[8,4]) %[[TUPLE]]), index=0
  // CHECK-NEXT: %[[TUPLE_ARG1:.*]] = f32[8,4] get-tuple-element((f32[8,2], f32[8,4]) %[[TUPLE]]), index=1
  // CHECK-NEXT: (f32[8,8], f32[8,16]) all-gather(f32[8,2] %[[TUPLE_ARG0]], f32[8,4] %[[TUPLE_ARG1]]), channel_id=1, replica_groups={{.*}}, dimensions={1}
  %0:2 = "mhlo.all_gather"(%arg0, %arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<8x2xf32>, tensor<8x4xf32>) -> (tensor<8x8xf32>, tensor<8x16xf32>)
  %1 = mhlo.tuple %0#0, %0#1 {xla_shape = "(f32[8,8]{0,1}, f32[8,16]{0,1})"} : tuple<tensor<8x8xf32>, tensor<8x16xf32>>
  return %1 : tuple<tensor<8x8xf32>, tensor<8x16xf32>>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----
// Test non-uniform sized replica groups.

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, -1], [1, 3, 5, 6]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2,4},{1,3,5,6}}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
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

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME:  use_global_device_ids=true
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----

func.func private @main(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tuple<tensor<8xf32>, tensor<f32>> {
  // CHECK:      %[[ARG0:.*]] = f32[8] parameter(0)
  // CHECK-NEXT: %[[ARG1:.*]] = f32[] parameter(1)
  // CHECK-NEXT: %[[TUPLE:.*]] = (f32[8], f32[]) tuple
  // CHECK-NEXT: %[[TUPLE_ARG0:.*]] = f32[8] get-tuple-element((f32[8], f32[]) %[[TUPLE]]), index=0
  // CHECK-NEXT: %[[TUPLE_ARG1:.*]] = f32[] get-tuple-element((f32[8], f32[]) %[[TUPLE]]), index=1
  // CHECK-NEXT: (f32[8], f32[]) all-reduce(f32[8] %[[TUPLE_ARG0]], f32[] %[[TUPLE_ARG1]]), replica_groups={}, to_apply={{.*}}
  %0:2 = "mhlo.all_reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %2 = mhlo.add %arg2, %arg3 : tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {replica_groups = dense<> : tensor<0x0xi64>} : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  %1 = mhlo.tuple %0#0, %0#1 {xla_shape = "(f32[8]{0}, f32[])"} : tuple<tensor<8xf32>, tensor<f32>>
  return %1 : tuple<tensor<8xf32>, tensor<f32>>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "mhlo.reduce_scatter"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    scatter_dimension = 0 : i64
  } : (tensor<10xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[5] reduce-scatter(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2},{1,3}}
// CHECK-SAME:  dimensions={0}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<5xf32> {
  %0 = "mhlo.reduce_scatter"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    scatter_dimension = 0 : i64,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

// CHECK:  %[[COMPUTATION:.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[5] reduce-scatter(f32[10] %[[ARG0]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2},{1,3}}
// CHECK-SAME:  use_global_device_ids=true
// CHECK-SAME:  dimensions={0}
// CHECK-SAME:  to_apply=%[[COMPUTATION]]

// -----

// CHECK:  HloModule
func.func @main(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %1 = "mhlo.tuple"(%0#0, %0#1, %0#2) : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  func.return %1 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,2,2,2] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[2] parameter(2)
// CHECK:  [[VAL_4:%.*]] = f32[2] parameter(3)
// CHECK:  [[VAL_5:%.*]] = f32[2,2,2,2] parameter(4)
// CHECK:  [[BNG:%.*]] = (f32[2,2,2,2], f32[2], f32[2]) batch-norm-grad(f32[2,2,2,2] [[VAL_1]], f32[2] [[VAL_2]], f32[2] [[VAL_3]], f32[2] [[VAL_4]], f32[2,2,2,2] [[VAL_5]]), epsilon=0.001, feature_index=0
// CHECK:  [[GTE0:%.*]] = f32[2,2,2,2] get-tuple-element((f32[2,2,2,2], f32[2], f32[2]) [[BNG]]), index=0
// CHECK:  [[GTE1:%.*]] = f32[2] get-tuple-element((f32[2,2,2,2], f32[2], f32[2]) [[BNG]]), index=1
// CHECK:  [[GTE2:%.*]] = f32[2] get-tuple-element((f32[2,2,2,2], f32[2], f32[2]) [[BNG]]), index=2
// CHECK:  ROOT
// CHECK-SAME: [[RES:%.*]] = (f32[2,2,2,2], f32[2], f32[2]) tuple(f32[2,2,2,2] [[GTE0]], f32[2] [[GTE1]], f32[2] [[GTE2]])


// -----

// CHECK:  HloModule
func.func @main(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>> {
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 3 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  %1 = "mhlo.tuple"(%0#0, %0#1, %0#2) : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
  func.return %1 : tuple<tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,2,2,2] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[2] parameter(2)
// CHECK:  [[BNT:%.*]] = (f32[2,2,2,2], f32[2], f32[2]) batch-norm-training(f32[2,2,2,2] [[VAL_1]], f32[2] [[VAL_2]], f32[2] [[VAL_3]]), epsilon=0.001, feature_index=3
// CHECK:  [[GTE0:%.*]] = f32[2,2,2,2] get-tuple-element((f32[2,2,2,2], f32[2], f32[2]) [[BNT]]), index=0
// CHECK:  [[GTE1:%.*]] = f32[2] get-tuple-element((f32[2,2,2,2], f32[2], f32[2]) [[BNT]]), index=1
// CHECK:  [[GTE2:%.*]] = f32[2] get-tuple-element((f32[2,2,2,2], f32[2], f32[2]) [[BNT]]), index=2
// CHECK:  ROOT
// CHECK-SAME: [[RES:%.*]] = (f32[2,2,2,2], f32[2], f32[2]) tuple(f32[2,2,2,2] [[GTE0]], f32[2] [[GTE1]], f32[2] [[GTE2]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xi32>, %arg3: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK:  [[VAL_1:%.*]] = f32[4] parameter(0)
  // CHECK:  [[VAL_2:%.*]] = f32[4] parameter(1)
  // CHECK:  [[ATAN2:%.*]] = f32[4] atan2(f32[4] [[VAL_1]], f32[4] [[VAL_2]])
  // CHECK:  [[VAL_3:%.*]] = s32[4] parameter(2)
  // CHECK:  [[VAL_4:%.*]] = s32[4] parameter(3)
  %0 = mhlo.atan2 %arg0, %arg1 : tensor<4xf32>

  // CHECK:  [[SHL:%.*]] = s32[4] shift-left(s32[4] [[VAL_3]], s32[4] [[VAL_4]])
  %1 = mhlo.shift_left %arg2, %arg3 : tensor<4xi32>

  // CHECK:  [[SHRA:%.*]] = s32[4] shift-right-arithmetic(s32[4] [[VAL_3]], s32[4] [[VAL_4]])
  %2 = mhlo.shift_right_arithmetic %arg2, %arg3 : tensor<4xi32>

  // CHECK:  [[SHRL:%.*]] = s32[4] shift-right-logical(s32[4] [[VAL_3]], s32[4] [[VAL_4]])
  %3 = mhlo.shift_right_logical %arg2, %arg3 : tensor<4xi32>

  // CHECK:  ROOT
  // CHECK-SAME:  [[VAL_9:%.*]] = (f32[4], s32[4], s32[4], s32[4]) tuple(f32[4] [[ATAN2]], s32[4] [[SHL]], s32[4] [[SHRA]], s32[4] [[SHRL]])
  func.return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "mhlo.bitcast_convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = s32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2] bitcast-convert(s32[2] %[[ARG]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xi32>) -> tensor<1x2x3x4xi32> {
  // CHECK:  [[ARG:%.*]] = s32[4] parameter(0)
  // CHECK-NEXT:  ROOT %broadcast.2 = s32[1,2,3,4] broadcast(s32[4] [[ARG]]), dimensions={3}
  %0 = "mhlo.broadcast"(%arg0) <{broadcast_sizes = dense<[1,2,3]> : tensor<3xi64>}> : (tensor<4xi32>) -> tensor<1x2x3x4xi32>
  func.return %0 : tensor<1x2x3x4xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %result = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<1xf32>) -> tensor<1x10xf32>
  func.return %result : tensor<1x10xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[1] parameter(0)
// CHECK:  ROOT %broadcast.2 = f32[1,10] broadcast(f32[1] [[ARG]]), dimensions={0}

// -----

// CHECK:  HloModule
func.func @main() -> !mhlo.token {
  %0 = "mhlo.create_token"() : () -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ROOT [[TOKEN:%.*]] = token[] after-all()

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  func.call @empty_callee() : () -> ()
  func.return %arg0 : tensor<4xi32>
}

func.func @empty_callee() {
  func.return
}

// CHECK:       [[CALLEE:%.*]] () -> () {
// CHECK-NEXT:    ROOT %{{.*}} = () tuple()
// CHECK-NEXT:  }

// CHECK:       ENTRY [[MAIN:%.*]] ([[ARG:.*]]: s32[4]) -> s32[4] {
// CHECK-NEXT:    ROOT %[[ARG]] = s32[4] parameter(0)
// CHECK-NEXT:    [[CALL:%.*]] = () call(), to_apply=[[CALLEE]]
// CHECK-NEXT:  }

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = func.call @callee(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = func.call @callee(%0, %0) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %1 : tensor<4xi32>
}

func.func @callee(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
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
func.func @main(%arg0: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0:2 = func.call @callee(%arg0, %arg0) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
  func.return %0#0, %0#1 : tensor<4xi32>, tensor<4xi32>
}

func.func @callee(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "mhlo.multiply"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0, %1 : tensor<4xi32>, tensor<4xi32>
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
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.collective_broadcast"(%arg0) {
    replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}
// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[128,32] parameter(0)
// CHECK:  ROOT [[RESULT:%.*]] = f32[128,32] collective-broadcast(f32[128,32] [[ARG]]), channel_id=1
// CHECK-SAME{LITERAL}:  replica_groups={{0,1},{2,3}}
// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}
// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[128,32] parameter(0)
// CHECK:  ROOT [[RESULT:%.*]] = f32[128,32] collective-permute(f32[128,32] [[ARG]]), channel_id=1, source_target_pairs={{\{\{}}0,1},{1,2},{2,3}}

// -----

// CHECK:  HloModule
func.func @main(%arg0 : tensor<5x2xf32>,
           %arg1 : tensor<5x5xf32>,
           %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  %result = "mhlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 1 : i64
  } : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  func.return %result : tensor<5x14xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[5,2] parameter(0)
// CHECK:  %[[ARG1:.*]] = f32[5,5] parameter(1)
// CHECK:  %[[ARG2:.*]] = f32[5,7] parameter(2)
// CHECK:  ROOT %[[RESULT:.*]] = f32[5,14] concatenate(f32[5,2] %[[ARG0]], f32[5,5] %[[ARG1]], f32[5,7] %[[ARG2]]), dimensions={1}

// -----

// CHECK:  HloModule
func.func @main() {
  // CHECK:  constant.{{.*}} = s64[] constant(1)
  %cst = arith.constant dense<1> : tensor<i64>
  // CHECK:  constant.{{.*}} = f32[2,2,1,1]
  // CHECK-SAME:  { { /*i0=0*/ { /*i1=0*/ {1} }, { /*i1=1*/ {2} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {4} } } }
  %cst_0 = arith.constant dense<
    [[[[1.000000e+00]], [[2.000000e+00]]], [[[3.000000e+00]], [[4.000000e+00]]]]
  > : tensor<2x2x1x1xf32>

  // CHECK:  s32[1] constant({1})
  %cst_1 = arith.constant dense<1> : tensor<1xi32>

  // CHECK:  %[[C:.*]] = s32[] constant(1)
  // CHECK:  s32[10] broadcast(s32[] %[[C]])
  %cst_2 = arith.constant dense<1> : tensor<10xi32>

  // CHECK:  s32[4] constant({1, 2, 3, 4})
  %cst_3 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  // CHECK:  s32[2,2] constant({ { 1, 2 }, { 3, 4 } })
  %cst_4 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>

  // CHECK:  s32[2,2] constant({ { 3, 2 }, { 1, 4 } })
  %cst_5 = arith.constant dense<[[3, 2], [1, 4]]> : tensor<2x2xi32>

  // CHECK:  u32[2,2] constant({ { 1, 2 }, { 4, 8 } })
  %cst_6 = arith.constant dense<[[1, 2], [4, 8]]> : tensor<2x2xui32>

  // CHECK: bf16[4] constant({1, 2, 3, 4})
  %cst_7 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xbf16>

  // CHECK: f16[4] constant({1, -4, -65504, 0.015625}
  %cst_8 = arith.constant dense<[1.0e+00, -4.0e+00, -65504.0e+00, 1.5625e-02]> : tensor<4xf16>

  // CHECK: c64[] constant((1, 0))
  %cst_9 = arith.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>

  // CHECK: c128[] constant((1, 0))
  %cst_10 = arith.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>

  // CHECK: f8e5m2[4] constant({1, 2, 3, 4})
  %cst_11 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E5M2>

  // CHECK: f8e4m3fn[4] constant({1, 2, 3, 4})
  %cst_12 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E4M3FN>

  // CHECK: f8e4m3b11fnuz[4] constant({1, 2, 3, 4})
  %cst_13 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E4M3B11FNUZ>

  // CHECK: f8e4m3fnuz[4] constant({1, 2, 3, 4})
  %cst_14 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E4M3FNUZ>

  // CHECK: f8e5m2fnuz[4] constant({1, 2, 3, 4})
  %cst_15 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E5M2FNUZ>

  // CHECK: f8e4m3[4] constant({1, 2, 3, 4})
  %cst_16 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E4M3>

  // CHECK: f8e3m4[4] constant({1, 2, 3, 4})
  %cst_17 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf8E3M4>

  func.return
}

// -----

// CHECK:  HloModule
func.func @main(%arg0 : tensor<100x26x26x32xf32>, %arg1 : tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32> {
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<3x3x1x32xf32>) -> tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[100,26,26,32] parameter(0)
// CHECK:  %[[ARG1:.*]] = f32[3,3,1,32] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = f32[100,28,28,1] convolution(f32[100,26,26,32] %[[ARG0]], f32[3,3,1,32] %[[ARG1]]),
// CHECK-SAME:  window={size=3x3 pad=2_2x2_2},
// CHECK-SAME:  dim_labels=b01f_01oi->b01f

// -----

// Test convolution i8xi8 -> i32.
// CHECK:  HloModule
func.func @main(%arg0 : tensor<100x26x26x32xi8>, %arg1 : tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  func.return %result : tensor<100x28x28x1xi32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = s8[100,26,26,32] parameter(0)
// CHECK:  %[[ARG1:.*]] = s8[3,3,1,32] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = s32[100,28,28,1] convolution(s8[100,26,26,32] %[[ARG0]], s8[3,3,1,32] %[[ARG1]]),
// CHECK-SAME:  window={size=3x3 pad=2_2x2_2},
// CHECK-SAME:  dim_labels=b01f_01oi->b01f

// -----

// Test convolution with window reversal.
// CHECK:  HloModule
func.func @main(%arg0 : tensor<100x26x26x32xi8>, %arg1 : tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32> {
  %result = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>,
    window_reversal = dense<1> : tensor<2xi1>
  } : (tensor<100x26x26x32xi8>, tensor<3x3x1x32xi8>) -> tensor<100x28x28x1xi32>
  func.return %result : tensor<100x28x28x1xi32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = s8[100,26,26,32] parameter(0)
// CHECK:  %[[ARG1:.*]] = s8[3,3,1,32] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = s32[100,28,28,1] convolution(s8[100,26,26,32] %[[ARG0]], s8[3,3,1,32] %[[ARG1]]),
// CHECK-SAME:  window={size=3x3 pad=2_2x2_2 rhs_reversal=1x1},
// CHECK-SAME:  dim_labels=b01f_01oi->b01f

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "mhlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = s32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2] convert(s32[2] %[[ARG]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "mhlo.convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf8E5M2>
  %1 = "mhlo.convert"(%0) : (tensor<2xf8E5M2>) -> tensor<2xf32>
  %2 = "mhlo.convert"(%1) : (tensor<2xf32>) -> tensor<2xf8E4M3FN>
  %3 = "mhlo.convert"(%2) : (tensor<2xf8E4M3FN>) -> tensor<2xf32>
  %4 = "mhlo.convert"(%3) : (tensor<2xf32>) -> tensor<2xf8E4M3FNUZ>
  %5 = "mhlo.convert"(%4) : (tensor<2xf8E4M3FNUZ>) -> tensor<2xf32>
  %6 = "mhlo.convert"(%5) : (tensor<2xf32>) -> tensor<2xf8E5M2FNUZ>
  %7 = "mhlo.convert"(%6) : (tensor<2xf8E5M2FNUZ>) -> tensor<2xf32>
  %8 = "mhlo.convert"(%7) : (tensor<2xf32>) -> tensor<2xf8E4M3>
  %9 = "mhlo.convert"(%8) : (tensor<2xf8E4M3>) -> tensor<2xf32>
  %10 = "mhlo.convert"(%9) : (tensor<2xf32>) -> tensor<2xf8E3M4>
  %11 = "mhlo.convert"(%10) : (tensor<2xf8E3M4>) -> tensor<2xf32>
  func.return %11 : tensor<2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = f32[2] parameter(0)
// CHECK:  %[[E5M2_VAL:.*]] = f8e5m2[2] convert(f32[2] %[[ARG]])
// CHECK:  %[[F32_VAL:.*]] = f32[2] convert(f8e5m2[2] %[[E5M2_VAL]])
// CHECK:  %[[E4M3FN_VAL:.*]] = f8e4m3fn[2] convert(f32[2] %[[F32_VAL]])
// CHECK:  %[[F32_VAL2:.*]] = f32[2] convert(f8e4m3fn[2] %[[E4M3FN_VAL]])
// CHECK:  %[[E4M3FNUZ_VAL:.*]] = f8e4m3fnuz[2] convert(f32[2] %[[F32_VAL2]])
// CHECK:  %[[F32_VAL3:.*]] = f32[2] convert(f8e4m3fnuz[2] %[[E4M3FNUZ_VAL]])
// CHECK:  %[[E5M2FNUZ_VAL:.*]] = f8e5m2fnuz[2] convert(f32[2] %[[F32_VAL3]])
// CHECK:  %[[F32_VAL4:.*]] = f32[2] convert(f8e5m2fnuz[2] %[[E5M2FNUZ_VAL]])
// CHECK:  %[[E4M3_VAL:.*]] = f8e4m3[2] convert(f32[2] %[[F32_VAL4]])
// CHECK:  %[[F32_VAL5:.*]] = f32[2] convert(f8e4m3[2] %[[E4M3_VAL]])
// CHECK:  %[[E3M4_VAL:.*]] = f8e3m4[2] convert(f32[2] %[[F32_VAL5]])
// CHECK:  ROOT %[[F32_VAL6:.*]] = f32[2] convert(f8e3m4[2] %[[E3M4_VAL]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xui32>) -> tensor<5x5xi8> {
  %result = "mhlo.stochastic_convert"(%arg0, %arg1) : (tensor<5x5xf32>, tensor<5x5xui32>) -> tensor<5x5xi8>
  func.return %result : tensor<5x5xi8>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[5,5] parameter(0)
// CHECK:  %[[ARG1:.*]] = u32[5,5] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = s8[5,5] stochastic-convert(f32[5,5] %[[ARG0]], u32[5,5] %[[ARG1]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "mhlo.copy"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = s32[2] copy(s32[2] [[ARG]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = mhlo.constant dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi32>
  %1 = "mhlo.cross-replica-sum"(%arg0) {replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %1 : tensor<10xf32>
}

// CHECK:  %[[SUM_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> f32[]
// CHECK:  ROOT %[[RESULT:.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10] all-reduce(f32[10] %[[ARG0]])
// CHECK-SAME{LITERAL}:  replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME:  to_apply=%[[SUM_COMPUTATION]]

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "SetBound", mhlo.literal = dense<1> : tensor<i32>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,3] parameter(0)
// CHECK:  ROOT
// CHECK-SAME:  f32[2,3] custom-call(f32[2,3] [[VAL_1]])
// CHECK-SAME:  custom_call_target="SetBound"
// CHECK-SAME:  literal=s32[] 1

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>, %arg2: tensor<3xi32>, %arg3: tensor<3xi32>, %arg4: tensor<3xi32>, %arg5: tensor<3xi32>) -> (tensor<6xf32>) {
  %0 = mhlo.custom_call @ragged_all_to_all(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {api_version = 4 : i32, backend_config = {replica_groups = dense<[[0, 1, 2]]> : tensor<1x3xi64>}} : (tensor<6xf32>, tensor<6xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK: ENTRY
// CHECK: [[ARG_0:%.*]] = f32[6] parameter(0)
// CHECK: [[ARG_1:%.*]] = f32[6] parameter(1)
// CHECK: [[ARG_2:%.*]] = s32[3] parameter(2)
// CHECK: [[ARG_3:%.*]] = s32[3] parameter(3)
// CHECK: [[ARG_4:%.*]] = s32[3] parameter(4)
// CHECK: [[ARG_5:%.*]] = s32[3] parameter(5)
// CHECK: ROOT
// CHECK-SAME: f32[6] ragged-all-to-all(f32[6] [[ARG_0]], f32[6] [[ARG_1]], s32[3] [[ARG_2]], s32[3] [[ARG_3]], s32[3] [[ARG_4]], /*index=5*/s32[3] [[ARG_5]])
// CHECK-SAME{LITERAL}: replica_groups={{0,1,2}}

// -----

// CHECK:  HloModule
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 4 : i64,
      is_fallback = true
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// CHECK: ENTRY
// CHECK-DAG:   [[ARG0:%.*]] = bf16[16,256] parameter(0)
// CHECK-DAG:   [[ARG1:%.*]] = s32[] parameter(1)
// CHECK-DAG:   [[ARG2:%.*]] = s32[16,256] parameter(2)
// CHECK-DAG:   [[ARG3:%.*]] = bf16[] parameter(3)
// CHECK-DAG:   [[VAL0:%.*]] = (bf16[16,256], s32[16,256]) sort(bf16[16,256] [[ARG0]], s32[16,256] [[ARG2]])
// CHECK-DAG:   [[VAL1:%.*]] = s32[16,256] get-tuple-element((bf16[16,256], s32[16,256]) [[VAL0]])
// CHECK-DAG:   [[VAL2:%.*]] = s32[16,4] slice(s32[16,256] [[VAL1]])

// -----

// CHECK:  HloModule
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// CHECK: %top_k_gt_comparator.[[COMPARATOR:[0-9]+]]
// CHECK:   s32[] parameter(2)
// CHECK:   s32[] parameter(3)
// CHECK:   [[ARG0:%.*]] = bf16[] parameter(0)
// CHECK:   [[ARG1:%.*]] = bf16[] parameter(1)
// CHECK:   ROOT [[VAL:%.*]] = pred[] compare(bf16[] [[ARG0]], bf16[] [[ARG1]]), direction=GT

// CHECK: ENTRY
// CHECK-DAG:   [[ARG0:%.*]] = bf16[16,256] parameter(0)
// CHECK-DAG:   [[ARG1:%.*]] = s32[] parameter(1)
// CHECK-DAG:   [[ARG2:%.*]] = s32[16,256] parameter(2)
// CHECK-DAG:   [[ARG3:%.*]] = bf16[] parameter(3)
// CHECK-DAG:   (bf16[16,128], s32[16,128]) custom-call(bf16[16,256] [[ARG0]], s32[16,256] [[ARG2]], bf16[] [[ARG3]], s32[] [[ARG1]]),
// CHECK-SAME: custom_call_target="PartialReduce", called_computations={%top_k_gt_comparator.[[COMPARATOR]]}
// CHECK-SAME: backend_config={"log2_reduction": 1, "reduction_dim": 1, "to_apply_type": "comparator", "top_k": 4, "recall_target": 0.949218}


// -----

// expected-error@-3 {{ApproxTopK aggregates to k=4, but got 5}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x5xbf16>, tensor<16x5xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x5xbf16>, tensor<16x5xi32>)
  return %4#0, %4#1 : tensor<16x5xbf16>, tensor<16x5xi32>
}

// -----

// expected-error@-3 {{input shape mismatch at position 1}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<17x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
     api_version = 4 : i32,
     called_computations = [@top_k_gt_comparator],
     backend_config = {
       aggregate_to_topk = true,
       recall_target = 9.492180e-01 : f32,
       reduction_dim = 1 : i64,
       reduction_input_size_override = -1 : i64,
      is_fallback = false,
       top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<17x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{input and init_value element type mismatch at position 1}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i64>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i64>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}


// -----

// expected-error@-3 {{called_computation type does not match the expected type. Got '(tensor<bf16>, tensor<bf16>, tensor<i32>, tensor<i32>) -> tensor<i32>' expected '(tensor<bf16>, tensor<bf16>, tensor<i32>, tensor<i32>) -> tensor<i1>'}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  return %0 : tensor<i32>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{result shape mismatch at position 1, index 0}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<17x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<17x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<17x4xi32>
}

// -----

// expected-error@-3 {{result element type mismatch at position 1}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi64>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi64>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi64>
}

// -----

// expected-error@-3 {{num_results does not match num_inputs}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>) {
  %4 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>)
  return %4: tensor<16x4xbf16>
}

// -----

// expected-error@-3 {{ApproxTopK takes an even number of operands.}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{invalid_attribute is not a supported attribute for ApproxTopK}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      },
      invalid_attribute = 123 : i64
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{invalid_attribute is not a supported backend_config attribute for ApproxTopK}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64,
      invalid_attribute = 123 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}


// -----

// expected-error@-3 {{ApproxTopK takes exactly 1 called_computation.}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator,
    @top_k_gt_comparator], backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{Missing backend_config attribute}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator]
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{Missing top_k attribute in backend_config}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      is_fallback = false,
      reduction_input_size_override = -1 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{Missing reduction_dim attribute in backend_config}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      is_fallback = false,
      top_k = 4 : i64,
      reduction_input_size_override = -1 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{Missing reduction_input_size_override attribute in backend_config}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{Missing aggregate_to_topk attribute in backend_config}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      is_fallback = false,
      top_k = 4 : i64,
      reduction_input_size_override = -1 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{top_k attribute in backend_config must be of i64 type}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i32
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{reduction_dim attribute in backend_config must be of i64 type}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i32,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{reduction_input_size_override attribute in backend_config must be of i64 type}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i32,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{Missing recall_target attribute in backend_config}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      reduction_dim = 1 : i64,
      is_fallback = false,
      top_k = 4 : i64,
      reduction_input_size_override = -1 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}


// -----

// expected-error@-3 {{recall_target attribute in backend_config must be of f32 type}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.01 : bf16,
      reduction_dim = 1 : i64,
      is_fallback = false,
      top_k = 4 : i64,
      reduction_input_size_override = -1 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{aggregate_to_topk attribute in backend_config must be of bool type}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = 3 : i32,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{is_fallback attribute in backend_config must be of bool type}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 9.492180e-01 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = 3 : i64,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{recall_target out of range}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 1.1 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// expected-error@-3 {{reduction_dim out of range}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x256xbf16>, tensor<16x256xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.5 : f32,
      reduction_dim = 400 : i64,
      reduction_input_size_override = -1 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x256xbf16>, tensor<16x256xi32>)
  return %4#0, %4#1 : tensor<16x256xbf16>, tensor<16x256xi32>
}

// -----

// expected-error@-3 {{reduction_input_size_override out of range}}
func.func @top_k_gt_comparator(%arg0: tensor<bf16>, %arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  %0 = mhlo.compare  GT, %arg0, %arg1 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
  return %0 : tensor<i1>
}
func.func public @main(%arg0: tensor<16x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<16x256xi32>, %arg3: tensor<bf16>) -> (tensor<16x4xbf16>, tensor<16x4xi32>) {
  %4:2 = mhlo.custom_call @ApproxTopK(%arg0, %arg2, %arg3, %arg1) {
    api_version = 4 : i32,
    called_computations = [@top_k_gt_comparator],
    backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.5 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = 3 : i64,
      is_fallback = false,
      top_k = 4 : i64
      }
    } : (tensor<16x256xbf16>, tensor<16x256xi32>, tensor<bf16>, tensor<i32>) -> (tensor<16x4xbf16>, tensor<16x4xi32>)
  return %4#0, %4#1 : tensor<16x4xbf16>, tensor<16x4xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {backend_config = "bar", call_target_name = "foo", custom_call_schedule = #mhlo<custom_call_schedule LATEST>, has_side_effect = true} : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,3] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[5,5] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[1,2,3] custom-call(f32[2,3] [[VAL_1]], f32[5,5] [[VAL_2]])
// CHECK-SAME:  custom_call_target="foo"
// CHECK-SAME:  custom_call_has_side_effect=true
// CHECK-SAME:  schedule=SCHEDULE_LATEST
// CHECK-SAME:  backend_config="bar"

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {backend_config = "bar", call_target_name = "foo", custom_call_schedule = #mhlo<custom_call_schedule EARLIEST>, has_side_effect = true} : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,3] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[5,5] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[1,2,3] custom-call(f32[2,3] [[VAL_1]], f32[5,5] [[VAL_2]])
// CHECK-SAME:  custom_call_target="foo"
// CHECK-SAME:  custom_call_has_side_effect=true
// CHECK-SAME:  schedule=SCHEDULE_EARLIEST
// CHECK-SAME:  backend_config="bar"

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>) -> tuple<tensor<2x3xf32>> {
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<2x3xf32>) -> tuple<tensor<2x3xf32>>
  func.return %0 : tuple<tensor<2x3xf32>>
}

// CHECK:  ENTRY
// CHECK:  [[ARG0:%.*]] = f32[2,3] parameter(0)
// CHECK:  ROOT
// CHECK-SAME:  (f32[2,3]) custom-call(f32[2,3] [[ARG0]])
// CHECK-SAME:  custom_call_target="foo"

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>) -> tuple<tensor<2x3xf32>, tensor<4x5xf16>> {
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<2x3xf32>) -> tuple<tensor<2x3xf32>, tensor<4x5xf16>>
  func.return %0 : tuple<tensor<2x3xf32>, tensor<4x5xf16>>
}

// CHECK:  ENTRY
// CHECK:  [[ARG0:%.*]] = f32[2,3] parameter(0)
// CHECK:  ROOT
// CHECK-SAME:  (f32[2,3], f16[4,5]) custom-call(f32[2,3] [[ARG0]])
// CHECK-SAME:  custom_call_target="foo"

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<4x5xf16>) {
  %0:2 = "mhlo.custom_call"(%arg0) {call_target_name = "foo"} : (tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<4x5xf16>)
  func.return %0#0, %0#1 : tensor<2x3xf32>, tensor<4x5xf16>
}

// CHECK:  ENTRY
// CHECK:  [[ARG0:%.*]] = f32[2,3] parameter(0)
// CHECK:  [[OUTS:%.*]] = (f32[2,3], f16[4,5]) custom-call(f32[2,3] [[ARG0]])
// CHECK-SAME:  custom_call_target="foo"
// CHECK-DAG:  [[OUT0:%.*]] = f32[2,3] get-tuple-element((f32[2,3], f16[4,5]) [[OUTS]]), index=0
// CHECK-DAG:  [[OUT1:%.*]] = f16[4,5] get-tuple-element((f32[2,3], f16[4,5]) [[OUTS]]), index=1
// CHECK:  ROOT
// CHECK-SAME: (f32[2,3], f16[4,5]) tuple(f32[2,3] [[OUT0]], f16[4,5] [[OUT1]])

// -----

// Test dot i8xi8 -> i64

func.func @main(%arg0: tensor<3xi8>, %arg1: tensor<3xi8>) -> tensor<i64> {
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3xi8>, tensor<3xi8>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// CHECK: ENTRY
// CHECK-SAME: ([[ARG0:.*]]: s8[3], [[ARG1:.*]]: s8[3]) -> s64[] {
// CHECK: %[[ARG0]] = s8[3] parameter(0)
// CHECK: %[[ARG1]] = s8[3] parameter(1)
// CHECK: ROOT
// CHECK-SAME: s64[] dot(s8[3] %[[ARG0]], s8[3] %[[ARG1]]),

// -----

// Test dot i4xi4 -> i8

func.func @main(%arg0: tensor<3xi4>, %arg1: tensor<3xi4>) -> tensor<i8> {
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3xi4>, tensor<3xi4>) -> tensor<i8>
  func.return %0 : tensor<i8>
}

// CHECK:  ENTRY
// CHECK:  [[CALLEE_1:%.*]] ([[ARG_1:.*]]: s4[3], [[ARG_2:.*]]: s4[3]) -> s8[]
// CHECK:  %[[ARG_1:.*]] = s4[3] parameter(0)
// CHECK:  %[[ARG_2:.*]] = s4[3] parameter(1)
// CHECK:  ROOT %[[DOT:.*]] = s8[] dot(s4[3] %[[ARG_1:.*]], s4[3] %[[ARG_2:.*]])

// -----

// Test dot ui4xui4 -> ui8

func.func @main(%arg0: tensor<3xui4>, %arg1: tensor<3xui4>) -> tensor<ui8> {
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3xui4>, tensor<3xui4>) -> tensor<ui8>
  func.return %0 : tensor<ui8>
}

// CHECK:  ENTRY
// CHECK:  [[CALLEE_1:%.*]] ([[ARG_1:.*]]: u4[3], [[ARG_2:.*]]: u4[3]) -> u8[]
// CHECK:  %[[ARG_1:.*]] = u4[3] parameter(0)
// CHECK:  %[[ARG_2:.*]] = u4[3] parameter(1)
// CHECK:  ROOT %[[DOT:.*]] = u8[] dot(u4[3] %[[ARG_1:.*]], u4[3] %[[ARG_2:.*]])

// -----

// Test dot i8xi8 -> i32.
// CHECK:  HloModule
func.func @main(%arg0: tensor<2x2x2xi8>, %arg1: tensor<2x2x3xi8>) -> tensor<2x2x3xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = []} : (tensor<2x2x2xi8>, tensor<2x2x3xi8>) -> tensor<2x2x3xi32>
  func.return %0 : tensor<2x2x3xi32>
}

// CHECK: ENTRY
// CHECK-SAME: ([[ARG0:.*]]: s8[2,2,2], [[ARG1:.*]]: s8[2,2,3]) -> s32[2,2,3] {
// CHECK: %[[ARG0]] = s8[2,2,2] parameter(0)
// CHECK: %[[ARG1]] = s8[2,2,3] parameter(1)
// CHECK: ROOT
// CHECK-SAME: s32[2,2,3] dot(s8[2,2,2] %[[ARG0]], s8[2,2,3] %[[ARG1]]),
// CHECK-SAME: lhs_batch_dims={0}
// CHECK-SAME: lhs_contracting_dims={2}
// CHECK-SAME: rhs_batch_dims={0}
// CHECK-SAME: rhs_contracting_dims={1}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10x16xbf16>, %arg1: tensor<32x20xbf16>, %meta: tensor<10x2xui16>) -> tensor<10x20xf32> {
  // CHECK:  dot(bf16[10,16] %{{.*}}, bf16[32,20] %{{.*}}, u16[10,2] %{{.*}}), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
  %0 = "mhlo.sparse_dot"(%arg0, %arg1, %meta) {
    lhs_sparsity = #mhlo.sparsity<dimension=1, n=2, m=4>,
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
    precision_config = []} : (tensor<10x16xbf16>, tensor<32x20xbf16>, tensor<10x2xui16>) -> tensor<10x20xf32>
  func.return %0 : tensor<10x20xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<3x4xi32>, %arg1: tensor<4x5xi32>) -> tensor<3x5xi32> {
  // Simple einsum is lowered to HLO dot op.
  // CHECK:  dot(s32[3,4] %{{.*}}, s32[4,5] %{{.*}}), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %0 = "mhlo.einsum"(%arg0, %arg1) <{einsum_config = "ab,bc->ac"}> : (tensor<3x4xi32>, tensor<4x5xi32>) -> tensor<3x5xi32>
  func.return %0 : tensor<3x5xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) <{fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT>}> : (tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[3,9] parameter(0)
// CHECK:  c64[3,5] fft(f32[3,9] [[ARG]]), fft_type=RFFT, fft_length={9}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>) -> tensor<10x300xf32> {
  // CHECK:  [[ARG0:%.*]] = f32[200,100,300] parameter(0)
  // CHECK:  [[ARG1:%.*]] = s32[10,2] parameter(1)
  // CHECK:  f32[10,300] gather(f32[200,100,300] [[ARG0]], s32[10,2] [[ARG1]])
  // CHECK-SAME:  offset_dims={1}
  // CHECK-SAME:  collapsed_slice_dims={0,1}
  // CHECK-SAME:  start_index_map={0,1}
  // CHECK-SAME:  index_vector_dim=1
  // CHECK-SAME:  slice_sizes={1,1,300}
  // CHECK-SAME:  indices_are_sorted=true
  %0 = "mhlo.gather"(%arg0, %arg1) <{
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [1],
      start_index_map = [0,1],
    >,
    indices_are_sorted = true,
    slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>
  }> : (tensor<200x100x300xf32>, tensor<10x2xi32>) -> tensor<10x300xf32>
  func.return %0 : tensor<10x300xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<100x200x1xi32>) -> tensor<100x200x300xf32> {
  // CHECK:  [[ARG0:%.*]] = f32[200,100,300] parameter(0)
  // CHECK:  [[ARG1:%.*]] = s32[100,200,1] parameter(1)
  // CHECK:  f32[100,200,300] gather(f32[200,100,300] [[ARG0]], s32[100,200,1] [[ARG1]])
  // CHECK-SAME:  offset_dims={2}
  // CHECK-SAME:  collapsed_slice_dims={}
  // CHECK-SAME:  start_index_map={2}
  // CHECK-SAME:  operand_batching_dims={0,1}
  // CHECK-SAME:  start_indices_batching_dims={1,0}
  // CHECK-SAME:  index_vector_dim=2
  // CHECK-SAME:  slice_sizes={1,1,300}
  // CHECK-SAME:  indices_are_sorted=true
  %0 = "mhlo.gather"(%arg0, %arg1) <{
    dimension_numbers = #mhlo.gather<
      operand_batching_dims = [0, 1],
      start_indices_batching_dims = [1, 0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [2],
    >,
    indices_are_sorted = true,
    slice_sizes = dense<[1, 1, 300]> : tensor<3xi64>
  }> : (tensor<200x100x300xf32>, tensor<100x200x1xi32>) -> tensor<100x200x300xf32>
  func.return %0 : tensor<100x200x300xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<4x2xf32>, %size: tensor<i32>) -> tensor<i32> {
  %0 = "mhlo.set_dimension_size"(%arg, %size) <{dimension = 1 : i64}> : (tensor<4x2xf32>, tensor<i32>) -> tensor<4x2xf32>
  %1 = "mhlo.get_dimension_size"(%0) <{dimension = 1 : i64}> : (tensor<4x2xf32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,2] parameter(0)
// CHECK:  [[SIZE:%.*]] = s32[] parameter(1)
// CHECK:  [[DYNAMIC:%.*]] = f32[4,<=2] set-dimension-size(f32[4,2] [[ARG]], s32[] [[SIZE]]), dimensions={1}
// CHECK:  ROOT %[[RESULT:.*]] = s32[] get-dimension-size(f32[4,<=2] [[DYNAMIC]]), dimensions={1}


// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<?x4xf32, #mhlo.type_extensions<bounds = [8, ?]>>) -> tensor<8x4xf32> {
  %size = mhlo.constant dense<8> : tensor<i32>
  %1 = "mhlo.set_dimension_size"(%arg, %size) <{dimension = 0 : i64}> : (tensor<?x4xf32, #mhlo.type_extensions<bounds = [8, ?]>>, tensor<i32>) -> tensor<8x4xf32>
  func.return %1 : tensor<8x4xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[<=8,4] parameter(0)
// CHECK:  [[SIZE:%.*]] = s32[] constant(8)
// CHECK:  ROOT [[DYNAMIC:%.*]] = f32[8,4] set-dimension-size(f32[<=8,4] [[ARG]], s32[] [[SIZE]]), dimensions={0}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "mhlo.get_tuple_element"(%arg0) <{index = 0 : i32}> : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = (f32[], s32[]) parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[] get-tuple-element((f32[], s32[]) %[[ARG0]]), index=0

// -----

// CHECK:  HloModule
func.func @main(%arg0: !mhlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token> {
  %0:3 = "mhlo.infeed"(%arg0) <{infeed_config = "foobar", layout=[[0, 1], [0]]}> : (!mhlo.token) -> (tensor<3x3xi32>, tensor<i1>, !mhlo.token)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<3x3xi32>, tensor<i1>) -> tuple<tensor<3x3xi32>, tensor<i1>>
  %2 = "mhlo.tuple"(%1, %0#2) : (tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token) -> tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token>

  func.return %2 : tuple<tuple<tensor<3x3xi32>, tensor<i1>>, !mhlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((s32[3,3], pred[]), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK:  [[GTE1:%.*]] = (s32[3,3], pred[]) get-tuple-element(((s32[3,3], pred[]), token[]) [[INFEED]]), index=0
// CHECK:  [[GTE2:%.*]] = s32[3,3] get-tuple-element((s32[3,3], pred[]) [[GTE1]]), index=0
// CHECK:  [[GTE3:%.*]] = pred[] get-tuple-element((s32[3,3], pred[]) [[GTE1]]), index=1
// CHECK:  [[GTE4:%.*]] = token[] get-tuple-element(((s32[3,3], pred[]), token[]) [[INFEED]]), index=1

// -----

// CHECK:  HloModule
func.func @main(%arg0: !mhlo.token) -> tensor<3x3xi32> {
  %0:2 = "mhlo.infeed"(%arg0) <{infeed_config = "foobar", layout=[[0,1]]}> : (!mhlo.token) -> (tensor<3x3xi32>, !mhlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((s32[3,3]), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK:  [[GTE0:%.*]] = (s32[3,3]) get-tuple-element(((s32[3,3]), token[]) [[INFEED]]), index=0
// CHECK:  ROOT [[GTE1:%.*]] = s32[3,3] get-tuple-element((s32[3,3]) [[GTE0]]), index=0
// CHECK:  [[GTE2:%.*]] = token[] get-tuple-element(((s32[3,3]), token[]) [[INFEED]]), index=1

// -----

// CHECK:  HloModule

func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.infeed"(%arg0) <{infeed_config = "foobar", layout = [], xla_shape = "((), token[])"}> : (!mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = token[] parameter(0)
// CHECK:  [[INFEED:%.*]] = ((), token[]) infeed(token[] [[ARG]]), infeed_config="foobar"
// CHECK:   ROOT [[TOKEN:%.*]] = token[] get-tuple-element(((), token[]) [[INFEED]]), index=1

// -----

// CHECK:  HloModule
func.func @main() -> tensor<1x10xf32> {
  %result = "mhlo.iota"() {
    iota_dimension = 1 : i64
  } : () -> tensor<1x10xf32>
  func.return %result : tensor<1x10xf32>
}

// CHECK:  ENTRY
// CHECK:  ROOT %[[RESULT:.*]] = f32[1,10] iota(), iota_dimension=1

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
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
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xi32>) -> tensor<4xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>):
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK:  [[COMPUTATION:%.*]] ({{.*}}: f32[], {{.*}}: s32[]) -> f32[] {
// CHECK:  ROOT [[ARG_0:%.*]] = f32[] parameter(0)
// CHECK:  }

// CHECK:  ENTRY
// CHECK:  [[ARG_2:%.*]] = f32[4] parameter(0)
// CHECK:  [[ARG_3:%.*]] = s32[4] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[4] map(f32[4] [[ARG_2]], s32[4] [[ARG_3]]), dimensions={0}, to_apply=[[COMPUTATION]]

// -----


// CHECK:  HloModule
func.func @main(%data: tensor<3xi32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.outfeed"(%data, %token) {outfeed_config = "foobar"} : (tensor<3xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[DATA:%.*]] = s32[3] parameter(0)
// CHECK-DAG: [[DATATUPLE:%.*]] = (s32[3]) tuple(s32[3] [[DATA]])
// CHECK-DAG:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = token[] outfeed((s32[3]) [[DATATUPLE]], token[] [[TOKEN]]), outfeed_shape=(s32[3]{0}), outfeed_config="foobar"

// -----

// The following op sharding is used:
// Proto debug string:
//   type: TUPLE
//   tuple_shardings {
//     type: OTHER
//     tile_assignment_dimensions: 2
//     tile_assignment_dimensions: 1
//     tile_assignment_devices: 0
//     tile_assignment_devices: 1
//   }
// Serialized string:
//   "\08\03\1A\02\02\01\22\02\00\01"

// CHECK:  HloModule
func.func @main(%data: tensor<3x2xi32>, %token: !mhlo.token) -> !mhlo.token {
  %shard = "mhlo.custom_call"(%data) {api_version = 1 : i32, backend_config = "", call_target_name = "Sharding",  mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<3x2xi32>) -> tensor<3x2xi32>
  %full_shaped_data = "mhlo.custom_call"(%shard) {api_version = 1 : i32, backend_config = "", call_target_name = "SPMDShardToFullShape",  mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<3x2xi32>) -> tensor<6x2xi32>
  %0 = "mhlo.outfeed"(%full_shaped_data, %token) {mhlo.sharding = "\08\02*\0A\08\03\1A\02\02\01\22\02\00\01*\08\08\01\1A\01\01\22\01\00", outfeed_config = "foobar"} : (tensor<6x2xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[DATA:%.*]] = s32[3,2] parameter(0)
// CHECK:  [[SHARD:%.*]] = s32[3,2] custom-call(s32[3,2] [[DATA]])
// CHECK-SAME: custom_call_target="Sharding"
// CHECK-SAME: sharding={devices=[1,2]0,1}
// CHECK:  [[FULL:%.*]] = s32[6,2] custom-call(s32[3,2] [[SHARD]])
// CHECK-SAME: custom_call_target="SPMDShardToFullShape"
// CHECK-SAME: sharding={devices=[1,2]0,1}
// CHECK-DAG: [[DATATUPLE:%.*]] = (s32[6,2]) tuple(s32[6,2] [[FULL]])
// CHECK-DAG:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = token[] outfeed((s32[6,2]) [[DATATUPLE]], token[] [[TOKEN]]), outfeed_shape=(s32[6,2]{1,0}), outfeed_config="foobar",
// CHECK-SAME: sharding={
// CHECK-SAME: {devices=[2,1]0,1}, {maximal device=0}
// CHECK-SAME: }

// -----

// CHECK:  HloModule
func.func @main(%data1: tensor<3xi32>, %data2: tensor<3xi32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.outfeed"(%data1, %data2,  %token) {outfeed_config = "foobar"} : (tensor<3xi32>, tensor<3xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[DATA1:%.*]] = s32[3] parameter(0)
// CHECK:  [[DATA2:%.*]] = s32[3] parameter(1)
// CHECK-DAG:  [[TUPLE:%.*]] = (s32[3], s32[3]) tuple(s32[3] [[DATA1]], s32[3] [[DATA2]])
// CHECK-DAG:  [[TOKEN:%.*]] = token[] parameter(2)
// CHECK:  ROOT %[[RESULT:.*]] = token[] outfeed((s32[3], s32[3]) [[TUPLE]], token[] [[TOKEN]]), outfeed_shape=(s32[3]{0}, s32[3]{0}), outfeed_config="foobar"

// -----

// CHECK:  HloModule
func.func @main(%token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.outfeed"(%token) {outfeed_config = "foobar"} : (!mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK: ENTRY
// CHECK-DAG:   [[EMPTY_TUPLE:%.*]] = () tuple()
// CHECK-DAG:   [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:   ROOT [[RESULT:%.*]] = token[] outfeed(() [[EMPTY_TUPLE]], token[] [[TOKEN]]), outfeed_shape=(), outfeed_config="foobar"

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<4x6xf32>, %pad: tensor<f32>) -> tensor<13x19xf32> {
  %0 = "mhlo.pad"(%arg, %pad) <{edge_padding_high = dense<[4,5]> : tensor<2xi64>, edge_padding_low = dense<[2,3]> : tensor<2xi64>, interior_padding = dense<1> : tensor<2xi64>}> : (tensor<4x6xf32>, tensor<f32>) -> tensor<13x19xf32>
  func.return %0 : tensor<13x19xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,6] parameter(0)
// CHECK:  [[PADDING_VAL:%.*]] = f32[] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[13,19] pad(f32[4,6] [[ARG]], f32[] [[PADDING_VAL]]), padding=2_4_1x3_5_1

// -----

// CHECK:  HloModule
func.func @main(%token: !mhlo.token) -> tuple<tensor<3x4xi32>, !mhlo.token> {
  %0:2 = "mhlo.recv"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 3  // Host to device channel
    >,
    is_host_transfer = true
  } : (!mhlo.token) -> (tensor<3x4xi32>, !mhlo.token)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, !mhlo.token) -> tuple<tensor<3x4xi32>, !mhlo.token>
  func.return %1 : tuple<tensor<3x4xi32>, !mhlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[RECV:%.*]] = (s32[3,4], u32[], token[]) recv(token[] [[TOKEN]]), channel_id=5, is_host_transfer=true
// CHECK:  (s32[3,4], token[]) recv-done((s32[3,4], u32[], token[]) [[RECV]]), channel_id=5, is_host_transfer=true

// -----

// CHECK:  HloModule
func.func @main(%token: !mhlo.token) -> tuple<tensor<3x4xi32>, !mhlo.token> {
  %0:2 = "mhlo.recv"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (!mhlo.token) -> (tensor<3x4xi32>, !mhlo.token)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, !mhlo.token) -> tuple<tensor<3x4xi32>, !mhlo.token>
  func.return %1 : tuple<tensor<3x4xi32>, !mhlo.token>
}

// CHECK:  ENTRY
// CHECK:  [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:  [[RECV:%.*]] = (s32[3,4], u32[], token[]) recv(token[] [[TOKEN]]), channel_id=5
// CHECK:  (s32[3,4], token[]) recv-done((s32[3,4], u32[], token[]) [[RECV]]), channel_id=5


// -----

// CHECK:  HloModule
func.func @main(%token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.recv"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (!mhlo.token) -> (!mhlo.token)
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK-NEXT:  [[ARG:%.*]] = token[] parameter(0)
// CHECK-NEXT:  [[RECV:%.*]] = ((), u32[], token[]) recv(token[] [[ARG]]), channel_id=5
// CHECK-NEXT:  [[RECV_DONE:%.*]] = ((), token[]) recv-done(((), u32[], token[]) [[RECV]]), channel_id=5
// CHECK-NEXT:  [[DATA:%.*]] =   () get-tuple-element(((), token[]) [[RECV_DONE]]), index=0
// CHECK-NEXT:  ROOT [[TOKEN:%.*]] =   token[] get-tuple-element(((), token[]) [[RECV_DONE]]), index=1

// -----

// CHECK:  HloModule
func.func @main(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1x10xi32>, %arg2 : tensor<f32>, %arg3 : tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %result0, %result1 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({
    ^bb0(%fa: tensor<f32>, %ia : tensor<i32>, %fb: tensor<f32>, %ib: tensor<i32>):
      %fmax = "mhlo.maximum"(%fa, %fb) {} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %imax = "mhlo.maximum"(%ia, %ib) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "mhlo.return"(%fmax, %imax) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  func.return %result0, %result1 : tensor<1xf32>, tensor<1xi32>
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
func.func @main(%arg0: tensor<2x17x31x7xi32>) -> tensor<2x5x8x7xi32> {
  %0 = mhlo.constant dense<-2147483648> : tensor<i32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = mhlo.maximum %arg1, %arg2 : tensor<i32>
    "mhlo.return"(%2) : (tensor<i32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>,
    base_dilations = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<2x17x31x7xi32>, tensor<i32>) -> tensor<2x5x8x7xi32>
  func.return %1 : tensor<2x5x8x7xi32>
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
func.func @main(%arg0: tensor<2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[2] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[1,2] reshape(f32[2] %[[ARG0]])

// -----

// CHECK:  HloModule
func.func @main(%arg0 : tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32> {
  %result = "mhlo.reverse"(%arg0) {
    dimensions = dense<[1,2]> : tensor<2xi64>
  } : (tensor<10x11x12x13xf32>) -> tensor<10x11x12x13xf32>
  func.return %result : tensor<10x11x12x13xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[10,11,12,13] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[10,11,12,13] reverse(f32[10,11,12,13] %[[ARG0]]), dimensions={1,2}

// -----

// CHECK:  HloModule
func.func @main(%mu: tensor<f32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = "mhlo.rng"(%mu, %sigma, %shape) <{rng_distribution = #mhlo.rng_distribution<NORMAL>}> : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[MU:.*]] = f32[] parameter(0)
// CHECK:  %[[SIGMA:.*]] = f32[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2,3,5] rng(f32[] %[[MU]], f32[] %[[SIGMA]]), distribution=rng_normal

// -----

// CHECK:  HloModule
func.func @main() -> tensor<2x3x5xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %2 = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %3 = "mhlo.rng"(%0, %1, %2) <{rng_distribution = #mhlo.rng_distribution<UNIFORM>}> : (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %3 : tensor<2x3x5xf32>
}

// CHECK:  ENTRY
// CHECK-DAG:  %[[A:.*]] = f32[] constant(0)
// CHECK-DAG:  %[[B:.*]] = f32[] constant(1)
// CHECK:  ROOT %[[RESULT:.*]] = f32[2,3,5] rng(f32[] %[[A]], f32[] %[[B]]), distribution=rng_uniform

// -----

// CHECK:  HloModule
func.func @main(%input_tensor: tensor<200x100x300xf32>, %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
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
func.func @main(%input_tensor: tensor<200x100x300xf32>, %scatter_indices: tensor<100x200x1xi32>, %updates: tensor<100x200x300xf32>) -> tensor<200x100x300xf32> {
  %0 = "mhlo.scatter" (%input_tensor, %scatter_indices, %updates) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [2],
      input_batching_dims = [0, 1],
      scatter_indices_batching_dims = [1, 0],
      scatter_dims_to_operand_dims = [2],
      index_vector_dim = 2
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<100x200x1xi32>, tensor<100x200x300xf32>) -> tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// CHECK:  [[COMPUTATION:%.*]] ({{.*}}: f32[], {{.*}}: f32[]) -> f32[]
// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[200,100,300] parameter(0)
// CHECK:  [[VAL_2:%.*]] = s32[100,200,1] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[100,200,300] parameter(2)
// CHECK:  ROOT
// CHECK-SAME:  f32[200,100,300] scatter(f32[200,100,300] [[VAL_1]], s32[100,200,1] [[VAL_2]], f32[100,200,300] [[VAL_3]]), update_window_dims={2}, inserted_window_dims={}, scatter_dims_to_operand_dims={2}, input_batching_dims={0,1}, scatter_indices_batching_dims={1,0}, index_vector_dim=2, indices_are_sorted=true, unique_indices=true, to_apply=[[COMPUTATION]]

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi64>, %arg2: tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>) {
    %0:2 = "mhlo.scatter"(%arg0, %arg0, %arg1, %arg2, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
      %2 = mhlo.add %arg3, %arg4 : tensor<f32>
      %3 = mhlo.add %arg5, %arg6 : tensor<f32>
      "mhlo.return"(%2, %3) : (tensor<f32>, tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<200x100x300xf32>, tensor<200x100x300xf32>, tensor<10x2xi64>, tensor<10x300xf32>, tensor<10x300xf32>) -> (tensor<200x100x300xf32>, tensor<200x100x300xf32>)
    return %0#0, %0#1 : tensor<200x100x300xf32>, tensor<200x100x300xf32>
  }

// CHECK:  [[COMPUTATION:%.*]] ({{.*}}: f32[], {{.*}}: f32[], {{.*}}: f32[], {{.*}}: f32[]) -> (f32[], f32[])
// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[200,100,300] parameter(0)
// CHECK:  [[VAL_2:%.*]] = s64[10,2] parameter(1)
// CHECK:  [[VAL_3:%.*]] = f32[10,300] parameter(2)
// CHECK: (f32[200,100,300], f32[200,100,300]) scatter(f32[200,100,300] [[VAL_1]], f32[200,100,300] [[VAL_1]], s64[10,2] [[VAL_2]], f32[10,300] [[VAL_3]], f32[10,300] [[VAL_3]]), update_window_dims={1}, inserted_window_dims={0,1}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=[[COMPUTATION]]

// -----


// CHECK:  HloModule
func.func @main(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK:  %[[ARG0:.*]] = pred[] parameter(0)
  // CHECK:  %[[COND:.*]] = pred[2,3] broadcast(pred[] %[[ARG0]]), dimensions={}
  // CHECK:  %[[ARG1:.*]] = s32[2,3] parameter(1)
  // CHECK:  %[[ARG2:.*]] = s32[2,3] parameter(2)

  // CHECK:  ROOT %[[RES:.*]] = s32[2,3] select(pred[2,3] %[[COND]], s32[2,3] %[[ARG1]], s32[2,3] %[[ARG2]])
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = "mhlo.compare"(%arg3, %arg4) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %1 : tensor<10x24x24x64xf32>
}

// CHECK:  %[[SELECT_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> pred[] {
// CHECK:  ROOT %[[RESULT:.*]] = pred[] compare(f32[] %[[ARG0]], f32[] %[[ARG1]]), direction=GE, type=TOTALORDER

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
func.func @main(%arg: tensor<3x4xi32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.send"(%arg, %token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2  // Device to host channel
    >,
    is_host_transfer = true
  } : (tensor<3x4xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  [[SEND:%.*]] = (s32[3,4], u32[], token[]) send(s32[3,4] [[ARG]], token[] [[TOKEN]]), channel_id=5, is_host_transfer=true
// CHECK:  ROOT
// CHECK-SAME:  token[] send-done((s32[3,4], u32[], token[]) [[SEND]]), channel_id=5, is_host_transfer=true

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xi32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.send"(%arg, %token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 1  // Device to device channel
    >,
    is_host_transfer = false
  } : (tensor<3x4xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  [[TOKEN:%.*]] = token[] parameter(1)
// CHECK:  [[SEND:%.*]] = (s32[3,4], u32[], token[]) send(s32[3,4] [[ARG]], token[] [[TOKEN]]), channel_id=5
// CHECK:  ROOT
// CHECK-SAME:  token[] send-done((s32[3,4], u32[], token[]) [[SEND]]), channel_id=5

// -----

// CHECK:  HloModule
func.func @main(%token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.send"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 1
    >,
    is_host_transfer = false
  } : (!mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK: ENTRY
// CHECK-DAG:   [[ARG:%.*]] = () tuple()
// CHECK-DAG:   [[TOKEN:%.*]] = token[] parameter(0)
// CHECK:   [[SEND:%.*]] = ((), u32[], token[]) send(() [[ARG]], token[] [[TOKEN]]), channel_id=5
// CHECK:  ROOT
// CHECK-SAME:   token[] send-done(((), u32[], token[]) [[SEND]]), channel_id=5

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<4x4xf32>, %size: tensor<i32>) -> tensor<4x4xf32> {
  %0 = "mhlo.set_dimension_size"(%arg, %size) {dimension = 1 : i64} : (tensor<4x4xf32>, tensor<i32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = f32[4,4] parameter(0)
// CHECK:  [[SIZE:%.*]] = s32[] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[4,<=4] set-dimension-size(f32[4,4] [[ARG]], s32[] [[SIZE]]), dimensions={1}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "mhlo.slice"(%arg) <{start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>}> : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// CHECK:  ENTRY
// CHECK:  [[ARG:%.*]] = s32[3,4] parameter(0)
// CHECK:  ROOT
// CHECK-SAME:  s32[1,2] slice(s32[3,4] [[ARG]]), slice={[1:2:1], [0:4:2]}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xi32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "mhlo.dynamic_slice"(%arg, %start1, %start2) <{slice_sizes = dense<[1, 4]> : tensor<2xi64>}> : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG:.*]] = s32[3,4] parameter(0)
// CHECK:  %[[ARG1:.*]] = s64[] parameter(1)
// CHECK:  %[[ARG2:.*]] = s64[] parameter(2)
// CHECK:  ROOT
// CHECK-SAME:  s32[1,4] dynamic-slice(s32[3,4] %[[ARG]], s64[] %[[ARG1]], s64[] %[[ARG2]]), dynamic_slice_sizes={1,4}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // CHECK:  [[ARG:%.*]] = s32[1,2,3,4] parameter(0)

  // CHECK-NEXT:  ROOT %transpose.2 = s32[2,1,4,3] transpose(s32[1,2,3,4] [[ARG]]), dimensions={1,0,3,2}
  %0 = "mhlo.transpose"(%arg0) <{permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}> : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0 : tensor<2x1x4x3xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// CHECK:  [[ARG_A:%.*]] = f32[4,4] parameter(0)
// CHECK:  [[ARG_B:%.*]] = f32[4,3] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[4,3] triangular-solve(f32[4,4] [[ARG_A]], f32[4,3] [[ARG_B]]), left_side=true, lower=true, unit_diagonal=true, transpose_a=NO_TRANSPOSE

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<f32>, %arg1 : tensor<i32>) -> tuple<tensor<f32>, tensor<i32>> {
  %result = "mhlo.tuple"(%arg0, %arg1) {} : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
  func.return %result : tuple<tensor<f32>, tensor<i32>>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[] parameter(0)
// CHECK:  %[[ARG1:.*]] = s32[] parameter(1)
// CHECK:  ROOT %[[RESULT:.*]] = (f32[], s32[]) tuple(f32[] %[[ARG0]], s32[] %[[ARG1]])

// -----

// CHECK:  HloModule
func.func @main(%arg_f32: tensor<4xf32>, %arg_i32: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>) {
  // CHECK:  [[ARG_F32:%.*]] = f32[4] parameter(0)
  // CHECK:  [[EXPM1:%.*]] = f32[4] exponential-minus-one(f32[4] [[ARG_F32]])
  %expm1 = "mhlo.exponential_minus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK:  [[LOG1P:%.*]] = f32[4] log-plus-one(f32[4] [[ARG_F32]])
  %log1p = "mhlo.log_plus_one"(%arg_f32) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK:  [[ARG_I32:%.*]] = s32[4] parameter(1)
  // CHECK:  [[NOT:%.*]] = s32[4] not(s32[4] [[ARG_I32]])
  %not = "mhlo.not"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>

  // CHECK:  [[POPCNT:%.*]] = s32[4] popcnt(s32[4] [[ARG_I32]])
  %popcnt = "mhlo.popcnt"(%arg_i32) : (tensor<4xi32>) -> tensor<4xi32>

  func.return %expm1, %log1p, %not, %popcnt : tensor<4xf32>, tensor<4xf32>, tensor<4xi32>, tensor<4xi32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK:  [[VAL_1:%.*]] = pred[4] parameter(0)
  // CHECK:  [[VAL_2:%.*]] = pred[4] parameter(1)
  %0 = mhlo.xor %arg0, %arg1 : tensor<4xi1>
  // CHECK:  ROOT [[VAL_3:%.*]] = pred[4] xor(pred[4] [[VAL_1]], pred[4] [[VAL_2]])
  func.return %0 : tensor<4xi1>
}

// -----

// CHECK:  HloModule
func.func @main(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// CHECK: %[[SORT_CMP:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[], {{.*}}: s32[], {{.*}}: s32[]) -> pred[] {
// CHECK:   ROOT %compare.{{[0-9+]}} = pred[] compare(f32[] %[[ARG0]], f32[] %[[ARG1]]), direction=GT

// CHECK: [[SORT:%.+]] = (f32[16,16], s32[16,16]) sort(f32[16,16] %Arg_0.1, s32[16,16] %Arg_1.2), dimensions={1}, is_stable=true, to_apply=%[[SORT_CMP]]
// CHECK: [[GET0:%.+]] = f32[16,16] get-tuple-element((f32[16,16], s32[16,16]) [[SORT]]), index=0
// CHECK: [[GET1:%.+]] = s32[16,16] get-tuple-element((f32[16,16], s32[16,16]) [[SORT]]), index=1

// -----

// CHECK:  HloModule
func.func @main(%input0: tensor<16x16xf32>) {
  %0 = "mhlo.sort"(%input0) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
  func.return
}

// CHECK: %[[SORT_CMP:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> pred[] {
// CHECK:   ROOT %[[CMP:.*]] = pred[] compare(f32[] %[[ARG0]], f32[] %[[ARG1]]), direction=GT

// CHECK: %[[RESULT:.*]] = f32[16,16] sort(f32[16,16] %Arg_0.1), dimensions={1}, is_stable=true, to_apply=%[[SORT_CMP]]

// -----

// The following op sharding is used:
// Proto debug string:
//   type: OTHER
//   tile_assignment_dimensions: 1
//   tile_assignment_dimensions: 2
//   tile_assignment_devices: 0
//   tile_assignment_devices: 1
// Serialized string:
//   "\08\03\1A\02\01\02\22\02\00\01"

// CHECK:  HloModule
func.func @main(%arg0: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = "mhlo.custom_call"(%arg0) {backend_config = "", call_target_name = "Sharding", mhlo.sharding = "\08\03\1A\02\01\02\22\02\00\01"} : (tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[16,16] parameter(0)
// CHECK:  ROOT %[[RESULT:.*]] = f32[16,16] custom-call(f32[16,16] %[[ARG0]])
// CHECK-SAME: custom_call_target="Sharding"
// CHECK-SAME: sharding={devices=[1,2]0,1}

// -----

// CHECK:  HloModule
// CHECK: %[[FOO:.*]] ([[ARG0:.*]]: f32[2,3], [[ARG1:.*]]: f32[5,5]) -> f32[2,3]
func.func @foo (%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<2x3xf32> {
  func.return %arg0 : tensor<2x3xf32>
}

// CHECK: ENTRY
func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<2x3xf32> {
  // CHECK:  ROOT
  // CHECK-SAME:  f32[2,3] custom-call
  // CHECK-SAME:  called_computations={%[[FOO]]}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    called_computations = [@foo]
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f64>>) -> (tensor<2xf32>, tensor<2xf64>) {
  %0 = "mhlo.abs"(%arg0) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %1 = "mhlo.abs"(%arg1) : (tensor<2xcomplex<f64>>) -> (tensor<2xf64>)
  func.return %0, %1 : tensor<2xf32>, tensor<2xf64>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = c64[2] parameter(0)
// CHECK:  %[[ABS0:.*]] = f32[2] abs(c64[2] %[[ARG0]])
// CHECK:  %[[ARG1:.*]] = c128[2] parameter(1)
// CHECK:  %[[ABS1:.*]] = f64[2] abs(c128[2] %[[ARG1]])
// CHECK:  ROOT %[[RESULT:.*]] = (f32[2], f64[2]) tuple(f32[2] %[[ABS0]], f64[2] %[[ABS1]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xui8>) -> tensor<4xui8> {
  %0 = "mhlo.not"(%arg0) : (tensor<4xui8>) -> tensor<4xui8>
  func.return %0 : tensor<4xui8>
}

// CHECK: ENTRY
// CHECK: %[[ARG0:.*]] = u8[4] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = u8[4] not(u8[4] %[[ARG0]])

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4xi32>) -> tensor<*xi32> {
  %0 = "mhlo.not"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %1 = tensor.cast %0 : tensor<4xi32> to tensor<*xi32>
  func.return %1 : tensor<*xi32>
}

// CHECK: ENTRY
// CHECK: %[[ARG0:.*]] = s32[4] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = s32[4] not(s32[4] %[[ARG0]])

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<4xf32>, %size: tensor<i32>) -> tensor<?xf32> {
  %0 = "mhlo.set_dimension_size"(%arg, %size) {dimension = 0 : i64} : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>
  %1 = tensor.cast %0 : tensor<?xf32, #mhlo.type_extensions<bounds = [4]>> to tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// CHECK: ENTRY
// CHECK: %[[ARG0:.*]] = f32[4] parameter(0)
// CHECK: %[[ARG1:.*]] = s32[] parameter(1)
// CHECK: ROOT %[[RESULT:.*]] = f32[<=4] set-dimension-size

// -----


// Tests ops with different frontend attributes have such attributes set
// correctly in HloModule as frontend_attributes.

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>, %token: !mhlo.token) -> tuple<tensor<3x4xf32>, !mhlo.token> {
  %0 = "mhlo.send"(%arg, %token) {channel_handle = #mhlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true, mhlo.frontend_attributes = {_xla_host_transfer_rendezvous = "channel_dtoh_0"}} : (tensor<3x4xf32>, !mhlo.token) -> !mhlo.token
  %1:2 = "mhlo.recv"(%0) {channel_handle = #mhlo.channel_handle<handle = 2, type = 3>, is_host_transfer = true, mhlo.frontend_attributes = {_xla_host_transfer_rendezvous = "channel_htod_0"}} : (!mhlo.token) -> (tensor<3x4xf32>, !mhlo.token)
  %2 = "mhlo.tuple"(%1#0, %1#1) : (tensor<3x4xf32>, !mhlo.token) -> tuple<tensor<3x4xf32>, !mhlo.token>
  func.return %2 : tuple<tensor<3x4xf32>, !mhlo.token>
}

// CHECK:  ENTRY
// CHECK:  %[[SEND:.*]] = (f32[3,4], u32[], token[]) send
// CHECK-SAME: frontend_attributes={_xla_host_transfer_rendezvous="channel_dtoh_0"}
// CHECK:  %[[SEND_DONE:.*]] = token[] send-done((f32[3,4], u32[], token[]) %[[SEND]])
// CHECK-SAME: frontend_attributes={_xla_host_transfer_rendezvous="channel_dtoh_0"}
// CHECK:  %[[RECV:.*]] = (f32[3,4], u32[], token[]) recv(token[] %[[SEND_DONE]])
// CHECK-SAME: frontend_attributes={_xla_host_transfer_rendezvous="channel_htod_0"}
// CHECK:  %{{.*}} = (f32[3,4], token[]) recv-done((f32[3,4], u32[], token[]) %[[RECV]])
// CHECK-SAME: frontend_attributes={_xla_host_transfer_rendezvous="channel_htod_0"}

// -----

// Tests ops with empty frontend attributes do not have frontend_attributes
// populated in HloModule.

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.send"(%arg, %token) {channel_handle = #mhlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true, mhlo.frontend_attributes = {}} : (tensor<3x4xf32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK-NOT:  frontend_attributes

// -----

// Tests ops with no frontend attributes do not have frontend_attributes
// populated in HloModule.

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>, %token: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.send"(%arg, %token) {channel_handle = #mhlo.channel_handle<handle = 1, type = 2>, is_host_transfer = true} : (tensor<3x4xf32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// CHECK-NOT:  frontend_attributes

// -----

// Checks exporting rng-bit-generator.

// CHECK:  HloModule
func.func @main(%arg: tensor<3xui64>) -> tuple<tensor<3xui64>, tensor<2x2xui32>> {
// CHECK: %[[ARG0:.*]] = u64[3] parameter(0)
// CHECK: [[RNG:%.*]] = (u64[3], u32[2,2]) rng-bit-generator(u64[3] %[[ARG0]]), algorithm=rng_philox
// CHECK:  [[GTE0:%.*]] = u64[3] get-tuple-element((u64[3], u32[2,2]) [[RNG]]), index=0
// CHECK:  [[GTE1:%.*]] = u32[2,2] get-tuple-element((u64[3], u32[2,2]) [[RNG]]), index=1
// CHECK:  ROOT
// CHECK-SAME: [[RES:%.*]] = (u64[3], u32[2,2]) tuple(u64[3] [[GTE0]], u32[2,2] [[GTE1]])
  %0:2 = "mhlo.rng_bit_generator"(%arg) <{rng_algorithm = #mhlo.rng_algorithm<PHILOX>}> : (tensor<3xui64>) -> (tensor<3xui64>, tensor<2x2xui32>)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<3xui64>, tensor<2x2xui32>) -> tuple<tensor<3xui64>, tensor<2x2xui32>>
  func.return %1 : tuple<tensor<3xui64>, tensor<2x2xui32>>
}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK: %[[ARG0:.*]] = f32[3,4] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[3,4] cbrt(f32[3,4] %[[ARG0]])
  %0 = "mhlo.cbrt"(%arg) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK: %[[ARG0:.*]] = f32[3,4] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[3,4] reduce-precision(f32[3,4] %[[ARG0]]), exponent_bits=8, mantissa_bits=10
  %0 = "mhlo.reduce_precision"(%arg) {exponent_bits = 8 : i32, mantissa_bits = 10 : i32} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>) -> tensor<3x4x1xf32> {
// CHECK: %[[ARG0:.*]] = f32[3,4] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = f32[3,4,1] bitcast(f32[3,4] %[[ARG0]])
  %0 = "mhlo.bitcast"(%arg) : (tensor<3x4xf32>) -> tensor<3x4x1xf32>
  func.return %0 : tensor<3x4x1xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> (tensor<4x4xf32>, tensor<3x4xf32>) {
// CHECK: %[[ARG0:.*]] = f32[4,4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[3,4] parameter(1)
// CHECK: %[[ARGS:.*]] = (f32[4,4], f32[3,4]) tuple(f32[4,4] %[[ARG0]], f32[3,4] %[[ARG1]]), sharding={{\{}}{replicated}, {devices=[1,2]<=[2]}}
// CHECK: %[[OPT:.*]] = (f32[4,4], f32[3,4]) opt-barrier((f32[4,4], f32[3,4]) %[[ARGS]]), sharding={{\{}}{replicated}, {devices=[1,2]<=[2]}}
// CHECK: %[[GTE0:.*]] = f32[4,4] get-tuple-element((f32[4,4], f32[3,4]) %[[OPT]]), index=0, sharding={replicated}
// CHECK: %[[GTE1:.*]] = f32[3,4] get-tuple-element((f32[4,4], f32[3,4]) %[[OPT]]), index=1, sharding={devices=[1,2]<=[2]}
// CHECK: ROOT %[[ROOT:.*]] = (f32[4,4], f32[3,4]) tuple(f32[4,4] %[[GTE0]], f32[3,4] %[[GTE1]])
  %0, %1 = "mhlo.optimization_barrier"(%arg0, %arg1) {mhlo.sharding = "{{replicated}, {devices=[1,2]<=[2]}}"} : (tensor<4x4xf32>, tensor<3x4xf32>) -> (tensor<4x4xf32>, tensor<3x4xf32>)
  func.return %0, %1 : tensor<4x4xf32>, tensor<3x4xf32>
}

// -----

// CHECK:  HloModule
func.func private @main() -> tensor<ui32> {
  // CHECK: u32[] partition-id()
  %1 = "mhlo.partition_id"() : () -> tensor<ui32>
  return %1 : tensor<ui32>
}

// -----

// CHECK:  HloModule
func.func private @main(%arg0: tensor<ui32>) -> tensor<ui32> {
  // CHECK: u32[] domain(
  // CHECK-SAME: domain={kind="sharding", entry={maximal device=1}, exit={}}
  %1 = "mhlo.domain"(%arg0) {entry_metadata = "\08\01\1A\01\01\22\01\01", exit_metadata = "\08\02", kind = #mhlo<kind sharding>} : (tensor<ui32>) -> tensor<ui32>
  return %1 : tensor<ui32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK: %[[ARG0:.*]] = f32[4,4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[3,4] parameter(1)
// CHECK: ROOT %[[RESULT:.*]] = f32[3,4] triangular-solve(f32[4,4] %[[ARG0]], f32[3,4] %[[ARG1]]), lower=true, transpose_a=NO_TRANSPOSE
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = false, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = false} : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0: tensor<3x4xf32>
}

// -----

// CHECK: HloModule
// CHECK: %[[APPLYFN:.*]] ({{.*}}) -> (f32[], s32[]) {
// CHECK: %[[A0:.*]] = f32[] parameter(0)
// CHECK: %[[B0:.*]] = f32[] parameter(2)
// CHECK: %[[ADDF32:.*]] = f32[] add(f32[] %[[A0]], f32[] %[[B0]])
// CHECK: %[[A1:.*]] = s32[] parameter(1)
// CHECK: %[[B1:.*]] = s32[] parameter(3)
// CHECK: %[[ADDS32:.*]] = s32[] add(s32[] %[[A1]], s32[] %[[B1]])
// CHECK: ROOT %{{.*}} = (f32[], s32[]) tuple(f32[] %[[ADDF32]], s32[] %[[ADDS32]])

// CHECK: ENTRY
// CHECK: %[[ARG0:.*]] = f32[4,2] parameter(0)
// CHECK: %[[ARG1:.*]] = s32[4,2] parameter(1)
// CHECK: %[[ARG2:.*]] = f32[] parameter(2)
// CHECK: %[[ARG3:.*]] = s32[] parameter(3)
// CHECK: (f32[2,2], s32[2,2]) reduce-window(f32[4,2] %[[ARG0]], s32[4,2] %[[ARG1]], f32[] %[[ARG2]], s32[] %[[ARG3]])
// CHECK-SAME: window={size=5x1 stride=3x1 pad=2_2x0_0}
// CHECK-SAME: to_apply=%[[APPLYFN]]
func.func @main(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>, %init0: tensor<f32>, %init1: tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>) {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>, %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = mhlo.add %a0, %b0 : tensor<f32>
              %3 = mhlo.add %a1, %b1 : tensor<i32>
              "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = dense<[5, 1]> : tensor<2xi64>,
           window_strides = dense<[3, 1]> : tensor<2xi64> } : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) -> (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

// -----

// CHECK: HloModule
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[ARG0:.*]] = f32[2] parameter(0)
  %0 = "mhlo.round_nearest_even"(%arg0) {} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: round-nearest-even(f32[2] %[[ARG0]])
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK: HloModule
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[ARG0:.*]] = f32[2] parameter(0)
  %0 = "mhlo.tan"(%arg0) {} : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: tan(f32[2] %[[ARG0]])
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK: HloModule
func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<4x2xf32>, tensor<4x2xi32>) {
  // CHECK: %[[ARG0:.*]] = f32[4,4] parameter(0)
  %0:2 = "mhlo.topk"(%arg0) {k = 2, largest = true} : (tensor<4x4xf32>) -> (tensor<4x2xf32>, tensor<4x2xi32>)
  // CHECK: (f32[4,2], s32[4,2]) topk(f32[4,4] %[[ARG0]]), k=2, largest=true
  func.return %0#0, %0#1 : tensor<4x2xf32>, tensor<4x2xi32>
}

// -----

// CHECK: HloModule
// CHECK{LITERAL}: output_to_operand_aliasing={{0}: (0, {1})}
func.func @main(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

// CHECK: HloModule
// CHECK{LITERAL}: output_to_operand_aliasing={{}: (0, {1})}
func.func @main(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tensor<2x3xf32>
  func.return
}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>) -> tensor<3x4xf32> {
// CHECK: %[[ARG0:.*]] = f32[3,4] parameter(0)
// CHECK: %[[TOK:.*]] = token[] after-all()
// CHECK: ROOT %[[RESULT:.*]] = f32[3,4] add-dependency(f32[3,4] %[[ARG0]], token[] %[[TOK]])
  %token = "mhlo.after_all"() : () -> !mhlo.token
  %0 = "mhlo.add_dependency"(%arg, %token) : (tensor<3x4xf32>, !mhlo.token) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {execution_thread = "test_thread"} {
  %token = "mhlo.after_all"() : () -> !mhlo.token
  %0 = "mhlo.add_dependency"(%arg, %token) : (tensor<3x4xf32>, !mhlo.token) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}
// CHECK{LITERAL}: }, execution_thread="test_thread"

// -----

// CHECK:  HloModule
func.func private @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
// CHECK: %[[ARG0:.*]] = s32[2,2] parameter(0)
// CHECK: ROOT %[[RESULT:.*]] = s32[2,2] all-to-all(s32[2,2] %[[ARG0]]), channel_id=1, replica_groups={{.}}{1,2},{0,3}}, dimensions={1}
  %0 = "mhlo.all_to_all"(%arg0) {
    concat_dimension = 1 : i64,
    replica_groups = dense<[[1, 2], [0, 3]]> : tensor<2x2xi64>,
    split_count = 2 : i64, split_dimension = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 1>
  } : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

func.func private @main(%arg0: tensor<128x4xf32>, %arg1: tensor<128x4xf32>) -> tuple<tensor<128x4xf32>, tensor<128x4xf32>> {
// CHECK: %[[ARG0:.*]] = f32[128,4] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[128,4] parameter(1)
// CHECK: (f32[128,4], f32[128,4]) all-to-all(f32[128,4] %[[ARG0]], f32[128,4] %[[ARG1]]), channel_id=1, replica_groups={{.}}{0,1}}
  %0:2 = "mhlo.all_to_all"(%arg0, %arg1) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 1>
  } : (tensor<128x4xf32>, tensor<128x4xf32>) -> (tensor<128x4xf32>, tensor<128x4xf32>)
  %1 = mhlo.tuple %0#0, %0#1 {
    result_layout = [dense<[0, 1]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
    xla_shape = "(f32[128,4]{0,1}, f32[128,4]{1,0})"
  } : tuple<tensor<128x4xf32>, tensor<128x4xf32>>
  return %1 : tuple<tensor<128x4xf32>, tensor<128x4xf32>>
}

// -----

func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) -> tensor<1x2x3xf32> {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    has_side_effect = true,
    api_version = 4 : i32,
    backend_config = {
      user_attr0 = 123 : i32,
      user_attr1 = dense<42> : tensor<i32>
    }
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<1x2x3xf32>
  func.return %0 : tensor<1x2x3xf32>
}

// CHECK:  ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[2,3] parameter(0)
// CHECK:  [[VAL_2:%.*]] = f32[5,5] parameter(1)
// CHECK:  ROOT
// CHECK-SAME:  f32[1,2,3] custom-call(f32[2,3] [[VAL_1]], f32[5,5] [[VAL_2]])
// CHECK-SAME:  custom_call_target="foo"
// CHECK-SAME:  custom_call_has_side_effect=true
// CHECK-SAME:  api_version=API_VERSION_TYPED_FFI
// CHECK-SAME:  backend_config={user_attr0 = 123 : i32, user_attr1 = dense<42> : tensor<i32>}

// -----

// CHECK: ENTRY
// CHECK:  [[VAL_1:%.*]] = f32[] parameter(0), parameter_replication={true}
// CHECK:  [[VAL_2:%.*]] = (f32[2,4], (f32[2,4])) parameter(1), parameter_replication={false,true}

func.func @main(%arg0: tensor<f32> {mhlo.parameter_replication = [true]}, %arg1: tuple<tensor<2x4xf32>, tuple<tensor<2x4xf32>>> {mhlo.parameter_replication = [false, true]}) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

// -----

func.func @main(%operand: tensor<?x784xf32>) -> tensor<?x784xf32> {
  %0 = mhlo.abs %operand : tensor<?x784xf32>
  func.return %0 : tensor<?x784xf32>
}

//       CHECK: HloModule {{.*}}, entry_computation_layout={(f32[?,784]{1,0})->f32[?,784]{1,0}}
// CHECK-EMPTY:
//  CHECK-NEXT: ENTRY {{.*}} ([[ARG0:.*]]: f32[?,784]) -> f32[?,784] {
//  CHECK-NEXT:   [[ARG0]] = f32[?,784] parameter(0)
//  CHECK-NEXT:   ROOT {{.*}} = f32[?,784] abs(f32[?,784] %Arg_0.1), {{.*}}
//  CHECK-NEXT: }

// -----

// reduce multiple implicit captures test
// CHECK: HloModule
// CHECK: [[REG0:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: f32[] constant(1)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: {{.*}} reduce{{.*}} to_apply=[[REG0]]
// CHECK: ROOT
func.func @main(%arg0: tensor<2x2xf32>) -> tuple<tensor<i1>> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
   reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
    %5 = mhlo.compare  NE, %arg1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = mhlo.compare  NE, %arg2, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = mhlo.or %5, %6 : tensor<i1>
    %8 = mhlo.select %7, %0, %1 : tensor<i1>, tensor<f32>
    mhlo.return %8 : tensor<f32>
  }
  %3 = mhlo.compare  NE, %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %4 = mhlo.tuple %3 {xla_shape = "(pred[])"} : tuple<tensor<i1>>
  return %4 : tuple<tensor<i1>>
}

// -----

// all_reduce implicit capture test
// CHECK: HloModule
// CHECK: [[REG0:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: ROOT {{.*}} all-reduce{{.*}} to_apply=[[REG0]]
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %c = mhlo.constant dense<0.0> : tensor<f32>
  %0 = "mhlo.all_reduce"(%arg0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = mhlo.add %arg1, %c : tensor<f32>
    mhlo.return %1 : tensor<f32>
  }) {replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// reduce_scatter implicit capture test
// CHECK:  HloModule
// CHECK: [[REG0:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: ROOT {{.*}} reduce-scatter{{.*}} to_apply=[[REG0]]
func.func @main(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %c = mhlo.constant dense<0.0> : tensor<f32>
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %c : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// reduce_window implicit capture test
// CHECK: HloModule
// CHECK: [[REG0:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: ROOT {{.*}} reduce-window{{.*}} to_apply=[[REG0]]
func.func @main(%arg0: tensor<2x17x31x7xf32>, %arg1: tensor<f32>) -> tensor<2x16x30x7xf32> {
    %c = mhlo.constant dense<0.0> : tensor<f32>
    %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = mhlo.maximum %arg2, %c : tensor<f32>
      mhlo.return %1 : tensor<f32>
    }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x16x30x7xf32>
    return %0 : tensor<2x16x30x7xf32>
  }

// -----

// Scatter implicit capture test
// CHECK: HloModule
// CHECK: [[REG0:%region.*]] ({{.*}} {
// CHECK: s32[] constant(0)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: ROOT {{.*}} scatter{{.*}} to_apply=[[REG0]]
func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
 %c = mhlo.constant dense<0> : tensor<i32>
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %x = mhlo.add %arg4, %c : tensor<i32>
    "mhlo.return"(%x) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// select_and_scatter implicit capture test
// CHECK: HloModule
// CHECK: [[SEL_REG:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: ROOT
// CHECK: [[SCAT_REG:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: ROOT {{.*}} select-and-scatter{{.*}} select=[[SEL_REG]], scatter=[[SCAT_REG]]
func.func @main(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x23x23x64xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
    %c1 = mhlo.constant dense<0.0> : tensor<f32>
    %c2 = mhlo.constant dense<0.0> : tensor<f32>
    %0 = "mhlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = mhlo.compare  GE, %arg3, %c1,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      mhlo.return %1 : tensor<i1>
    }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = mhlo.add %arg4, %c2 : tensor<f32>
      mhlo.return %1 : tensor<f32>
    }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<10x24x24x64xf32>, tensor<10x23x23x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
    return %0 : tensor<10x24x24x64xf32>
  }

// -----

// sort implicit capture test
// CHECK: HloModule
// CHECK: [[REG0:%region.*]] ({{.*}} {
// CHECK: f32[] constant(0)
// CHECK: ROOT
// CHECK: ENTRY
// CHECK: {{.*}} sort{{.*}} to_apply=[[REG0]]
// CHECK: ROOT
func.func @main(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %c = mhlo.constant dense<0.0> : tensor<f32>
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %c) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}
