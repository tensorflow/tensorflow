// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s
// RUN: xla-translate --print-sugar=false -split-input-file -mlir-hlo-to-hlo-text --via-builder=true %s | FileCheck %s

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

// CHECK:  HloModule
func.func @all_gather_0(%arg1: tensor<128x32xf32>) -> tensor<128x128xf32> attributes {execution_thread = "main"} {
  %0 = "mhlo.all_gather"(%arg1) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    shard_count = 4,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<128x32xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

func.func @main(%arg0: tensor<128x32xf32>) -> tensor<128x128xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation = @all_gather_0, execution_thread = "main"} : (tensor<128x32xf32>) -> !mhlo.async_bundle<tensor<128x32xf32>, tensor<128x128xf32>>
  %1 = "mhlo.async_done"(%0) {called_computation = @all_gather_0, execution_thread = "main"} : (!mhlo.async_bundle<tensor<128x32xf32>, tensor<128x128xf32>>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[128,32] parameter(0)
// CHECK: %[[OUTPUT:.*]] = f32[128,128] all-gather-start(f32[128,32] %[[INPUT]])
// CHECK-SAME: channel_id=1
// CHECK-SAME{LITERAL}: replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME: dimensions={1}
// CHECK-SAME: use_global_device_ids=true
// CHECK: ROOT {{.*}} f32[128,128] all-gather-done(f32[128,128] %[[OUTPUT]]

// -----

// CHECK:  HloModule
func.func @all_reduce_0(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes {execution_thread = "main"} {
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

func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation = @all_reduce_0, execution_thread = "main"} : (tensor<10xf32>) -> !mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>
  %1 = "mhlo.async_done"(%0) {called_computation = @all_reduce_0, execution_thread = "main"} : (!mhlo.async_bundle<tensor<10xf32>, tensor<10xf32>>) -> tensor<10xf32>
  return %1 : tensor<10xf32>
}

// CHECK: ENTRY
// CHECK: %[[INPUT:.*]] = f32[10] parameter(0)
// CHECK: %[[OUTPUT:.*]] = f32[10] all-reduce-start(f32[10] %[[INPUT]])
// CHECK-SAME:  channel_id=5
// CHECK-SAME{LITERAL}:  replica_groups={{0,2,4,6},{1,3,5,7}}
// CHECK-SAME: use_global_device_ids=true
// CHECK: ROOT {{.*}} f32[10] all-reduce-done(f32[10] %[[OUTPUT]]

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
