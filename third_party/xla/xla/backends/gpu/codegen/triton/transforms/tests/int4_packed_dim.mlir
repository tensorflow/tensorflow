// RUN: xla-opt --split-input-file --int4-to-packed-int4-rewrite --canonicalize %s | FileCheck %s

// CHECK-LABEL: @major_1d
tt.func @major_1d(%arg0: !tt.ptr<i4>) -> (tensor<8x1xi8>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64

  %0 = tt.make_tensor_ptr %arg0, [%c8, %c1], [%c1, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<8x1xi4>>
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<4x1xi8>>
  %1 = tt.load %0 : !tt.ptr<tensor<8x1xi4>>
  // CHECK-DAG: %[[SHLI:.*]] = arith.shli %[[LOAD]]
  // CHECK-DAG: %[[LO:.*]] = arith.shrsi %[[SHLI]]
  // CHECK-DAG: %[[HI:.*]] = arith.shrsi %[[LOAD]]
  // CHECK: %[[JOIN:.*]] = tt.join %[[LO]], %[[HI]]
  // CHECK: %[[TRANS:.*]] = tt.trans %[[JOIN]] {order = array<i32: 0, 2, 1>}
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[TRANS]]
  // CHECK: tt.return %[[RESHAPE]] : tensor<8x1xi8>
  %2 = arith.extsi %1 : tensor<8x1xi4> to tensor<8x1xi8>
  tt.return %2 : tensor<8x1xi8>
}

// -----

// CHECK-LABEL: @minor_1d
tt.func @minor_1d(%arg0: !tt.ptr<i4>) -> (tensor<1x8xi8>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64

  %0 = tt.make_tensor_ptr %arg0, [%c1, %c8], [%c1, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<1x8xi4>>
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<1x4xi8>>
  %1 = tt.load %0 : !tt.ptr<tensor<1x8xi4>>
  // CHECK-DAG: %[[SHLI:.*]] = arith.shli %[[LOAD]]
  // CHECK-DAG: %[[LO:.*]] = arith.shrsi %[[SHLI]]
  // CHECK-DAG: %[[HI:.*]] = arith.shrsi %[[LOAD]]
  // CHECK: %[[JOIN:.*]] = tt.join %[[LO]], %[[HI]]
  // CHECK-NOT: tt.trans
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[JOIN]]
  // CHECK: tt.return %[[RESHAPE]] : tensor<1x8xi8>
  %2 = arith.extsi %1 : tensor<1x8xi4> to tensor<1x8xi8>
  tt.return %2 : tensor<1x8xi8>
}

// -----

// CHECK-LABEL: @major_2d
tt.func @major_2d(%arg0: !tt.ptr<i4>) -> (tensor<8x8xi8>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64

  %0 = tt.make_tensor_ptr %arg0, [%c8, %c8], [%c1, %c8], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<8x8xi4>>
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<4x8xi8>>
  %1 = tt.load %0 : !tt.ptr<tensor<8x8xi4>>
  // CHECK-DAG: %[[SHLI:.*]] = arith.shli %[[LOAD]]
  // CHECK-DAG: %[[LO:.*]] = arith.shrsi %[[SHLI]]
  // CHECK-DAG: %[[HI:.*]] = arith.shrsi %[[LOAD]]
  // CHECK: %[[JOIN:.*]] = tt.join %[[LO]], %[[HI]]
  // CHECK: %[[TRANS:.*]] = tt.trans %[[JOIN]] {order = array<i32: 0, 2, 1>}
  // CHECK-SAME: tensor<4x8x2xi8> -> tensor<4x2x8xi8>
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[TRANS]]
  // CHECK: tt.return %[[RESHAPE]] : tensor<8x8xi8>
  %2 = arith.extsi %1 : tensor<8x8xi4> to tensor<8x8xi8>
  tt.return %2 : tensor<8x8xi8>
}

// -----

// CHECK-LABEL: @minor_2d
tt.func @minor_2d(%arg0: !tt.ptr<i4>) -> (tensor<8x8xi8>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64

  %0 = tt.make_tensor_ptr %arg0, [%c8, %c8], [%c8, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<8x8xi4>>
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<8x4xi8>>
  %1 = tt.load %0 : !tt.ptr<tensor<8x8xi4>>
  // CHECK-DAG: %[[SHLI:.*]] = arith.shli %[[LOAD]]
  // CHECK-DAG: %[[LO:.*]] = arith.shrsi %[[SHLI]]
  // CHECK-DAG: %[[HI:.*]] = arith.shrsi %[[LOAD]]
  // CHECK: %[[JOIN:.*]] = tt.join %[[LO]], %[[HI]]
  // CHECK-NOT: tt.trans
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[JOIN]]
  // CHECK: tt.return %[[RESHAPE]] : tensor<8x8xi8>
  %2 = arith.extsi %1 : tensor<8x8xi4> to tensor<8x8xi8>
  tt.return %2 : tensor<8x8xi8>
}

// -----

// CHECK-LABEL: @major_3d
tt.func @major_3d(%arg0: !tt.ptr<i4>) -> (tensor<8x8x8xi8>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64
  %c64 = arith.constant 64 : i64

  %0 = tt.make_tensor_ptr %arg0, [%c8, %c8, %c8], [%c1, %c8, %c64], [%c0, %c0, %c0] {order = array<i32: 2, 1, 0>} : <tensor<8x8x8xi4>>
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : !tt.ptr<tensor<4x8x8xi8>>
  %1 = tt.load %0 : !tt.ptr<tensor<8x8x8xi4>>
  // CHECK-DAG: %[[SHLI:.*]] = arith.shli %[[LOAD]]
  // CHECK-DAG: %[[LO:.*]] = arith.shrsi %[[SHLI]]
  // CHECK-DAG: %[[HI:.*]] = arith.shrsi %[[LOAD]]
  // CHECK: %[[JOIN:.*]] = tt.join %[[LO]], %[[HI]]
  // CHECK: %[[TRANS:.*]] = tt.trans %[[JOIN]] {order = array<i32: 0, 3, 1, 2>}
  // CHECK-SAME: tensor<4x8x8x2xi8> -> tensor<4x2x8x8xi8>
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[TRANS]]
  // CHECK: tt.return %[[RESHAPE]] : tensor<8x8x8xi8>
  %2 = arith.extsi %1 : tensor<8x8x8xi4> to tensor<8x8x8xi8>
  tt.return %2 : tensor<8x8x8xi8>
}
