// RUN: xla-opt %s -xtile-strip-mine-scan="tile-size=4" -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @scan
// CHECK-SAME: %[[INPUT:.*]]: memref<10x1xf32>, %[[INIT:.*]]: tensor<1xf32>, %[[OUTPUT:.*]]: memref<10x1xf32>
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<10> : tensor<4xi32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C10]] step %[[C4]] iter_args(%[[ITER:.*]] = %[[INIT]]) -> (tensor<1xf32>) {
// CHECK:   %[[EXTRACT:.*]] = xtile.extract %[[INPUT]][%[[IV]], %[[C0]]] [4, 1] [1, 1] : memref<10x1xf32> -> tensor<4x1xf32>
// CHECK:   %[[MUL:.*]] = arith.mulf %[[EXTRACT]], %[[EXTRACT]] : tensor<4x1xf32>
// CHECK:   %[[RANGE:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:   %[[CAST:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:   %[[SPLAT:.*]] = tt.splat %[[CAST]] : i32 -> tensor<4xi32>
// CHECK:   %[[ADDI:.*]] = arith.addi %[[RANGE]], %[[SPLAT]] : tensor<4xi32>
// CHECK:   %[[CMPI:.*]] = arith.cmpi slt, %[[ADDI]], %[[CST]] : tensor<4xi32>
// CHECK:   %[[EXPAND_MASK:.*]] = tt.expand_dims %[[CMPI]] {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
// CHECK:   %[[EXPAND_INIT:.*]] = tt.expand_dims %[[INIT]] {axis = 0 : i32} : tensor<1xf32> -> tensor<1x1xf32>
// CHECK:   %[[BCAST_INIT:.*]] = tt.broadcast %[[EXPAND_INIT]] : tensor<1x1xf32> -> tensor<4x1xf32>
// CHECK:   %[[SELECT:.*]] = arith.select %[[EXPAND_MASK]], %[[MUL]], %[[BCAST_INIT]] : tensor<4x1xi1>, tensor<4x1xf32>
// CHECK:   %[[OUTPUTS:.*]], %[[CARRIES:.*]] = xtile.scan(%[[SELECT]]) inits(%[[ITER]]) dimension = 0 {scan_dim_size = 4 : i64} : (tensor<4x1xf32>), (tensor<1xf32>) -> (tensor<4x1xf32>), (tensor<1xf32>) {
// CHECK:   ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
// CHECK:     %[[ADD:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
// CHECK:     xtile.yield %[[ADD]], %[[ADD]] : f32, f32
// CHECK:   }
// CHECK:   %[[DIV:.*]] = arith.divf %[[OUTPUTS]], %[[OUTPUTS]] : tensor<4x1xf32>
// CHECK:   xtile.insert %[[DIV]] into %[[OUTPUT]][%[[IV]], %[[C0]]] [4, 1] [1, 1] : tensor<4x1xf32> -> memref<10x1xf32>
// CHECK:   scf.yield %[[CARRIES]] : tensor<1xf32>
// CHECK: }
// CHECK: return %[[FOR]] : tensor<1xf32>

func.func @scan(%arg0: memref<10x1xf32>, %arg1: tensor<1xf32>, %arg2: memref<10x1xf32>) -> tensor<1xf32> {

  %c0 = arith.constant 0 : index
  %0 = xtile.extract %arg0[%c0, %c0] [10, 1] [1, 1] : memref<10x1xf32> -> tensor<10x1xf32>
  %mul = arith.mulf %0, %0 : tensor<10x1xf32>  // Some prologue.

  %1, %2 = xtile.scan(%mul) inits(%arg1) dimension = 0 {scan_dim_size = 10 : i64}
    : (tensor<10x1xf32>), (tensor<1xf32>) -> (tensor<10x1xf32>), (tensor<1xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %add = arith.addf %arg3, %arg4 : f32
    xtile.yield %add, %add : f32, f32
  }

  %div = arith.divf %1, %1 : tensor<10x1xf32>  // Some epilogue.
  xtile.insert %div into %arg2[%c0, %c0] [10, 1] [1, 1] : tensor<10x1xf32> -> memref<10x1xf32>

  return %2 : tensor<1xf32>
}
