// RUN: xla-opt %s -xtile-scalarize-scan -cse | FileCheck %s

// CHECK-LABEL: func.func @scan
// CHECK:       %[[OUTPUT:.*]], %{{.*}} = xtile.scan(%arg0) inits(%arg1)
// CHECK-SAME:          dimension = 0 {scan_dim_size = 16 : i64}
// CHECK-SAME:      : (tensor<16x16xf32>), (tensor<16xf32>)
// CHECK-SAME:      -> (tensor<16x16xf32>), (tensor<16xf32>) {
// CHECK:       ^bb0(%arg2: f32, %arg3: f32):
// CHECK-DAG:     %[[LHS:.*]] = tensor.from_elements %arg2 : tensor<f32>
// CHECK-DAG:     %[[RHS:.*]] = tensor.from_elements %arg3 : tensor<f32>
// CHECK:         %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<f32>
// CHECK:         %[[EXTRACT:.*]] = tensor.extract %[[ADD]][] : tensor<f32>
// CHECK:         xtile.yield %[[EXTRACT]], %[[EXTRACT]] : f32, f32
// CHECK:       }
// CHECK:       return %[[OUTPUT]] : tensor<16x16xf32>

func.func @scan(%arg0: tensor<16x16xf32>, %arg1: tensor<16xf32>) -> tensor<16x16xf32> {
  %0, %1 = xtile.scan(%arg0) inits(%arg1) dimension = 0 {scan_dim_size = 16 : i64}
    : (tensor<16x16xf32>), (tensor<16xf32>) -> (tensor<16x16xf32>), (tensor<16xf32>) {
  ^bb0(%arg2: tensor<16xf32>, %arg3: tensor<16xf32>):
    %add = stablehlo.add %arg2, %arg3 : tensor<16xf32>
    stablehlo.return %add, %add : tensor<16xf32>, tensor<16xf32>
  }
  return %0 : tensor<16x16xf32>
}
