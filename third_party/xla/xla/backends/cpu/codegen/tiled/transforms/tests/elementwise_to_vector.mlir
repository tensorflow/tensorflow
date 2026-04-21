// RUN: fusion_compiler_opt %s --xtile-cpu-elementwise-to-vector -split-input-file | FileCheck %s

// CHECK-LABEL: @elementwise_add
func.func @elementwise_add(%arg0: tensor<8x1024xf32>, %arg1: tensor<8x1024xf32>) -> tensor<8x1024xf32> {
  // CHECK-DAG: %[[V0:.*]] = vector.transfer_read %arg0[%{{.*}}, %{{.*}}], %{{.*}} : tensor<8x1024xf32>, vector<8x1024xf32>
  // CHECK-DAG: %[[V1:.*]] = vector.transfer_read %arg1[%{{.*}}, %{{.*}}], %{{.*}} : tensor<8x1024xf32>, vector<8x1024xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %[[V0]], %[[V1]] : vector<8x1024xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x1024xf32>
  // CHECK: %[[WRITE:.*]] = vector.transfer_write %[[ADD]], %[[EMPTY]][%{{.*}}, %{{.*}}] {in_bounds = [true, true]} : vector<8x1024xf32>, tensor<8x1024xf32>
  // CHECK: return %[[WRITE]] : tensor<8x1024xf32>
  %0 = arith.addf %arg0, %arg1 : tensor<8x1024xf32>
  return %0 : tensor<8x1024xf32>
}

// -----

// CHECK-LABEL: @constant_convert
func.func @constant_convert() -> tensor<8x1024xf32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8x1024xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8x1024xf32>
  // CHECK: %[[WRITE:.*]] = vector.transfer_write %[[CST]], %[[EMPTY]][%c0, %c0] {in_bounds = [true, true]} : vector<8x1024xf32>, tensor<8x1024xf32>
  // CHECK: return %[[WRITE]] : tensor<8x1024xf32>
  %0 = arith.constant dense<1.000000e+00> : tensor<8x1024xf32>
  return %0 : tensor<8x1024xf32>
}
