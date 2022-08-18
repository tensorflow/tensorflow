// RUN: mhlo-tosa-opt %s --tosa-legalize-mhlo | FileCheck %s

// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.add
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @compare_eq
func.func @compare_eq(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  // CHECK: tosa.equal
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @compare_lt
func.func @compare_lt(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xi1> {
  // CHECK: mhlo.compare
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @compare_ne
func.func @compare_ne(%arg0 : tensor<10xi32>, %arg1 : tensor<10xi32>) -> tensor<10xi1> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.equal"(%arg0, %arg1)
  // CHECK-DAG: %[[VAR1:.*]] = "tosa.logical_not"(%[[VAR0]])
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @maximum
func.func @maximum(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.maximum
  %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @maximum_f64
func.func @maximum_f64(%arg0 : tensor<10xf64>, %arg1 : tensor<10xf64>) -> tensor<10xf64> {
  // CHECK: mhlo.maximum
  %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<10xf64>, tensor<10xf64>) -> tensor<10xf64>
  return %0 : tensor<10xf64>
}

// CHECK-LABEL: @multiply
func.func @multiply(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.mul
  %0 = "mhlo.multiply"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @reduce_max
func.func @reduce_max(%arg0: tensor<1x10xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  // CHECK: tosa.reduce_max
  // CHECK: tosa.reshape
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.maximum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: @reduce_sum
func.func @reduce_sum(%arg0: tensor<5x4xf32>, %arg1: tensor<f32>) -> tensor<4xf32> {
  // CHECK: tosa.reduce_sum
  // CHECK: tosa.reshape
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @subtract
func.func @subtract(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.sub
  %0 = "mhlo.subtract"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
