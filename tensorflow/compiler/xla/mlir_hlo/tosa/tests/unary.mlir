// RUN: mhlo-tosa-opt %s --tosa-legalize-mhlo | FileCheck %s

// CHECK-LABEL: @abs
func.func @abs(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.abs
  %0 = "mhlo.abs"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @ceil
func.func @ceil(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.ceil
  %0 = "mhlo.ceil"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @convert
func.func @convert(%arg : tensor<10xi32>) -> tensor<10xf32> {
  // CHECK: tosa.cast
  %0 = "mhlo.convert"(%arg) : (tensor<10xi32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @exponential
func.func @exponential(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.exp
  %0 = "mhlo.exponential"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @floor
func.func @floor(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.floor
  %0 = "mhlo.floor"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @is_finite
func.func @is_finite(%arg : tensor<10xf32>) -> tensor<10xi1> {
  // CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() {value = dense<0x7F800000>
  // CHECK-DAG: %[[VAR1:.*]] = "tosa.abs"(%arg0)
  // CHECK-DAG: %[[VAR2:.*]] = "tosa.equal"(%[[VAR1]], %[[VAR0]])
  // CHECK-DAG: %[[VAR3:.*]] = "tosa.logical_not"(%[[VAR2]])
  %0 = "mhlo.is_finite"(%arg) : (tensor<10xf32>) -> tensor<10xi1>
  return %0 : tensor<10xi1>
}

// CHECK-LABEL: @log
func.func @log(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.log
  %0 = "mhlo.log"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @negate
func.func @negate(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.negate
  %0 = "mhlo.negate"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: @tanh
func.func @tanh(%arg : tensor<10xf32>) -> tensor<10xf32> {
  // CHECK: tosa.tanh
  %0 = "mhlo.tanh"(%arg) : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
