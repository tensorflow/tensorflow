// RUN: tf-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

// -----

// CHECK-LABEL: func @same_type
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @same_type(%arg: tensor<f32>) -> tensor<f32> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @int_widening
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @int_widening(%arg: tensor<i32>) -> tensor<i64> {
  // CHECK-NEXT: [[RES:%.+]] = "xla_hlo.convert"([[ARG]]) : (tensor<i32>) -> tensor<i64>
  %0 = "xla_hlo.convert"(%arg) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @int_narrowing
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @int_narrowing(%arg: tensor<i32>) -> tensor<i16> {
  // CHECK-NEXT: [[RES:%.+]] = "xla_hlo.convert"([[ARG]]) : (tensor<i32>) -> tensor<i16>
  %0 = "xla_hlo.convert"(%arg) : (tensor<i32>) -> tensor<i16>
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: func @float_int
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @float_int(%arg: tensor<f32>) -> tensor<i32> {
  // CHECK-NEXT: [[RES:%.+]] = "xla_hlo.convert"([[ARG]]) : (tensor<f32>) -> tensor<i32>
  %0 = "xla_hlo.convert"(%arg) : (tensor<f32>) -> tensor<i32>
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @int_float
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @int_float(%arg: tensor<i32>) -> tensor<f32> {
  // CHECK-NEXT: [[RES:%.+]] = "xla_hlo.convert"([[ARG]]) : (tensor<i32>) -> tensor<f32>
  %0 = "xla_hlo.convert"(%arg) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @high_rank_tensor
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @high_rank_tensor(%arg: tensor<2x3xi32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: [[RES:%.+]] = "xla_hlo.convert"([[ARG]]) : (tensor<2x3xi32>) -> tensor<2x3xf32>
  %0 = "xla_hlo.convert"(%arg) : (tensor<2x3xi32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[RES]]
  return %0 : tensor<2x3xf32>
}

// -----


// CHECK-LABEL: func @const_same_type
func @const_same_type() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<42> : tensor<i32>
  %cst = xla_hlo.constant dense<42> : tensor<i32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_float_int
func @const_float_int() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<42> : tensor<i32>
  %cst = xla_hlo.constant dense<42.0> : tensor<f32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<f32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_int_float
func @const_int_float() -> tensor<f32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<4.{{0*}}e+00> : tensor<f32>
  %cst = xla_hlo.constant dense<4> : tensor<i32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_negative_int_float
func @const_negative_int_float() -> tensor<f32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<-4.{{0*}}e+00> : tensor<f32>
  %cst = xla_hlo.constant dense<-4> : tensor<i32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_int_bf16
func @const_int_bf16() -> tensor<bf16> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<4.{{0*}}e+00> : tensor<bf16>
  %cst = xla_hlo.constant dense<4> : tensor<i32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i32>) -> tensor<bf16>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @const_bf16_int
func @const_bf16_int() -> tensor<i16> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<42> : tensor<i16>
  %cst = xla_hlo.constant dense<42.0> : tensor<bf16>
  %0 = "xla_hlo.convert"(%cst) : (tensor<bf16>) -> tensor<i16>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: func @const_int_narrowing
func @const_int_narrowing() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<42> : tensor<i32>
  %cst = xla_hlo.constant dense<42> : tensor<i64>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i64>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_int_widening
func @const_int_widening() -> tensor<i64> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<42> : tensor<i64>
  %cst = xla_hlo.constant dense<42> : tensor<i32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @const_negative_int_widening
func @const_negative_int_widening() -> tensor<i64> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<-42> : tensor<i64>
  %cst = xla_hlo.constant dense<-42> : tensor<i32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @const_float_narrowing
func @const_float_narrowing() -> tensor<f32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<4.2{{0*}}e+00> : tensor<f32>
  %cst = xla_hlo.constant dense<4.2> : tensor<f64>
  %0 = "xla_hlo.convert"(%cst) : (tensor<f64>) -> tensor<f32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_f32_bf16
func @const_f32_bf16() -> tensor<bf16> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<4.2{{0*}}e+01> : tensor<bf16>
  %cst = xla_hlo.constant dense<42.0> : tensor<f32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<f32>) -> tensor<bf16>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @const_bf16_f64
func @const_bf16_f64() -> tensor<f64> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<4.2{{0*}}e+00> : tensor<f64>
  %cst = xla_hlo.constant dense<4.2> : tensor<bf16>
  %0 = "xla_hlo.convert"(%cst) : (tensor<bf16>) -> tensor<f64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<f64>
}

// -----

// CHECK-LABEL: func @const_bf16_int
func @const_bf16_int() -> tensor<i64> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<42> : tensor<i64>
  %cst = xla_hlo.constant dense<42.0> : tensor<bf16>
  %0 = "xla_hlo.convert"(%cst) : (tensor<bf16>) -> tensor<i64>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<i64>
}


// -----

// CHECK-LABEL: func @const_high_rank_tensor
func @const_high_rank_tensor() -> tensor<2x3xi32> {
  // CHECK-NEXT: [[CST:%.+]] = xla_hlo.constant dense<[
  // CHECK-SAME:     [1, 2, 3], [4, 5, 6]
  // CHECK-SAME: ]> : tensor<2x3xi32>
  %cst = xla_hlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %0 = "xla_hlo.convert"(%cst) : (tensor<2x3xf32>) -> tensor<2x3xi32>
  // CHECK-NEXT: return [[CST]]
  return %0 : tensor<2x3xi32>
}

