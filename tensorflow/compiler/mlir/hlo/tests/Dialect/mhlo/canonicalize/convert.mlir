// RUN: mlir-hlo-opt %s -split-input-file -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// -----

// CHECK-LABEL: func @same_type
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @same_type(%arg: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.convert(%arg) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return [[ARG]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @non_const_chained_convert_unused_parent
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_convert_unused_parent(%arg: tensor<f16>) -> tensor<f64> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<f16>) -> tensor<f64>
  %0 = mhlo.convert(%arg) : (tensor<f16>) -> tensor<f32>
  %1 = mhlo.convert(%0) : (tensor<f32>) -> tensor<f64>
  // CHECK-NEXT: return [[RES]]
  func.return %1 : tensor<f64>
}

// -----

// CHECK-LABEL: func @non_const_chained_convert_unused_parent_integer
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_convert_unused_parent_integer(%arg: tensor<ui16>) -> tensor<i64> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<ui16>) -> tensor<i64>
  %0 = mhlo.convert(%arg) : (tensor<ui16>) -> tensor<i32>
  %1 = mhlo.convert(%0) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[RES]]
  func.return %1 : tensor<i64>
}

// -----

// CHECK-LABEL: func @not_convert_float_lower_width
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @not_convert_float_lower_width(%arg: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: [[VAL0:%.+]] = mhlo.convert([[ARG]]) : (tensor<f32>) -> tensor<f16>
  // CHECK-NEXT: [[VAL1:%.+]] = mhlo.convert([[VAL0]]) : (tensor<f16>) -> tensor<f32>
  %0 = mhlo.convert(%arg) : (tensor<f32>) -> tensor<f16>
  %1 = mhlo.convert(%0) : (tensor<f16>) -> tensor<f32>
  // CHECK-NEXT: return [[VAL1]]
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: func @non_const_chained_convert_becomes_noop
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_convert_becomes_noop(%arg: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.convert(%arg) : (tensor<f32>) -> tensor<f64>
  %1 = mhlo.convert(%0) : (tensor<f64>) -> tensor<f32>
  // CHECK-NEXT: return [[ARG]]
  func.return %1 : tensor<f32>
}

// -----
// CHECK-LABEL: func @non_const_chained_convert
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @non_const_chained_convert(%arg: tensor<f16>) -> (tensor<f32>, tensor<f64>) {
  // CHECK-NEXT: [[RES0:%.+]] = mhlo.convert([[ARG]]) : (tensor<f16>) -> tensor<f32>
  // CHECK-NEXT: [[RES1:%.+]] = mhlo.convert([[ARG]]) : (tensor<f16>) -> tensor<f64>
  %0 = mhlo.convert(%arg) : (tensor<f16>) -> tensor<f32>
  %1 = mhlo.convert(%0) : (tensor<f32>) -> tensor<f64>
  // CHECK-NEXT: return [[RES0]], [[RES1]]
  func.return %0, %1 : tensor<f32>, tensor<f64>
}

// -----

// CHECK-LABEL: func @int_widening
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @int_widening(%arg: tensor<i32>) -> tensor<i64> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<i32>) -> tensor<i64>
  %0 = mhlo.convert(%arg) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[RES]]
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @int_narrowing
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @int_narrowing(%arg: tensor<i32>) -> tensor<i16> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<i32>) -> tensor<i16>
  %0 = mhlo.convert(%arg) : (tensor<i32>) -> tensor<i16>
  // CHECK-NEXT: return [[RES]]
  func.return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: func @float_int
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @float_int(%arg: tensor<f32>) -> tensor<i32> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<f32>) -> tensor<i32>
  %0 = mhlo.convert(%arg) : (tensor<f32>) -> tensor<i32>
  // CHECK-NEXT: return [[RES]]
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @int_float
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @int_float(%arg: tensor<i32>) -> tensor<f32> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<i32>) -> tensor<f32>
  %0 = mhlo.convert(%arg) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return [[RES]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @high_rank_tensor
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @high_rank_tensor(%arg: tensor<2x3xi32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: [[RES:%.+]] = mhlo.convert([[ARG]]) : (tensor<2x3xi32>) -> tensor<2x3xf32>
  %0 = mhlo.convert(%arg) : (tensor<2x3xi32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return [[RES]]
  func.return %0 : tensor<2x3xf32>
}

// -----


// CHECK-LABEL: func @const_same_type
func.func @const_same_type() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i32>
  %cst = mhlo.constant dense<42> : tensor<i32>
  %0 = mhlo.convert(%cst) : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_float_int
func.func @const_float_int() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i32>
  %cst = mhlo.constant dense<42.0> : tensor<f32>
  %0 = mhlo.convert(%cst) : (tensor<f32>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_int_float
func.func @const_int_float() -> tensor<f32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<4.{{0*}}e+00> : tensor<f32>
  %cst = mhlo.constant dense<4> : tensor<i32>
  %0 = mhlo.convert(%cst) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_negative_int_float
func.func @const_negative_int_float() -> tensor<f32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<-4.{{0*}}e+00> : tensor<f32>
  %cst = mhlo.constant dense<-4> : tensor<i32>
  %0 = mhlo.convert(%cst) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_int_bf16
func.func @const_int_bf16() -> tensor<bf16> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<4.{{0*}}e+00> : tensor<bf16>
  %cst = mhlo.constant dense<4> : tensor<i32>
  %0 = mhlo.convert(%cst) : (tensor<i32>) -> tensor<bf16>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @const_bool_f32
func.func @const_bool_f32() -> tensor<2xf32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
  %cst = mhlo.constant dense<[0, 1]> : tensor<2xi1>
  %0 = mhlo.convert(%cst) : (tensor<2xi1>) -> tensor<2xf32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @const_bf16_int16
func.func @const_bf16_int16() -> tensor<i16> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i16>
  %cst = mhlo.constant dense<42.0> : tensor<bf16>
  %0 = mhlo.convert(%cst) : (tensor<bf16>) -> tensor<i16>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: func @const_int_narrowing
func.func @const_int_narrowing() -> tensor<i32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i32>
  %cst = mhlo.constant dense<42> : tensor<i64>
  %0 = mhlo.convert(%cst) : (tensor<i64>) -> tensor<i32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_bool_widening
func.func @const_bool_widening() -> tensor<i64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i64>
  %cst = mhlo.constant dense<42> : tensor<i32>
  %0 = mhlo.convert(%cst) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @const_int_widening
func.func @const_int_widening() -> tensor<2xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<[0, 1]> : tensor<2xi32>
  %cst = mhlo.constant dense<[0, 1]> : tensor<2xi1>
  %0 = mhlo.convert(%cst) : (tensor<2xi1>) -> tensor<2xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @const_negative_int_widening
func.func @const_negative_int_widening() -> tensor<i64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<-42> : tensor<i64>
  %cst = mhlo.constant dense<-42> : tensor<i32>
  %0 = mhlo.convert(%cst) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @const_float_narrowing
func.func @const_float_narrowing() -> tensor<f32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<4.2{{0*}}e+00> : tensor<f32>
  %cst = mhlo.constant dense<4.2> : tensor<f64>
  %0 = mhlo.convert(%cst) : (tensor<f64>) -> tensor<f32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_f32_bf16
func.func @const_f32_bf16() -> tensor<bf16> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<4.2{{0*}}e+01> : tensor<bf16>
  %cst = mhlo.constant dense<42.0> : tensor<f32>
  %0 = mhlo.convert(%cst) : (tensor<f32>) -> tensor<bf16>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @const_bf16_f64
func.func @const_bf16_f64() -> tensor<f64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<4.187500e+00> : tensor<f64>
  %cst = mhlo.constant dense<4.2> : tensor<bf16>
  %0 = mhlo.convert(%cst) : (tensor<bf16>) -> tensor<f64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<f64>
}

// -----

// CHECK-LABEL: func @const_bf16_int64
func.func @const_bf16_int64() -> tensor<i64> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<42> : tensor<i64>
  %cst = mhlo.constant dense<42.0> : tensor<bf16>
  %0 = mhlo.convert(%cst) : (tensor<bf16>) -> tensor<i64>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<i64>
}


// -----

// CHECK-LABEL: func @const_high_rank_tensor
func.func @const_high_rank_tensor() -> tensor<2x3xi32> {
  // CHECK-NEXT: [[CST:%.+]] = mhlo.constant dense<[
  // CHECK-SAME:     [1, 2, 3], [4, 5, 6]
  // CHECK-SAME: ]> : tensor<2x3xi32>
  %cst = mhlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %0 = mhlo.convert(%cst) : (tensor<2x3xf32>) -> tensor<2x3xi32>
  // CHECK-NEXT: return [[CST]]
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @const_int_complex
func.func @const_int_complex() -> tensor<2xcomplex<f32>> {
  %cst = mhlo.constant dense<[0, 1]> : tensor<2xi1>
  // CHECK: mhlo.convert
  %0 = mhlo.convert(%cst) : (tensor<2xi1>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @const_float_complex
func.func @const_float_complex() -> tensor<2xcomplex<f64>> {
  %cst = mhlo.constant dense<[0.0, 1.0]> : tensor<2xf32>
  // CHECK: mhlo.convert
  %0 = mhlo.convert(%cst) : (tensor<2xf32>) -> tensor<2xcomplex<f64>>
  func.return %0 : tensor<2xcomplex<f64>>
}


// -----

// CHECK-LABEL: func @const_complex_int
func.func @const_complex_int() -> tensor<i32> {
  %cst = mhlo.constant dense<(0.0, 1.0)> : tensor<complex<f32>>
  // CHECK: mhlo.convert
  %0 = mhlo.convert(%cst) : (tensor<complex<f32>>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @const_complex_float
func.func @const_complex_float() -> tensor<f32> {
  %cst = mhlo.constant dense<(0.0, 1.0)> : tensor<complex<f32>>
  // CHECK: mhlo.convert
  %0 = mhlo.convert(%cst) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @const_complex_complex
func.func @const_complex_complex() -> tensor<complex<f64>> {
  %cst = mhlo.constant dense<(0.0, 1.0)> : tensor<complex<f32>>
  // CHECK: mhlo.convert
  %0 = mhlo.convert(%cst) : (tensor<complex<f32>>) -> tensor<complex<f64>>
  func.return %0 : tensor<complex<f64>>
}
