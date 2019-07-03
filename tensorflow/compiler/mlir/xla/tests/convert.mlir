// RUN: tf-opt %s -split-input-file -xla-legalize-to-std | FileCheck %s

// -----

// CHECK-LABEL: func @convert.1(%arg0: tensor<f32>) -> tensor<f32> {
func @convert.1(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = "xla.convert"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "xla.convert"(%arg0) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return %0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @convert.2(%arg0: tensor<i32>) -> tensor<i32> {
func @convert.2(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: %0 = "xla.convert"(%arg0) : (tensor<i32>) -> tensor<i32>
  %0 = "xla.convert"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: return %0 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @convert.3(%arg0: tensor<i32>) -> tensor<i64> {
func @convert.3(%arg0: tensor<i32>) -> tensor<i64> {
  // CHECK-NEXT: %0 = "xla.convert"(%arg0) : (tensor<i32>) -> tensor<i64>
  %0 = "xla.convert"(%arg0) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return %0 : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @convert.4(%arg0: tensor<f32>) -> tensor<i32> {
func @convert.4(%arg0: tensor<f32>) -> tensor<i32> {
  // CHECK-NEXT: %0 = "xla.convert"(%arg0) : (tensor<f32>) -> tensor<i32>
  %0 = "xla.convert"(%arg0) : (tensor<f32>) -> tensor<i32>
  // CHECK-NEXT: return %0 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @convert.5(%arg0: tensor<i32>) -> tensor<f32> {
func @convert.5(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK-NEXT: %0 = "xla.convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  %0 = "xla.convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return %0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----


// CHECK-LABEL: func @convert.const.1() -> tensor<f32> {
func @convert.const.1() -> tensor<f32> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<f32>
  %cst = constant  dense<42.0> : tensor<f32>
  %0 = "xla.convert"(%cst) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT: return %cst : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// check-label: func @convert.const.2() -> tensor<i32> {
func @convert.const.2() -> tensor<i32> {
  // check-next: %cst = constant dense<42> : tensor<i32>
  %cst = constant  dense<42> : tensor<i32>
  %0 = "xla.convert"(%cst) : (tensor<i32>) -> tensor<i32>
  // check-next: return %cst : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @convert.const.3() -> tensor<i32> {
func @convert.const.3() -> tensor<i32> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<i32>
  %cst = constant  dense<42.0> : tensor<f32>
  %0 = "xla.convert"(%cst) : (tensor<f32>) -> tensor<i32>
  // CHECK-NEXT: return %cst : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @convert.const.4() -> tensor<f32> {
func @convert.const.4() -> tensor<f32> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<f32>
  %cst = constant  dense<42> : tensor<i32>
  %0 = "xla.convert"(%cst) : (tensor<i32>) -> tensor<f32>
  // CHECK-NEXT: return %cst : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @convert.const.5() -> tensor<bf16> {
func @convert.const.5() -> tensor<bf16> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<bf16>
  %cst = constant  dense<42> : tensor<i32>
  %0 = "xla.convert"(%cst) : (tensor<i32>) -> tensor<bf16>
  // CHECK-NEXT: return %cst : tensor<bf16>
  return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @convert.const.6() -> tensor<i16> {
func @convert.const.6() -> tensor<i16> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<i16>
  %cst = constant  dense<42.0> : tensor<bf16>
  %0 = "xla.convert"(%cst) : (tensor<bf16>) -> tensor<i16>
  // CHECK-NEXT: return %cst : tensor<i16>
  return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: func @convert.const.7() -> tensor<i32> {
func @convert.const.7() -> tensor<i32> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<i32>
  %cst = constant  dense<42> : tensor<i64>
  %0 = "xla.convert"(%cst) : (tensor<i64>) -> tensor<i32>
  // CHECK-NEXT: return %cst : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func @convert.const.8() -> tensor<i64> {
func @convert.const.8() -> tensor<i64> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<i64>
  %cst = constant  dense<42> : tensor<i32>
  %0 = "xla.convert"(%cst) : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: return %cst : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @convert.const.9() -> tensor<f32> {
func @convert.const.9() -> tensor<f32> {
  // CHECK-NEXT: %cst = constant  dense<4.200000e+01> : tensor<f32>
  %cst = constant  dense<42.0> : tensor<f64>
  %0 = "xla.convert"(%cst) : (tensor<f64>) -> tensor<f32>
  // CHECK-NEXT: return %cst : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @convert.const.9() -> tensor<bf16> {
func @convert.const.9() -> tensor<bf16> {
  // CHECK-NEXT: %cst = constant  dense<4.200000e+01> : tensor<bf16>
  %cst = constant  dense<42.0> : tensor<f32>
  %0 = "xla.convert"(%cst) : (tensor<f32>) -> tensor<bf16>
  // CHECK-NEXT: return %cst : tensor<bf16>
  return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: func @convert.const.10() -> tensor<f64> {
func @convert.const.10() -> tensor<f64> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<f64>
  %cst = constant  dense<42.0> : tensor<bf16>
  %0 = "xla.convert"(%cst) : (tensor<bf16>) -> tensor<f64>
  // CHECK-NEXT: return %cst : tensor<f64>
  return %0 : tensor<f64>
}

// -----

// CHECK-LABEL: func @convert.const.11() -> tensor<f64> {
func @convert.const.11() -> tensor<f64> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<f64>
  %cst = constant  dense<42.0> : tensor<bf16>
  %0 = "xla.convert"(%cst) : (tensor<bf16>) -> tensor<f64>
  // CHECK-NEXT: return %cst : tensor<f64>
  return %0 : tensor<f64>
}


// -----

// CHECK-LABEL: func @convert.const.12() -> tensor<i64> {
func @convert.const.12() -> tensor<i64> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<i64>
  %cst = constant  dense<42.0> : tensor<bf16>
  %0 = "xla.convert"(%cst) : (tensor<bf16>) -> tensor<i64>
  // CHECK-NEXT: return %cst : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @convert.const.13() -> tensor<i64> {
func @convert.const.13() -> tensor<i64> {
  // CHECK-NEXT: %cst = constant dense<42> : tensor<i64>
  %cst = constant  dense<42> : tensor<i16>
  %0 = "xla.convert"(%cst) : (tensor<i16>) -> tensor<i64>
  // CHECK-NEXT: return %cst : tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: func @convert.const.14() -> tensor<f64> {
func @convert.const.14() -> tensor<f64> {
  // CHECK-NEXT: %cst = constant dense<4.200000e+01> : tensor<f64>
  %cst = constant  dense<42> : tensor<i16>
  %0 = "xla.convert"(%cst) : (tensor<i16>) -> tensor<f64>
  // CHECK-NEXT: return %cst : tensor<f64>
  return %0 : tensor<f64>
}
