// RUN: tf-opt %s -canonicalize | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: @add_float
func.func @add_float() -> (tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.constant dense<4.5> : tensor<f32>
  %1 = arith.constant dense<1.5> : tensor<f32>

  %2 = arith.constant dense< 3.5> : tensor<4xf32>
  %3 = arith.constant dense<-0.5> : tensor<4xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<3.500000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<-5.000000e-01> : tensor<4xf32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<6.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST_2:.*]] = arith.constant dense<4.000000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_3:.*]] = arith.constant dense<5.000000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_4:.*]] = arith.constant dense<3.000000e+00> : tensor<4xf32>
  // CHECK: %0 = tfl.add %[[CST]], %[[CST_0]] {fused_activation_function = "SIGN_BIT"} : tensor<4xf32>

  %5 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<  f32>) -> tensor<  f32>
  %6 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = "tfl.add"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<  f32>) -> tensor<4xf32>
  %8 = "tfl.add"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %9 = "tfl.add"(%2, %3) {fused_activation_function = "SIGN_BIT"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %5, %6, %7, %8, %9 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @add_int
func.func @add_int() -> (tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = arith.constant dense<8> : tensor<i32>
  %1 = arith.constant dense<1> : tensor<i32>

  %2 = arith.constant dense< 4> : tensor<4xi32>
  %3 = arith.constant dense<-2> : tensor<4xi32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<9> : tensor<i32>
  // CHECK-DAG: %[[CST_0:.*]]  = arith.constant dense<6> : tensor<4xi32>
  // CHECK-DAG: %[[CST_1:.*]]  = arith.constant dense<5> : tensor<4xi32>
  // CHECK-DAG: %[[CST_2:.*]]  = arith.constant dense<2> : tensor<4xi32>

  %5 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<  i32>) -> tensor<  i32>
  %6 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<4xi32>) -> tensor<4xi32>
  %7 = "tfl.add"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<  i32>) -> tensor<4xi32>
  %8 = "tfl.add"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %5, %6, %7, %8 : tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @sub_float
func.func @sub_float() -> (tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.constant dense<4.5> : tensor<f32>
  %1 = arith.constant dense<1.5> : tensor<f32>

  %2 = arith.constant dense< 3.5> : tensor<4xf32>
  %3 = arith.constant dense<-0.5> : tensor<4xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<3.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST_0:.*]]  = arith.constant dense<5.000000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_1:.*]]  = arith.constant dense<2.000000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_2:.*]]  = arith.constant dense<4.000000e+00> : tensor<4xf32>

  %5 = "tfl.sub"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<  f32>) -> tensor<  f32>
  %6 = "tfl.sub"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = "tfl.sub"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<  f32>) -> tensor<4xf32>
  %8 = "tfl.sub"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %5, %6, %7, %8 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @sub_int
func.func @sub_int() -> (tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = arith.constant dense<8> : tensor<i32>
  %1 = arith.constant dense<1> : tensor<i32>

  %2 = arith.constant dense< 4> : tensor<4xi32>
  %3 = arith.constant dense<-2> : tensor<4xi32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<7> : tensor<i32>
  // CHECK-DAG: %[[CST_0:.*]]  = arith.constant dense<10> : tensor<4xi32>
  // CHECK-DAG: %[[CST_1:.*]]  = arith.constant dense<3> : tensor<4xi32>
  // CHECK-DAG: %[[CST_2:.*]]  = arith.constant dense<6> : tensor<4xi32>

  %5 = "tfl.sub"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<  i32>) -> tensor<  i32>
  %6 = "tfl.sub"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<4xi32>) -> tensor<4xi32>
  %7 = "tfl.sub"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<  i32>) -> tensor<4xi32>
  %8 = "tfl.sub"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %5, %6, %7, %8 : tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @sub_zero
func.func @sub_zero(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<4xi32>
  %zero_float = arith.constant dense<0.0> : tensor<4xf32>

  // CHECK-NOT: tfl.sub
  // CHECK: return %arg0, %arg1

  %0 = "tfl.sub"(%arg0, %zero_int) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.sub"(%arg1, %zero_float) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @sub_zero_lhs
func.func @sub_zero_lhs(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<4xi32>
  %zero_float = arith.constant dense<0.0> : tensor<4xf32>

  // CHECK: tfl.sub
  // CHECK: return %0, %1

  %0 = "tfl.sub"(%zero_int, %arg0) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.sub"(%zero_float, %arg1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @sub_zero_quant
func.func @sub_zero_quant(%arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>> {
  %zero = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<u8:f32, 1.0>>, value = dense<0> : tensor<32xi8>} : () -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  // CHECK: %[[SUB:.*]] = tfl.sub
  // CHECK: return %[[SUB]]

  %0 = "tfl.sub"(%arg0, %zero) {fused_activation_function = "NONE"} : (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  func.return %0 : tensor<32x!quant.uniform<u8:f32, 1.0>>
}

// CHECK-LABEL: @mul_float
func.func @mul_float() -> (tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.constant dense<4.5> : tensor<f32>
  %1 = arith.constant dense<1.5> : tensor<f32>

  %2 = arith.constant dense< 3.5> : tensor<4xf32>
  %3 = arith.constant dense<-0.5> : tensor<4xf32>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<6.750000e+00> : tensor<f32>
  // CHECK-DAG: %[[CST_0:.*]]  = arith.constant dense<-2.250000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_1:.*]]  = arith.constant dense<5.250000e+00> : tensor<4xf32>
  // CHECK-DAG: %[[CST_2:.*]]  = arith.constant dense<-1.750000e+00> : tensor<4xf32>

  %5 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<  f32>) -> tensor<  f32>
  %6 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = "tfl.mul"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<  f32>) -> tensor<4xf32>
  %8 = "tfl.mul"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %5, %6, %7, %8 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @mul_bf16
func.func @mul_bf16() -> (tensor<bf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>) {
  %0 = arith.constant dense<4.5> : tensor<bf16>
  %1 = arith.constant dense<1.5> : tensor<bf16>

  %2 = arith.constant dense< 3.5> : tensor<4xbf16>
  %3 = arith.constant dense<-0.5> : tensor<4xbf16>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<6.750000e+00> : tensor<bf16>
  // CHECK-DAG: %[[CST_0:.*]]  = arith.constant dense<-2.250000e+00> : tensor<4xbf16>
  // CHECK-DAG: %[[CST_1:.*]]  = arith.constant dense<5.250000e+00> : tensor<4xbf16>
  // CHECK-DAG: %[[CST_2:.*]]  = arith.constant dense<-1.750000e+00> : tensor<4xbf16>

  %5 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  bf16>, tensor<  bf16>) -> tensor<  bf16>
  %6 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  bf16>, tensor<4xbf16>) -> tensor<4xbf16>
  %7 = "tfl.mul"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xbf16>, tensor<  bf16>) -> tensor<4xbf16>
  %8 = "tfl.mul"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xbf16>

  func.return %5, %6, %7, %8 : tensor<bf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>
}

// CHECK-LABEL: @mul_f16
func.func @mul_f16() -> (tensor<f16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) {
  %0 = arith.constant dense<4.5> : tensor<f16>
  %1 = arith.constant dense<1.5> : tensor<f16>

  %2 = arith.constant dense< 3.5> : tensor<4xf16>
  %3 = arith.constant dense<-0.5> : tensor<4xf16>

  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<6.750000e+00> : tensor<f16>
  // CHECK-DAG: %[[CST_0:.*]]  = arith.constant dense<-2.250000e+00> : tensor<4xf16>
  // CHECK-DAG: %[[CST_1:.*]]  = arith.constant dense<5.250000e+00> : tensor<4xf16>
  // CHECK-DAG: %[[CST_2:.*]]  = arith.constant dense<-1.750000e+00> : tensor<4xf16>

  %5 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f16>, tensor<  f16>) -> tensor<  f16>
  %6 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f16>, tensor<4xf16>) -> tensor<4xf16>
  %7 = "tfl.mul"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf16>, tensor<  f16>) -> tensor<4xf16>
  %8 = "tfl.mul"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf16>, tensor<4xf16>) -> tensor<4xf16>

  func.return %5, %6, %7, %8 : tensor<f16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>
}

// CHECK-LABEL: @mul_zero
func.func @mul_zero(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<4xi32>
  %zero_float = arith.constant dense<0.0> : tensor<4xf32>

  // CHECK-NOT: tfl.mul
  // CHECK: return %cst, %cst_0

  %0 = "tfl.mul"(%arg0, %zero_int) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.mul"(%arg1, %zero_float) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @mul_zero_lhs
func.func @mul_zero_lhs(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<4xi32>
  %zero_float = arith.constant dense<0.0> : tensor<4xf32>

  // CHECK-NOT: tfl.mul
  // CHECK: return %cst, %cst_0

  %0 = "tfl.mul"(%zero_int, %arg0) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.mul"(%zero_float, %arg1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @mul_one
func.func @mul_one(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %one_int = arith.constant dense<1> : tensor<4xi32>
  %one_float = arith.constant dense<1.0> : tensor<4xf32>

  // CHECK-NOT: tfl.mul
  // CHECK: return %arg0, %arg1

  %0 = "tfl.mul"(%arg0, %one_int) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.mul"(%arg1, %one_float) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @mul_one_lhs
func.func @mul_one_lhs(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %one_int = arith.constant dense<1> : tensor<4xi32>
  %one_float = arith.constant dense<1.0> : tensor<4xf32>

  // CHECK-NOT: tfl.mul
  // CHECK: return %arg0, %arg1

  %0 = "tfl.mul"(%one_int, %arg0) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.mul"(%one_float, %arg1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @mul_one_quant
func.func @mul_one_quant(%arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>> {
  %one = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<u8:f32, 1.0>>, value = dense<1> : tensor<32xi8>} : () -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  // CHECK: %[[MUL:.*]] = tfl.mul
  // CHECK: return %[[MUL]]

  %0 = "tfl.mul"(%one, %arg0) {fused_activation_function = "NONE"} : (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  func.return %0 : tensor<32x!quant.uniform<u8:f32, 1.0>>
}


// CHECK-LABEL: @elementwise_unary_ops
func.func @elementwise_unary_ops() -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  %0 = arith.constant dense<-1.0> : tensor<f32>
  %1 = arith.constant dense<1.0> : tensor<f32>
  %2 = arith.constant dense<1.0> : tensor<f32>
  %3 = arith.constant dense<1.0> : tensor<f32>
  %4 = arith.constant dense<4.0> : tensor<f32>
  %5 = arith.constant dense<4.0> : tensor<f32>
  %6 = arith.constant dense<2.0> : tensor<f32>

  // CHECK-DAG: [[cst0:%.*]] = arith.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: [[cst1:%.*]] = arith.constant dense<0.841470957> : tensor<f32>
  // CHECK-DAG: [[cst2:%.*]] = arith.constant dense<0.540302277> : tensor<f32>
  // CHECK-DAG: [[cst3:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[cst4:%.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: [[cst5:%.*]] = arith.constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: [[cst6:%.*]] = arith.constant dense<4.000000e+00> : tensor<f32>
  // CHECK: return [[cst0]], [[cst1]], [[cst2]], [[cst3]], [[cst4]], [[cst5]], [[cst6]]

  %7 = "tfl.abs"(%0) : (tensor<f32>) -> tensor<f32>
  %8 = "tfl.sin"(%1) : (tensor<f32>) -> tensor<f32>
  %9 = "tfl.cos"(%2) : (tensor<f32>) -> tensor<f32>
  %10 = "tfl.log"(%3) : (tensor<f32>) -> tensor<f32>
  %11 = "tfl.sqrt"(%4) : (tensor<f32>) -> tensor<f32>
  %12 = "tfl.rsqrt"(%5) : (tensor<f32>) -> tensor<f32>
  %13 = "tfl.square"(%6) : (tensor<f32>) -> tensor<f32>

  func.return %7, %8, %9, %10, %11, %12, %13 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: @max_with_neg_f32_max_val
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f32>)
func.func @max_with_neg_f32_max_val(%arg0 : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %neg_f32_max = arith.constant dense<-3.40282347E+38> : tensor<f32>
  %0 = "tfl.maximum"(%arg0, %neg_f32_max) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tfl.maximum"(%neg_f32_max, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0, %1 : tensor<f32>, tensor<f32>
  // CHECK: return %[[ARG0]], %[[ARG0]]
}

// CHECK-LABEL: @max_with_neg_inf
func.func @max_with_neg_inf(%arg0 : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %neg_inf = arith.constant dense<0xFF800000> : tensor<f32>
  %0 = "tfl.maximum"(%arg0, %neg_inf) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tfl.maximum"(%neg_inf, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0, %1 : tensor<f32>, tensor<f32>
  // CHECK: return %[[ARG0]], %[[ARG0]]
}

// CHECK-LABEL: @min_with_f32_max_val
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f32>)
func.func @min_with_f32_max_val(%arg0 : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %f32_max = arith.constant dense<3.40282347E+38> : tensor<f32>
  %0 = "tfl.minimum"(%arg0, %f32_max) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tfl.minimum"(%f32_max, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0, %1 : tensor<f32>, tensor<f32>
  // CHECK: return %[[ARG0]], %[[ARG0]]
}

// CHECK-LABEL: @min_with_inf
func.func @min_with_inf(%arg0 : tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %inf = arith.constant dense<0x7F800000> : tensor<f32>
  %0 = "tfl.minimum"(%arg0, %inf) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %1 = "tfl.minimum"(%inf, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0, %1 : tensor<f32>, tensor<f32>
  // CHECK: return %[[ARG0]], %[[ARG0]]
}

// CHECK-LABEL: @max_with_neg_f64_max_val
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f64>)
func.func @max_with_neg_f64_max_val(%arg0 : tensor<f64>) -> (tensor<f64>, tensor<f64>) {
  %neg_f64_max = arith.constant dense<-1.7976931348623157E+308> : tensor<f64>
  %0 = "tfl.maximum"(%arg0, %neg_f64_max) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  %1 = "tfl.maximum"(%neg_f64_max, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %0, %1 : tensor<f64>, tensor<f64>
  // CHECK: return %[[ARG0]], %[[ARG0]]
}

// CHECK-LABEL: @min_with_f64_max_val
// CHECK-SAME: (%[[ARG0:.+]]: tensor<f64>)
func.func @min_with_f64_max_val(%arg0 : tensor<f64>) -> (tensor<f64>, tensor<f64>) {
  %f64_max = arith.constant dense<1.7976931348623157E+308> : tensor<f64>
  %0 = "tfl.minimum"(%arg0, %f64_max) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  %1 = "tfl.minimum"(%f64_max, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %0, %1 : tensor<f64>, tensor<f64>
  // CHECK: return %[[ARG0]], %[[ARG0]]
}

// CHECK-LABEL: @min_dense_splat_int
func.func @min_dense_splat_int() -> tensor<4xi32> {
  %0 = arith.constant dense<[-10, -1, 42, 100]> : tensor<4xi32>
  %1 = arith.constant dense<5> : tensor<4xi32>

  %2 = "tfl.minimum"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>
}

// CHECK:  arith.constant dense<[-10, -1, 5, 5]> : tensor<4xi32>

// CHECK-LABEL: @min_dense_splat_float
func.func @min_dense_splat_float() -> tensor<4xf32> {
  %0 = arith.constant dense<[-10.0, -1.0, 42.0, 100.0]> : tensor<4xf32>
  %1 = arith.constant dense<5.0> : tensor<4xf32>

  %2 = "tfl.minimum"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %2 : tensor<4xf32>
}

// CHECK: arith.constant dense<[-1.000000e+01, -1.000000e+00, 5.000000e+00, 5.000000e+00]> : tensor<4xf32>

// CHECK-LABEL: @min_dense_float
func.func @min_dense_float() -> tensor<2xf32> {
  %0 = arith.constant dense<[-10.0, 10.0]> : tensor<2xf32>
  %1 = arith.constant dense<[5.0, 5.0]> : tensor<2xf32>

  %2 = "tfl.minimum"(%0, %1) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  func.return %2 : tensor<2xf32>
}

// CHECK: arith.constant dense<[-1.000000e+01, 5.000000e+00]> : tensor<2xf32>

// CHECK-LABEL: @max_dense_splat_int
func.func @max_dense_splat_int() -> tensor<4xi32> {
  %0 = arith.constant dense<[-10, -1, 42, 100]> : tensor<4xi32>
  %1 = arith.constant dense<5> : tensor<4xi32>

  %2 = "tfl.maximum"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>
}

// CHECK:  arith.constant dense<[5, 5, 42, 100]> : tensor<4xi32>

// CHECK-LABEL: @max_dense_splat_float
func.func @max_dense_splat_float() -> tensor<4xf32> {
  %0 = arith.constant dense<[-10.0, -1.0, 42.0, 100.0]> : tensor<4xf32>
  %1 = arith.constant dense<5.0> : tensor<4xf32>

  %2 = "tfl.maximum"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %2 : tensor<4xf32>
}

// CHECK: arith.constant dense<[5.000000e+00, 5.000000e+00, 4.200000e+01, 1.000000e+02]> : tensor<4xf32>

// CHECK-LABEL: @max_dense_float
func.func @max_dense_float() -> tensor<2xf32> {
  %0 = arith.constant dense<[-10.0, 10.0]> : tensor<2xf32>
  %1 = arith.constant dense<[5.0, 5.0]> : tensor<2xf32>

  %2 = "tfl.maximum"(%0, %1) {fused_activation_function = "NONE"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  func.return %2 : tensor<2xf32>
}

// CHECK: arith.constant dense<[5.000000e+00, 1.000000e+01]> : tensor<2xf32>

// CHECK-LABEL: @mul_int
func.func @mul_int() -> (tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = arith.constant dense<8> : tensor<i32>
  %1 = arith.constant dense<1> : tensor<i32>

  %2 = arith.constant dense< 4> : tensor<4xi32>
  %3 = arith.constant dense<-2> : tensor<4xi32>

  // CHECK-DAG: [[cst0:%.*]] = arith.constant dense<8> : tensor<i32>
  // CHECK-DAG: [[cst1:%.*]] = arith.constant dense<-16> : tensor<4xi32>
  // CHECK-DAG: [[cst2:%.*]] = arith.constant dense<4> : tensor<4xi32>
  // CHECK-DAG: [[cst3:%.*]] = arith.constant dense<-8> : tensor<4xi32>
  // CHECK: return [[cst0]], [[cst1]], [[cst2]], [[cst3]]

  %5 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<  i32>) -> tensor<  i32>
  %6 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<4xi32>) -> tensor<4xi32>
  %7 = "tfl.mul"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<  i32>) -> tensor<4xi32>
  %8 = "tfl.mul"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %5, %6, %7, %8 : tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @add_dense_splat_int
func.func @add_dense_splat_int() -> tensor<4xi32> {
  %0 = arith.constant dense<[-10, -1, 42, 100]> : tensor<4xi32>
  %1 = arith.constant dense< 5> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>

// CHECK:  %[[CST:.*]] = arith.constant dense<[-5, 4, 47, 105]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_splat_dense_int
func.func @add_splat_dense_int() -> tensor<4xi32> {
  %0 = arith.constant dense< 5> : tensor<4xi32>
  %1 = arith.constant dense<[-10, -1, 42, 100]> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>

// CHECK:  %[[CST:.*]] = arith.constant dense<[-5, 4, 47, 105]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_int_same_shape
func.func @add_dense_dense_int_same_shape() -> tensor<4xi32> {
  %0 = arith.constant dense<[15, 23, -44, -2]> : tensor<4xi32>
  %1 = arith.constant dense<[-10, -1, 42, 100]> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>

// CHECK:  %[[CST:.*]] = arith.constant dense<[5, 22, -2, 98]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_int_trailing_dim
func.func @add_dense_dense_int_trailing_dim() -> (tensor<2x2xi32>, tensor<2x2x2xi32>, tensor<2x2x2xi32>) {
  %cst_0 = arith.constant dense<[10, 20]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %cst_2 = arith.constant dense<[[[1, 1], [2, 2]], [[3, 3], [4, 4]]]> : tensor<2x2x2xi32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<    2xi32>, tensor<  2x2xi32>) -> tensor<  2x2xi32>
  %1 = "tfl.add"(%cst_2, %cst_1) {fused_activation_function = "NONE"} : (tensor<2x2x2xi32>, tensor<  2x2xi32>) -> tensor<2x2x2xi32>
  %2 = "tfl.add"(%cst_0, %cst_2) {fused_activation_function = "NONE"} : (tensor<    2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi32>

  func.return %0, %1, %2 : tensor<2x2xi32>, tensor<2x2x2xi32>, tensor<2x2x2xi32>

// CHECK-DAG:  %[[CST:.*]] = arith.constant dense<{{\[\[}}11, 22], [13, 24]]> : tensor<2x2xi32>
// CHECK-DAG:  %[[CST_0:.*]]  = arith.constant dense<{{\[\[\[}}2, 3], [5, 6]], {{\[\[}}4, 5], [7, 8]]]> : tensor<2x2x2xi32>
// CHECK-DAG:  %[[CST_1:.*]]  = arith.constant dense<{{\[\[\[}}11, 21], [12, 22]], {{\[\[}}13, 23], [14, 24]]]> : tensor<2x2x2xi32>
// CHECK:  return %[[CST]], %[[CST_0]], %[[CST_1]]
}

// CHECK-LABEL: @add_dense_dense_int_mixing_1_n
func.func @add_dense_dense_int_mixing_1_n() -> tensor<2x2xi32> {
  %cst_0 = arith.constant dense<[[1, 2]]> : tensor<1x2xi32>
  %cst_1 = arith.constant dense<[[3], [4]]> : tensor<2x1xi32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>

  func.return %0 : tensor<2x2xi32>
// CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}4, 5], [5, 6]]> : tensor<2x2xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_splat_float
func.func @add_dense_splat_float() -> tensor<4xf32> {
  %0 = arith.constant dense<[-10.0, -1.5, 42.0, 7.25]> : tensor<4xf32>
  %1 = arith.constant dense< 3.5> : tensor<4xf32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %2 : tensor<4xf32>

// CHECK:  %[[CST:.*]] = arith.constant dense<[-6.500000e+00, 2.000000e+00, 4.550000e+01, 1.075000e+01]> : tensor<4xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_splat_dense_float
func.func @add_splat_dense_float() -> tensor<4xf32> {
  %0 = arith.constant dense< 3.5> : tensor<4xf32>
  %1 = arith.constant dense<[-10.0, -1.5, 42.0, 7.25]> : tensor<4xf32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %2 : tensor<4xf32>

// CHECK:  %[[CST:.*]] = arith.constant dense<[-6.500000e+00, 2.000000e+00, 4.550000e+01, 1.075000e+01]> : tensor<4xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_float_same_shape
func.func @add_dense_dense_float_same_shape() -> (tensor<4xf32>) {
  %0 = arith.constant dense<[1.5, 2.3, -4.4, -2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[-10.4, -1.3, 42.4, 100.0]> : tensor<4xf32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %2 : tensor<4xf32>

// CHECK:  %[[CST:.*]] = arith.constant dense<[-8.89999961, 1.000000e+00, 3.800000e+01, 9.800000e+01]> : tensor<4xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_float_trailing_dim
func.func @add_dense_dense_float_trailing_dim() -> (tensor<2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>) {
  %cst_0 = arith.constant dense<[1., -4.]> : tensor<2xf32>
  %cst_1 = arith.constant dense<[[-5.5, 1.5], [7.5, -4.5]]> : tensor<2x2xf32>
  %cst_2 = arith.constant dense<[[[1., 1.], [2., 2.]], [[3., 3.], [4., 4.]]]> : tensor<2x2x2xf32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<    2xf32>, tensor<  2x2xf32>) -> tensor<  2x2xf32>
  %1 = "tfl.add"(%cst_2, %cst_1) {fused_activation_function = "NONE"} : (tensor<2x2x2xf32>, tensor<  2x2xf32>) -> tensor<2x2x2xf32>
  %2 = "tfl.add"(%cst_0, %cst_2) {fused_activation_function = "NONE"} : (tensor<    2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>

  func.return %0, %1, %2 : tensor<2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>

// CHECK-DAG:  %[[CST:.*]] = arith.constant dense<{{\[\[}}-4.500000e+00, -2.500000e+00], [8.500000e+00, -8.500000e+00]]> : tensor<2x2xf32>
// CHECK-DAG:  %[[CST_0:.*]]  = arith.constant dense<{{\[\[\[}}-4.500000e+00, 2.500000e+00], [9.500000e+00, -2.500000e+00]], {{\[\[}}-2.500000e+00, 4.500000e+00], [1.150000e+01, -5.000000e-01]]]> : tensor<2x2x2xf32>
// CHECK-DAG:  %[[CST_1:.*]]  = arith.constant dense<{{\[\[\[}}2.000000e+00, -3.000000e+00], [3.000000e+00, -2.000000e+00]], {{\[\[}}4.000000e+00, -1.000000e+00], [5.000000e+00, 0.000000e+00]]]> : tensor<2x2x2xf32>
// CHECK:  return %[[CST]], %[[CST_0]], %[[CST_1]]
}

// CHECK-LABEL: @add_dense_dense_float_mixfng_1_n
func.func @add_dense_dense_float_mixfng_1_n() -> tensor<2x2xf32> {
  %cst_0 = arith.constant dense<[[1.5, -2.5]]> : tensor<1x2xf32>
  %cst_1 = arith.constant dense<[[-3.], [4.]]> : tensor<2x1xf32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>

  func.return %0 : tensor<2x2xf32>

// CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}-1.500000e+00, -5.500000e+00], [5.500000e+00, 1.500000e+00]]> : tensor<2x2xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_zero
func.func @add_zero(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<4xi32>
  %zero_float = arith.constant dense<0.0> : tensor<4xf32>

  // CHECK-NOT: tfl.add
  // CHECK: return %arg0, %arg1

  %0 = "tfl.add"(%arg0, %zero_int) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.add"(%arg1, %zero_float) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @add_zero_broadcast
func.func @add_zero_broadcast(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<1xi32>
  %zero_float = arith.constant dense<0.0> : tensor<1xf32>

  // CHECK-NOT: tfl.add
  // CHECK: return %arg0, %arg1

  %0 = "tfl.add"(%arg0, %zero_int) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1 = "tfl.add"(%arg1, %zero_float) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @add_zero_dynamic
func.func @add_zero_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<?xf32>) -> (tensor<?xi32>, tensor<?xf32>) {
  %zero_int = arith.constant dense<0> : tensor<1xi32>
  %zero_float = arith.constant dense<0.0> : tensor<1xf32>

  // CHECK-NOT: tfl.add
  // CHECK: return %arg0, %arg1

  %0 = "tfl.add"(%arg0, %zero_int) {fused_activation_function = "NONE"} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %1 = "tfl.add"(%arg1, %zero_float) {fused_activation_function = "NONE"} : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>

  func.return %0, %1 : tensor<?xi32>, tensor<?xf32>
}

// CHECK-LABEL: @add_zero_lhs
func.func @add_zero_lhs(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %zero_int = arith.constant dense<0> : tensor<4xi32>
  %zero_float = arith.constant dense<0.0> : tensor<4xf32>

  // CHECK-NOT: tfl.add
  // CHECK: return %arg0, %arg1

  %0 = "tfl.add"(%zero_int, %arg0) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.add"(%zero_float, %arg1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @add_zero_quant
func.func @add_zero_quant(%arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>> {
  %zero = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<u8:f32, 1.0>>, value = dense<0> : tensor<32xi8>} : () -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  // CHECK: %[[ADD:.*]] = tfl.add
  // CHECK: return %[[ADD]]

  %0 = "tfl.add"(%zero, %arg0) {fused_activation_function = "NONE"} : (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  func.return %0 : tensor<32x!quant.uniform<u8:f32, 1.0>>
}

// CHECK-LABEL: @rank
func.func @rank() -> tensor<1xi32> {
  %cst = arith.constant dense<[[1], [2]]> : tensor<2x1xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.rank"(%cst) : (tensor<2x1xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @rank_input_known_rank
func.func @rank_input_known_rank(%arg0 : tensor<2x1xi32>) -> tensor<1xi32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<2> : tensor<1xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.rank"(%arg0) : (tensor<2x1xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}

// CHECK-LABEL: @reshape
func.func @reshape() -> tensor<4xi32> {
  %input = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %shape = arith.constant dense<[4]> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.reshape"(%input, %shape) : (tensor<2x2xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// CHECK-LABEL: @reshape_dynamic_output
func.func @reshape_dynamic_output() -> tensor<?xi32> {
  %input = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %shape = arith.constant dense<[4]> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi32>}> : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.reshape"(%input, %shape) : (tensor<2x2xi32>, tensor<1xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}


// CHECK-LABEL: @pseudo_const
func.func @pseudo_const() -> tensor<i32> {
  // CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<i32>
  // CHECK: return %[[CST]]
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}


// CHECK-LABEL: @range_int
func.func @range_int() -> tensor<?xi32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<4> : tensor<i32>
  %cst_2 = arith.constant dense<1> : tensor<i32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[0, 1, 2, 3]> : tensor<4xi32>}> : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: @range_float
func.func @range_float() -> tensor<?xf32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %cst_1 = arith.constant dense<4.0> : tensor<f32>
  %cst_2 = arith.constant dense<1.0> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>}> : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}


// CHECK-LABEL: @range_float_neg_delta
func.func @range_float_neg_delta() -> tensor<?xf32> {
  %cst = arith.constant dense<0.0> : tensor<f32>
  %cst_1 = arith.constant dense<-4.0> : tensor<f32>
  %cst_2 = arith.constant dense<-1.0> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[0.000000e+00, -1.000000e+00, -2.000000e+00, -3.000000e+00]> : tensor<4xf32>}> : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @range_float_nonzero_base
func.func @range_float_nonzero_base() -> tensor<?xf32> {
  %cst = arith.constant dense<2.0> : tensor<f32>
  %cst_1 = arith.constant dense<7.0> : tensor<f32>
  %cst_2 = arith.constant dense<1.5> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[2.000000e+00, 3.500000e+00, 5.000000e+00, 6.500000e+00]> : tensor<4xf32>}> : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: @transpose_no_fold
func.func @transpose_no_fold(%arg0 : tensor<2xi32>) -> tensor<2x2xi32> {
  %cst = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>

  // CHECK: tfl.transpose
  %0 = "tfl.transpose"(%cst, %arg0) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}


// CHECK-LABEL: @transpose_1d
// Basic 1D identity
func.func @transpose_1d() -> tensor<3xi32> {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %cst_perm = arith.constant dense<0> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<{{\[}}1, 2, 3]> : tensor<3xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<3xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @transpose_dynamic
func.func @transpose_dynamic() -> tensor<?xi32> {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %cst_perm = arith.constant dense<0> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<{{\[}}1, 2, 3]> : tensor<3xi32>}> : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<3xi32>, tensor<1xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: @transpose_2d
func.func @transpose_2d() -> tensor<2x2xi32> {
  %cst = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %cst_perm = arith.constant dense<[1, 0]> : tensor<2xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}0, 2], {{\[}}1, 3]]> : tensor<2x2xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_2d_splat
func.func @transpose_2d_splat() -> tensor<3x2xi32> {
  %cst = arith.constant dense<0> : tensor<2x3xi32>
  %cst_perm = arith.constant dense<[1, 0]> : tensor<2xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<3x2xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x3xi32>, tensor<2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// CHECK-LABEL: @transpose_2d_identity
func.func @transpose_2d_identity() -> tensor<2x2xi32> {
  %cst = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %cst_perm = arith.constant dense<[0, 1]> : tensor<2xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}0, 1], {{\[}}2, 3]]> : tensor<2x2xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_3d
// A test case adopted from TransposeTest.Test3DInputConstTensor in
// tensorflow/lite/kernels/transpose_test.cc
func.func @transpose_3d() -> tensor<4x2x3xi32> {
  %cst = arith.constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi32>
  %cst_perm = arith.constant dense<[2, 0, 1]> : tensor<3xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[\[}}0, 4, 8], {{\[}}12, 16, 20]], {{\[\[}}1, 5, 9], {{\[}}13, 17, 21]], {{\[\[}}2, 6, 10], {{\[}}14, 18, 22]], {{\[\[}}3, 7, 11], {{\[}}15, 19, 23]]]> : tensor<4x2x3xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x3x4xi32>, tensor<3xi32>) -> tensor<4x2x3xi32>
  func.return %0 : tensor<4x2x3xi32>
}

// CHECK-LABEL: @ConstantFoldBinaryOpDynamicOutput
func.func @ConstantFoldBinaryOpDynamicOutput() -> tensor<?xi32> {
  %cst = arith.constant dense<10> : tensor<i32>
  %cst_0 = "tfl.pseudo_const"() {value = dense<[5, 10]> : tensor<2xi32>} : () -> tensor<?xi32>
  %87 = "tfl.sub"(%cst_0, %cst) {fused_activation_function = "NONE"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %87 : tensor<?xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[-5, 0]> : tensor<2xi32>}> : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_int_same_shape_dynamic
func.func @add_dense_dense_int_same_shape_dynamic() -> tensor<?xi32> {
  %0 = arith.constant dense<[15, 23, -44, -2]> : tensor<4xi32>
  %1 = arith.constant dense<[-10, -1, 42, 100]> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<?xi32>

  func.return %2 : tensor<?xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() <{value = dense<[5, 22, -2, 98]> : tensor<4xi32>}> : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @concat_2_tensors_1_empty
func.func @concat_2_tensors_1_empty() -> tensor<2xi32> {
  %1 = arith.constant dense<1> : tensor<2xi32>
  %2 = arith.constant dense<[]> : tensor<0xi32>
  %3 = "tfl.concatenation"(%1, %2) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<0xi32>) -> tensor<2xi32>
  func.return %3 : tensor<2xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<2xi32>
  // CHECK: return %[[CST]] : tensor<2xi32>
}

// CHECK-LABEL: @concat_3_tensors_1_empty
func.func @concat_3_tensors_1_empty() -> tensor<?xi32> {
  %0 = arith.constant dense<1> : tensor<2xi32>
  %1 = arith.constant dense<1> : tensor<2xi32>
  %2 = arith.constant dense<[]> : tensor<0xi32>
  %3 = "tfl.concatenation"(%0, %1, %2) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>, tensor<0xi32>) -> tensor<?xi32>
  func.return %3 : tensor<?xi32>

  // CHECK: %0 = "tfl.concatenation"(%[[CST]], %[[CST]]) <{axis = 0 : i32, fused_activation_function = "NONE"}>
  // CHECK: return %0 : tensor<?xi32>
}

// CHECK-LABEL: @concatConstantTensorsFirstDim
func.func @concatConstantTensorsFirstDim() -> tensor<2x2x3xi32> {
  %cst_0 = arith.constant dense<0> : tensor<1x2x3xi32>
  %cst_1 = arith.constant dense<1> : tensor<1x2x3xi32>
  %0 = "tfl.concatenation"(%cst_0, %cst_1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> tensor<2x2x3xi32>
  func.return %0 : tensor<2x2x3xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<[{{\[}}{{\[}}0, 0, 0], {{\[}}0, 0, 0]], {{\[}}{{\[}}1, 1, 1], {{\[}}1, 1, 1]]]> : tensor<2x2x3xi32>
  // CHECK-NOT: constant-dense
  // CHECK-NOT: "tfl.concatenation"
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @concatConstantTensorsMiddleDim
func.func @concatConstantTensorsMiddleDim() -> tensor<1x4x3xi32> {
  %cst_0 = arith.constant dense<0> : tensor<1x2x3xi32>
  %cst_1 = arith.constant dense<1> : tensor<1x2x3xi32>
  %0 = "tfl.concatenation"(%cst_0, %cst_1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> tensor<1x4x3xi32>
  func.return %0 : tensor<1x4x3xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<[{{\[}}{{\[}}0, 0, 0], {{\[}}0, 0, 0], {{\[}}1, 1, 1], {{\[}}1, 1, 1]]]> : tensor<1x4x3xi32>
  // CHECK-NOT: constant-dense
  // CHECK-NOT: "tfl.concatenation"
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @concatConstantTensorsLastDim
func.func @concatConstantTensorsLastDim() -> tensor<1x2x6xi32> {
  %cst_0 = arith.constant dense<0> : tensor<1x2x3xi32>
  %cst_1 = arith.constant dense<1> : tensor<1x2x3xi32>
  %0 = "tfl.concatenation"(%cst_0, %cst_1) {axis = 2 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> tensor<1x2x6xi32>
  func.return %0 : tensor<1x2x6xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<[{{\[}}{{\[}}0, 0, 0, 1, 1, 1], {{\[}}0, 0, 0, 1, 1, 1]]]> : tensor<1x2x6xi32>
  // CHECK-NOT: constant-dense
  // CHECK-NOT: "tfl.concatenation"
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @div_dense_dense_float_mixfng_1_n
func.func @div_dense_dense_float_mixfng_1_n() -> tensor<2x2xf32> {
  %cst_0 = arith.constant dense<[[1.5, -2.5]]> : tensor<1x2xf32>
  %cst_1 = arith.constant dense<[[-3.], [4.]]> : tensor<2x1xf32>

  %0 = "tfl.div"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>

  func.return %0 : tensor<2x2xf32>

// CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}-5.000000e-01, 0.833333313], [3.750000e-01, -6.250000e-01]]> : tensor<2x2xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @div_dense_different_rank
func.func @div_dense_different_rank() -> tensor<1x2x2xf32> {
  %cst_0 = arith.constant dense<[[[1.0],[2.0]]]> : tensor<1x2x1xf32>
  %cst_1 = arith.constant dense<[[2.0, 3.0]]> : tensor<1x2xf32>

  %0 = "tfl.div"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2x1xf32>, tensor<1x2xf32>) -> tensor<1x2x2xf32>

  func.return %0 : tensor<1x2x2xf32>

// CHECK: %[[CST:.*]] = arith.constant dense<[{{\[}}{{\[}}5.000000e-01, 0.333333343], [1.000000e+00, 0.666666686]]]> : tensor<1x2x2xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @div_one
func.func @div_one(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %one_int = arith.constant dense<1> : tensor<4xi32>
  %one_float = arith.constant dense<1.0> : tensor<4xf32>

  // CHECK-NOT: tfl.div
  // CHECK: return %arg0, %arg1

  %0 = "tfl.div"(%arg0, %one_int) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %1 = "tfl.div"(%arg1, %one_float) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0, %1 : tensor<4xi32>, tensor<4xf32>
}

// CHECK-LABEL: @div_one_quant
func.func @div_one_quant(%arg0: tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>> {
  %one = "tfl.pseudo_qconst"() {qtype = tensor<32x!quant.uniform<u8:f32, 1.0>>, value = dense<1> : tensor<32xi8>} : () -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  // CHECK: %[[DIV:.*]] = tfl.div
  // CHECK: return %[[DIV]]

  %0 = "tfl.div"(%arg0, %one) {fused_activation_function = "NONE"} : (tensor<32x!quant.uniform<u8:f32, 1.0>>, tensor<32x!quant.uniform<u8:f32, 1.0>>) -> tensor<32x!quant.uniform<u8:f32, 1.0>>

  func.return %0 : tensor<32x!quant.uniform<u8:f32, 1.0>>
}

// CHECK-LABEL: @rsqrt_bf16
func.func @rsqrt_bf16() -> tensor<bf16> {
  %cst = arith.constant dense<4.0> : tensor<bf16>
  %0 = "tfl.rsqrt"(%cst) : (tensor<bf16>) -> tensor<bf16>
  func.return %0 : tensor<bf16>

// CHECK: %[[CST:.*]] = arith.constant dense<5.000000e-01> : tensor<bf16>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_i64_to_i32
func.func @cast_i64_to_i32() -> tensor<5xi32> {
  %cst = arith.constant dense<[-1, 0, 1, 2147483647, 2147483648]> : tensor<5xi64>
  %0 = "tfl.cast"(%cst) : (tensor<5xi64>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>

// CHECK: %[[CST:.*]] = arith.constant dense<[-1, 0, 1, 2147483647, -2147483648]> : tensor<5xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_i32_to_ui8
func.func @cast_i32_to_ui8() -> tensor<6xui8> {
  %cst = arith.constant dense<[0, -1, 256, 127, -128, -129]> : tensor<6xi32>
  %0 = "tfl.cast"(%cst) : (tensor<6xi32>) -> tensor<6xui8>
  func.return %0 : tensor<6xui8>

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 255, 0, 127, 128, 127]> : tensor<6xui8>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_ui8_to_i8
func.func @cast_ui8_to_i8() -> tensor<4xi8> {
  %cst = arith.constant dense<[0, 255, 127, 128]> : tensor<4xui8>
  %0 = "tfl.cast"(%cst) : (tensor<4xui8>) -> tensor<4xi8>
  func.return %0 : tensor<4xi8>

// CHECK: %[[CST:.*]] = arith.constant dense<[0, -1, 127, -128]> : tensor<4xi8>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_i8_to_i32
func.func @cast_i8_to_i32() -> tensor<4xi32> {
  %cst = arith.constant dense<[0, 128, -1, -128]> : tensor<4xi8>
  %0 = "tfl.cast"(%cst) : (tensor<4xi8>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>

// CHECK: %[[CST:.*]] = arith.constant dense<[0, -128, -1, -128]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_ui8_to_i32
func.func @cast_ui8_to_i32() -> tensor<4xi32> {
  %cst = arith.constant dense<[0, 128, 129, 255]> : tensor<4xui8>
  %0 = "tfl.cast"(%cst) : (tensor<4xui8>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 128, 129, 255]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_identity
func.func @cast_identity(%arg0 : tensor<7xf32>) -> tensor<7xf32> {
  %0 = "tfl.cast"(%arg0) : (tensor<7xf32>) -> tensor<7xf32>
  func.return %0 : tensor<7xf32>
  // CHECK: return %arg0 : tensor<7xf32>
}

// CHECK-LABEL: @cast_i1_to_i8
func.func @cast_i1_to_i8() -> tensor<2xi8> {
  %cst = arith.constant dense<[false, true]> : tensor<2xi1>
  %0 = "tfl.cast"(%cst) : (tensor<2xi1>) -> tensor<2xi8>
  func.return %0 : tensor<2xi8>

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1]> : tensor<2xi8>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_i1_to_ui8
func.func @cast_i1_to_ui8() -> tensor<2xui8> {
  %cst = arith.constant dense<[false, true]> : tensor<2xi1>
  %0 = "tfl.cast"(%cst) : (tensor<2xi1>) -> tensor<2xui8>
  func.return %0 : tensor<2xui8>

// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1]> : tensor<2xui8>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_i8_to_i1
func.func @cast_i8_to_i1() -> tensor<4xi1> {
  %cst = arith.constant dense<[0, 1, 2, -1]> : tensor<4xi8>
  %0 = "tfl.cast"(%cst) : (tensor<4xi8>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>

// CHECK: %[[CST:.*]] = arith.constant dense<[false, true, true, true]> : tensor<4xi1>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_ui8_to_i1
func.func @cast_ui8_to_i1() -> tensor<4xi1> {
  %cst = arith.constant dense<[0, 127, 128, 255]> : tensor<4xui8>
  %0 = "tfl.cast"(%cst) : (tensor<4xui8>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>

// CHECK: %[[CST:.*]] = arith.constant dense<[false, true, true, true]> : tensor<4xi1>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @cast_f32_to_i32
func.func @cast_f32_to_i32() -> tensor<8xi32> {
  %cst = arith.constant dense<[-1.0, 0.0, 1.5, 0.99, 1.175494351e-38, 3.402823466e+38, -3.402823466e+38, -1.175494351e-38]> : tensor<8xf32>
  %0 = "tfl.cast"(%cst) : (tensor<8xf32>) -> tensor<8xi32>
  func.return %0 : tensor<8xi32>
}

// CHECK: %cst = arith.constant dense<[-1, 0, 1, 0, 0, 2147483647, -2147483648, 0]> : tensor<8xi32>

// CHECK-LABEL: @cast_f32_to_i64
func.func @cast_f32_to_i64() -> tensor<4xi64> {
  %cst = arith.constant dense<[-1.0, 0.0, 1.5, 0.99]> : tensor<4xf32>
  %0 = "tfl.cast"(%cst) : (tensor<4xf32>) -> tensor<4xi64>
  func.return %0 : tensor<4xi64>
}

// CHECK: %cst = arith.constant dense<[-1, 0, 1, 0]> : tensor<4xi64>

// CHECK-LABEL: @cast_i32_to_f32
func.func @cast_i32_to_f32() -> tensor<5xf32> {
  %cst = arith.constant dense<[-1, 0, 2, 2147483647, -2147483648]> : tensor<5xi32>
  %0 = "tfl.cast"(%cst) : (tensor<5xi32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

// CHECK: %cst = arith.constant dense<[-1.000000e+00, 0.000000e+00, 2.000000e+00, 2.14748365E+9, -2.14748365E+9]> : tensor<5xf32>

// CHECK-LABEL: @cast_bool_to_f32
func.func @cast_bool_to_f32() -> tensor<2xf32> {
  %cst = arith.constant dense<[true, false]> : tensor<2xi1>
  %0 = "tfl.cast"(%cst) : (tensor<2xi1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: %cst = arith.constant dense<[1.000000e+00, 0.000000e+00]> : tensor<2xf32>

// CHECK-LABEL: @cast_f64_to_f32
func.func @cast_f64_to_f32() -> tensor<4xf32> {
  %cst = arith.constant dense<[-1.0, 0.0, 1.5, 100.0]> : tensor<4xf64>
  %0 = "tfl.cast"(%cst) : (tensor<4xf64>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: %cst = arith.constant dense<[-1.000000e+00, 0.000000e+00, 1.500000e+00, 1.000000e+02]> : tensor<4xf32>

// CHECK-LABEL: @cast_f32_to_f64
func.func @cast_f32_to_f64() -> tensor<4xf64> {
  %cst = arith.constant dense<[-1.0, 0.0, 1.5, 100.0]> : tensor<4xf32>
  %0 = "tfl.cast"(%cst) : (tensor<4xf32>) -> tensor<4xf64>
  func.return %0 : tensor<4xf64>
}

// CHECK: %cst = arith.constant dense<[-1.000000e+00, 0.000000e+00, 1.500000e+00, 1.000000e+02]> : tensor<4xf64>

// CHECK-LABEL: @ConstantFoldFullyConnectedSmall
func.func @ConstantFoldFullyConnectedSmall() -> tensor<3xf32> {
  %cst_input = arith.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %cst_weights = arith.constant dense<[[5.0, 7.0], [11.0, 13.0], [17.0, 19.0]]> : tensor<3x2xf32>
  %cst_bias = arith.constant dense<[23.0, 29.0, 31.0]> : tensor<3xf32>

  %0 = "tfl.fully_connected" (%cst_input, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2xf32>, tensor<3x2xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>

  // [54, 90, 122]
  // CHECK: %[[CST:.*]] = arith.constant dense<[5.400000e+01, 9.000000e+01, 1.220000e+02]> : tensor<3xf32>
  // CHECK:  return %[[CST]]
}

// CHECK-LABEL: @ConstantFoldFullyConnectedLarge
func.func @ConstantFoldFullyConnectedLarge() -> tensor<1024xf32> {
  %cst_input = arith.constant dense<1.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<2.0> : tensor<1024x512xf32>
  %cst_bias = arith.constant dense<4.0> : tensor<1024xf32>

  %0 = "tfl.fully_connected" (%cst_input, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<1024xf32>

  func.return %0 : tensor<1024xf32>

  // 1.0 * 2.0 * 512 + 4.0 = 1028.0
  // CHECK: %[[CST:.*]] = arith.constant dense<1.028000e+03> : tensor<1024xf32>
  // CHECK:  return %[[CST]]
}

// CHECK-LABEL: @ConstantFoldFullyConnectedNoBias
func.func @ConstantFoldFullyConnectedNoBias() -> tensor<1024xf32> {
  %cst_input = arith.constant dense<1.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<2.0> : tensor<1024x512xf32>
  %cst_bias = "tfl.no_value"() {value = unit} : () -> none

  %0 = "tfl.fully_connected" (%cst_input, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<512xf32>, tensor<1024x512xf32>, none) -> tensor<1024xf32>

  func.return %0 : tensor<1024xf32>

  // 1.0 * 2.0 * 512 = 1024.0
  // CHECK: %[[CST:.*]] = arith.constant dense<1.024000e+03> : tensor<1024xf32>
  // CHECK:  return %[[CST]]
}

// CHECK-LABEL: @NoFoldFullyConnectedNonFloat
func.func @NoFoldFullyConnectedNonFloat() -> tensor<1024xf32> {
  %cst_input = arith.constant dense<1.0> : tensor<512xf32>
  %cst_weights = arith.constant dense<2> : tensor<1024x512xi8>
  %cst_bias = arith.constant dense<4.0> : tensor<1024xf32>

  %0 = "tfl.fully_connected" (%cst_input, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<512xf32>, tensor<1024x512xi8>, tensor<1024xf32>) -> tensor<1024xf32>

  func.return %0 : tensor<1024xf32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<512xf32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<2> : tensor<1024x512xi8>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<4.000000e+00> : tensor<1024xf32>
  // CHECK: %[[VAL:.*]] = "tfl.fully_connected"(%[[CST]], %[[CST_0]], %[[CST_1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<512xf32>, tensor<1024x512xi8>, tensor<1024xf32>) -> tensor<1024xf32>
  // CHECK: return %[[VAL]] : tensor<1024xf32>
}

// CHECK-LABEL: @NoFoldFullyConnectedHighRank
func.func @NoFoldFullyConnectedHighRank() -> tensor<2x1024xf32> {
  %cst_input = arith.constant dense<1.0> : tensor<2x512xf32>
  %cst_weights = arith.constant dense<2.0> : tensor<1024x512xf32>
  %cst_bias = arith.constant dense<4.0> : tensor<1024xf32>

  %0 = "tfl.fully_connected" (%cst_input, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>

  func.return %0 : tensor<2x1024xf32>
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<2x512xf32>
  // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<2.000000e+00> : tensor<1024x512xf32>
  // CHECK-DAG: %[[CST_1:.*]] = arith.constant dense<4.000000e+00> : tensor<1024xf32>
  // CHECK: %[[VAL:.*]] = "tfl.fully_connected"(%[[CST]], %[[CST_0]], %[[CST_1]]) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<2x512xf32>, tensor<1024x512xf32>, tensor<1024xf32>) -> tensor<2x1024xf32>
  // CHECK: return %[[VAL]] : tensor<2x1024xf32>
}

// CHECK-LABEL: @ConstantFoldFullyConnectedCheckPrecision
func.func @ConstantFoldFullyConnectedCheckPrecision() -> tensor<1xf32> {
  %cst_input = arith.constant dense<1.0> : tensor<4xf32>
  %cst_weights = arith.constant dense<[[1.0, 1.0e38, 1.0, -1.0e38]]> : tensor<1x4xf32>
  %cst_bias = arith.constant dense<0.0> : tensor<1xf32>

  %0 = "tfl.fully_connected" (%cst_input, %cst_weights, %cst_bias) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4xf32>, tensor<1x4xf32>, tensor<1xf32>) -> tensor<1xf32>

  func.return %0 : tensor<1xf32>
  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<1xf32>
  // CHECK:  return %[[CST]]
}

// CHECK-LABEL: fully_connected_with_unit_dim
func.func @fully_connected_with_unit_dim() -> tensor<1x5xf32> {
  %0 = "tfl.pseudo_const"() <{value = dense<1.0> : tensor<1x5xf32>}> : () -> tensor<1x5xf32>
  %1 = "tfl.pseudo_const"() <{value = dense<1.0> : tensor<5x5xf32>}> : () -> tensor<5x5xf32>
  %2 = "tfl.pseudo_const"() <{value = dense<1.0> : tensor<1x5xf32>}> : () -> tensor<1x5xf32>
  %3 = "tfl.fully_connected"(%0, %1, %2) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<1x5xf32>, tensor<5x5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
  return %3 : tensor<1x5xf32>
}

// CHECK:     %cst = arith.constant dense<6.000000e+00> : tensor<1x5xf32>
// CHECK-NOT: fully_connected

// CHECK-LABEL: @ShapeOpI32
func.func @ShapeOpI32(%arg0 : tensor<576x72xf32>) -> tensor<2xi32> {
  %0 = "tfl.shape"(%arg0) : (tensor<576x72xf32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
  // CHECK: %[[CST:.*]] = arith.constant dense<[576, 72]> : tensor<2xi32>
  // CHECK:  return %[[CST]]
}

// CHECK-LABEL: @ShapeOpI64
func.func @ShapeOpI64(%arg0 : tensor<576x72xf32>) -> tensor<2xi64> {
  %0 = "tfl.shape"(%arg0) : (tensor<576x72xf32>) -> tensor<2xi64>
  func.return %0 : tensor<2xi64>
  // CHECK: %[[CST:.*]] = arith.constant dense<[576, 72]> : tensor<2xi64>
  // CHECK:  return %[[CST]]
}

// CHECK-LABEL: @ConstFoldStridedSlice
func.func @ConstFoldStridedSlice(%arg0 : tensor<15600xf32>) -> tensor<15600xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<15600> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tfl.pseudo_const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %2 = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %3 = "tfl.strided_slice"(%arg0, %1, %0, %2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<15600xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<15600xf32>
  func.return %3 : tensor<15600xf32>
  // CHECK:  return %arg0
}

func.func @ConstFoldStridedSliceMultiDims(%arg0 : tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<[10, 10, 10]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tfl.pseudo_const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tfl.strided_slice"(%arg0, %1, %0, %2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<10x10x10xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<10x10x10xf32>
  func.return %3 : tensor<10x10x10xf32>
  // CHECK:  return %arg0
}

func.func @NotFoldStridedSlice(%arg0 : tensor<10x10x10xf32>) -> tensor<9x9x9xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<[9, 9, 9]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi32>} : () -> tensor<3xi32>
  %2 = "tfl.pseudo_const"() {value = dense<1> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tfl.strided_slice"(%arg0, %1, %0, %2) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32, offset = false} : (tensor<10x10x10xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<9x9x9xf32>
  func.return %3 : tensor<9x9x9xf32>
  // CHECK: %[[STRIDED_SLICE:.*]] = "tfl.strided_slice"
  // CHECK:  return %[[STRIDED_SLICE]]
}

func.func @ConstFoldPad(%arg0: tensor<15600xf32>) -> tensor<15600xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %1 = "tfl.pad"(%arg0, %0) : (tensor<15600xf32>, tensor<1x2xi32>) -> tensor<15600xf32>
  func.return %1 : tensor<15600xf32>
  // CHECK:  return %arg0
}

func.func @ConstFoldPadV2(%arg0: tensor<15600xf32>) -> tensor<15600xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %1 = "tfl.pseudo_const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "tfl.padv2"(%arg0, %0, %1) : (tensor<15600xf32>, tensor<1x2xi32>, tensor<f32>) -> tensor<15600xf32>
  func.return %2 : tensor<15600xf32>
  // CHECK:  return %arg0
}

// CHECK-LABEL: @ConstFoldEmbeddingLookup
func.func @ConstFoldEmbeddingLookup() -> (tensor<5x2xf32>, tensor<3x2x2xf32>) {
  %index0 = "tfl.pseudo_const"() {value = dense<[2, 1, 0, 0, 2]> : tensor<5xi32>} : () -> tensor<5xi32>
  %index1 = "tfl.pseudo_const"() {value = dense<[0, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
  %value0 = "tfl.pseudo_const"() {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  %value1 = "tfl.pseudo_const"() {value = dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>} : () -> tensor<2x2x2xf32>
  %lookup0 = "tfl.embedding_lookup"(%index0, %value0) : (tensor<5xi32>, tensor<3x2xf32>) -> tensor<5x2xf32>
  %lookup1 = "tfl.embedding_lookup"(%index1, %value1) : (tensor<3xi32>, tensor<2x2x2xf32>) -> tensor<3x2x2xf32>
  func.return %lookup0, %lookup1 : tensor<5x2xf32>, tensor<3x2x2xf32>

  // CHECK-DAG: %[[LOOKUP0:.*]] = arith.constant dense<{{\[\[}}5.000000e+00, 6.000000e+00], [3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00], [1.000000e+00, 2.000000e+00], [5.000000e+00, 6.000000e+00]]> : tensor<5x2xf32>
  // CHECK-DAG: %[[LOOKUP1:.*]] = arith.constant dense<{{\[\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]], {{\[\[}}5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]], {{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]> : tensor<3x2x2xf32>
  // CHECK: return %[[LOOKUP0]], %[[LOOKUP1]] : tensor<5x2xf32>, tensor<3x2x2xf32>
}

// CHECK-LABEL: @less_int_both_splat
func.func @less_int_both_splat() -> tensor<4xi1> {
  %0 = arith.constant dense<3> : tensor<4xi32>
  %1 = arith.constant dense<10> : tensor<4xi32>

  %2 = "tfl.less"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<true> : tensor<4xi1>

// CHECK-LABEL: @less_int_one_splat
func.func @less_int_one_splat() -> tensor<4xi1> {
  %0 = arith.constant dense<3> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.less"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK:%cst = arith.constant dense<[true, false, false, false]> : tensor<4xi1>

// CHECK-LABEL: @less_int
func.func @less_int() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.less"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, false, false, true]> : tensor<4xi1>

// CHECK-LABEL: @less_int64
func.func @less_int64() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi64>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi64>

  %2 = "tfl.less"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, false, false, true]> : tensor<4xi1>

// CHECK-LABEL: @less_float
func.func @less_float() -> tensor<4xi1> {
  %0 = arith.constant dense<[11.0, 2.0, 0.0, 2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, -1.0, 3.0]> : tensor<4xf32>

  %2 = "tfl.less"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, false, false, true]> : tensor<4xi1>

// CHECK-LABEL: @less_equal_int
func.func @less_equal_int() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.less_equal"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, true, false, true]> : tensor<4xi1>

// CHECK-LABEL: @less_equal_int64
func.func @less_equal_int64() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi64>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi64>

  %2 = "tfl.less_equal"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, true, false, true]> : tensor<4xi1>

// CHECK-LABEL: @less_equal_float
func.func @less_equal_float() -> tensor<4xi1> {
  %0 = arith.constant dense<[11.0, 2.0, 0.0, 2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, -1.0, 3.0]> : tensor<4xf32>

  %2 = "tfl.less_equal"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, true, false, true]> : tensor<4xi1>

// CHECK-LABEL: @greater_int
func.func @greater_int() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.greater"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true, false]> : tensor<4xi1>

// CHECK-LABEL: @greater_int64
func.func @greater_int64() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi64>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi64>

  %2 = "tfl.greater"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true, false]> : tensor<4xi1>

// CHECK-LABEL: @greater_float
func.func @greater_float() -> tensor<4xi1> {
  %0 = arith.constant dense<[11.0, 2.0, 0.0, 2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, -1.0, 3.0]> : tensor<4xf32>

  %2 = "tfl.greater"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true, false]> : tensor<4xi1>

// CHECK-LABEL: @greater_equal_int
func.func @greater_equal_int() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.greater_equal"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, true, true, false]> : tensor<4xi1>

// CHECK-LABEL: @greater_equal_int64
func.func @greater_equal_int64() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi64>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi64>

  %2 = "tfl.greater_equal"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, true, true, false]> : tensor<4xi1>

// CHECK-LABEL: @greater_equal_float
func.func @greater_equal_float() -> tensor<4xi1> {
  %0 = arith.constant dense<[11.0, 2.0, 0.0, 2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, -1.0, 3.0]> : tensor<4xf32>

  %2 = "tfl.greater_equal"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, true, true, false]> : tensor<4xi1>

// CHECK-LABEL: @equal_int
func.func @equal_int() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.equal"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, true, false, false]> : tensor<4xi1>

// CHECK-LABEL: @equal_int64
func.func @equal_int64() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi64>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi64>

  %2 = "tfl.equal"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, true, false, false]> : tensor<4xi1>

// CHECK-LABEL: @equal_float
func.func @equal_float() -> tensor<4xi1> {
  %0 = arith.constant dense<[11.0, 2.0, 0.0, 2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, -1.0, 3.0]> : tensor<4xf32>

  %2 = "tfl.equal"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[false, true, false, false]> : tensor<4xi1>

// CHECK-LABEL: @not_equal_int
func.func @not_equal_int() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi32>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi32>

  %2 = "tfl.not_equal"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true, true]> : tensor<4xi1>

// CHECK-LABEL: @not_equal_int64
func.func @not_equal_int64() -> tensor<4xi1> {
  %0 = arith.constant dense<[11, 2, 0, 2]> : tensor<4xi64>
  %1 = arith.constant dense<[10, 2, -1, 3]> : tensor<4xi64>

  %2 = "tfl.not_equal"(%0, %1) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true, true]> : tensor<4xi1>

// CHECK-LABEL: @not_equal_float
func.func @not_equal_float() -> tensor<4xi1> {
  %0 = arith.constant dense<[11.0, 2.0, 0.0, 2.0]> : tensor<4xf32>
  %1 = arith.constant dense<[10.0, 2.0, -1.0, 3.0]> : tensor<4xf32>

  %2 = "tfl.not_equal"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>

  func.return %2 : tensor<4xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true, true]> : tensor<4xi1>

// CHECK-LABEL: @logical_or
func.func @logical_or() -> tensor<3xi1> {
  %0 = arith.constant dense<[true, false, true]> : tensor<3xi1>
  %1 = arith.constant dense<[false, false, true]> : tensor<3xi1>

  %2 = "tfl.logical_or"(%0, %1) : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>

  func.return %2 : tensor<3xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true]> : tensor<3xi1>

// CHECK-LABEL: @logical_and
func.func @logical_and() -> tensor<3xi1> {
  %0 = arith.constant dense<[true, false, true]> : tensor<3xi1>
  %1 = arith.constant dense<[false, false, true]> : tensor<3xi1>

  %2 = "tfl.logical_and"(%0, %1) : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>

  func.return %2 : tensor<3xi1>
}

// CHECK: %cst = arith.constant dense<[false, false, true]> : tensor<3xi1>

// CHECK-LABEL: @select_splat_cond
func.func @select_splat_cond() -> tensor<4xi32> {
  %cond = arith.constant dense<true> : tensor<4xi1>
  %0 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %1 = arith.constant dense<[-1, -2, -3, -4]> : tensor<4xi32>

  %2 = "tfl.select"(%cond, %0, %1) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>
}

// CHECK: %cst = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

// CHECK-LABEL: select_splat_lhs
func.func @select_splat_lhs() -> tensor<4xi32> {
  %cond = arith.constant dense<[true, true, false, false]> : tensor<4xi1>
  %0 = arith.constant dense<0> : tensor<4xi32>
  %1 = arith.constant dense<[-1, -2, -3, -4]> : tensor<4xi32>

  %2 = "tfl.select"(%cond, %0, %1) : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  func.return %2 : tensor<4xi32>
}

// CHECK: %cst = arith.constant dense<[0, 0, -3, -4]> : tensor<4xi32>

// CHECK-LABEL: select_float
func.func @select_float() -> tensor<4xf32> {
  %cond = arith.constant dense<[true, true, false, false]> : tensor<4xi1>
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = arith.constant dense<[-1.0, -2.0, -3.0, -4.0]> : tensor<4xf32>

  %2 = "tfl.select"(%cond, %0, %1) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %2 : tensor<4xf32>
}

// CHECK: %cst = arith.constant dense<[1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00]> : tensor<4xf32

// CHECK-LABEL: floor
func.func @floor() -> tensor<3xf32> {
  %cst = arith.constant dense<[-1.0, 0.0, 0.99]> : tensor<3xf32>
  %0 = "tfl.floor"(%cst) : (tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// CHECK: %cst = arith.constant dense<[-1.000000e+00, 0.000000e+00, 0.000000e+00]> : tensor<3xf32>

// CHECK-LABEL: floor_f64
func.func @floor_f64() -> tensor<3xf64> {
  %cst = arith.constant dense<[-1.0, 0.0, 0.99]> : tensor<3xf64>
  %0 = "tfl.floor"(%cst) : (tensor<3xf64>) -> tensor<3xf64>
  func.return %0 : tensor<3xf64>
}

// CHECK: tfl.floor

// CHECK-LABEL: exp
func.func @exp() -> tensor<4xf32> {
  %cst = arith.constant dense<[-1.0, 0.0, 0.99, 0.36787944117]> : tensor<4xf32>
  %0 = "tfl.exp"(%cst) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK: %cst = arith.constant dense<[0.36787945, 1.000000e+00, 2.69123459, 1.44466782]> : tensor<4xf32>

// CHECK-LABEL: exp_f64
func.func @exp_f64() -> tensor<4xf64> {
  %cst = arith.constant dense<[-1.0, 0.0, 0.99, 0.36787944117]> : tensor<4xf64>
  %0 = "tfl.exp"(%cst) : (tensor<4xf64>) -> tensor<4xf64>
  func.return %0 : tensor<4xf64>
}

// CHECK: tfl.exp

// CHECK-LABEL: pow_float
func.func @pow_float() -> tensor<3xf32> {
  %0 = arith.constant dense<[1.0, 0.0, 2.0]> : tensor<3xf32>
  %1 = arith.constant dense<[2.0, 3.0, -1.5]> : tensor<3xf32>

  %2 = "tfl.pow"(%0, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>

  func.return %2 : tensor<3xf32>
}

// CHECK: %cst = arith.constant dense<[1.000000e+00, 0.000000e+00, 0.353553385]> : tensor<3xf32>

// CHECK-LABEL: pow_int
func.func @pow_int() -> tensor<3xi32> {
  %0 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %1 = arith.constant dense<[2, 3, -1]> : tensor<3xi32>

  %2 = "tfl.pow"(%0, %1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>

  func.return %2 : tensor<3xi32>
}

// CHECK: %cst = arith.constant dense<[1, 0, 0]> : tensor<3xi32>

// CHECK-LABEL: logical_not
func.func @logical_not() -> tensor<3xi1> {
  %cst = arith.constant dense<[false, true, false]> : tensor<3xi1>
  %0 = "tfl.logical_not"(%cst) : (tensor<3xi1>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// CHECK: %cst = arith.constant dense<[true, false, true]> : tensor<3xi1>

// CHECK-LABEL: logical_not_splat
func.func @logical_not_splat() -> tensor<3xi1> {
  %cst = arith.constant dense<false> : tensor<3xi1>
  %0 = "tfl.logical_not"(%cst) : (tensor<3xi1>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// CHECK: %cst = arith.constant dense<true> : tensor<3xi1>

// CHECK-LABEL: bitwise_xor_i32
func.func @bitwise_xor_i32() -> tensor<3xi32> {
  %0 = arith.constant dense<[0, 5, 3]> : tensor<3xi32>
  %1 = arith.constant dense<[5, 0, 7]> : tensor<3xi32>

  %2 = "tfl.bitwise_xor"(%0, %1) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>

  func.return %2 : tensor<3xi32>
}

// CHECK: %cst = arith.constant dense<[5, 5, 4]> : tensor<3xi32>

// CHECK-LABEL: bitwise_xor_ui8
func.func @bitwise_xor_ui8() -> tensor<3xui8> {
  %0 = arith.constant dense<[0, 5, 3]> : tensor<3xui8>
  %1 = arith.constant dense<[5, 0, 7]> : tensor<3xui8>

  %2 = "tfl.bitwise_xor"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>

  func.return %2 : tensor<3xui8>
}

// CHECK: %cst = arith.constant dense<[5, 5, 4]> : tensor<3xui8>

// CHECK-LABEL: relu
func.func @relu() -> tensor<3xf32> {
  %cst = arith.constant dense<[-1.0, 0.0, 0.99]> : tensor<3xf32>
  %0 = "tfl.relu"(%cst) : (tensor<3xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// CHECK: %cst = arith.constant dense<[0.000000e+00, 0.000000e+00, 9.900000e-01]> : tensor<3xf32>

// CHECK-LABEL: slice
func.func @slice_first_dim() -> tensor<1x1x5x6xf32> {
  %cst_0 = arith.constant dense<9.000000e+00> : tensor<2x1x5x6xf32>
  %cst_1 = arith.constant dense<0> : tensor<4xi32>
  %cst_2 = arith.constant dense<[1, 1, 5, 6]> : tensor<4xi32>
  %0 = "tfl.slice"(%cst_0, %cst_1, %cst_2) : (tensor<2x1x5x6xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x1x5x6xf32>
  func.return %0 : tensor<1x1x5x6xf32>
}

// CHECK %cst = arith.constant dense<9.000000e+00> : tensor<1x1x5x6xf32>

// CHECK-LABEL: slice_trivial
func.func @slice_trivial(%arg0: tensor<2x1x5x6xf32>) -> tensor<2x1x5x6xf32> {
  %cst_1 = arith.constant dense<0> : tensor<4xi32>
  %cst_2 = arith.constant dense<[2, 1, 5, 6]> : tensor<4xi32>
  %0 = "tfl.slice"(%arg0, %cst_1, %cst_2) : (tensor<2x1x5x6xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<2x1x5x6xf32>
  func.return %0 : tensor<2x1x5x6xf32>
}

// CHECK-NOT: tfl.slice


// CHECK-LABEL: sum
func.func @sum() -> tensor<2xf32> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]> : tensor<2x2x2xf32>
  %0 = "tfl.sum"(%cst_1, %cst) <{keep_dims = false}> : (tensor<2x2x2xf32>, tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK: arith.constant dense<[1.200000e+01, 1.600000e+01]> : tensor<2xf32>

// CHECK-LABEL: sum_keep_dims
func.func @sum_keep_dims() -> tensor<1x1x2xf32> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xi32>
  %cst_1 = arith.constant dense<[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]> : tensor<2x2x2xf32>
  %0 = "tfl.sum"(%cst_1, %cst) <{keep_dims = true}> : (tensor<2x2x2xf32>, tensor<2xi32>) -> tensor<1x1x2xf32>
  func.return %0 : tensor<1x1x2xf32>
}

// CHECK-LITERAL: arith.constant dense<[[[1.200000e+01, 1.600000e+01]]]> : tensor<1x1x2xf32>

// CHECK-LABEL: gather
func.func @gather() -> (tensor<2x3x4x5xi16>, tensor<2x3x4x5xi16>) {
  %params = arith.constant dense<[
    [[[1111, 1112, 1113, 1114, 1115],
      [1121, 1122, 1123, 1124, 1125],
      [1131, 1132, 1133, 1134, 1135],
      [1141, 1142, 1143, 1144, 1145],
      [1151, 1152, 1153, 1154, 1155],
      [1161, 1162, 1163, 1164, 1165]],
     [[1211, 1212, 1213, 1214, 1215],
      [1221, 1222, 1223, 1224, 1225],
      [1231, 1232, 1233, 1234, 1235],
      [1241, 1242, 1243, 1244, 1245],
      [1251, 1252, 1253, 1254, 1255],
      [1261, 1262, 1263, 1264, 1265]],
     [[1311, 1312, 1313, 1314, 1315],
      [1321, 1322, 1323, 1324, 1325],
      [1331, 1332, 1333, 1334, 1335],
      [1341, 1342, 1343, 1344, 1345],
      [1351, 1352, 1353, 1354, 1355],
      [1361, 1362, 1363, 1364, 1365]]],
    [[[2111, 2112, 2113, 2114, 2115],
      [2121, 2122, 2123, 2124, 2125],
      [2131, 2132, 2133, 2134, 2135],
      [2141, 2142, 2143, 2144, 2145],
      [2151, 2152, 2153, 2154, 2155],
      [2161, 2162, 2163, 2164, 2165]],
     [[2211, 2212, 2213, 2214, 2215],
      [2221, 2222, 2223, 2224, 2225],
      [2231, 2232, 2233, 2234, 2235],
      [2241, 2242, 2243, 2244, 2245],
      [2251, 2252, 2253, 2254, 2255],
      [2261, 2262, 2263, 2264, 2265]],
     [[2311, 2312, 2313, 2314, 2315],
      [2321, 2322, 2323, 2324, 2325],
      [2331, 2332, 2333, 2334, 2335],
      [2341, 2342, 2343, 2344, 2345],
      [2351, 2352, 2353, 2354, 2355],
      [2361, 2362, 2363, 2364, 2365]]]]> : tensor<2x3x6x5xi16>
  %indices = arith.constant dense<[[5, 4, 3, 2], [3, 2, 1, 0]]> : tensor<2x4xi64>
  %gathered = "tfl.gather"(%params, %indices) <{axis = 2 : i32, batch_dims = 1 : i32}> : (tensor<2x3x6x5xi16>, tensor<2x4xi64>) -> tensor<2x3x4x5xi16>
  // This is the same tensor as the one on the CHECK line
  %expected = arith.constant dense<[
    [[[1161, 1162, 1163, 1164, 1165],
      [1151, 1152, 1153, 1154, 1155],
      [1141, 1142, 1143, 1144, 1145],
      [1131, 1132, 1133, 1134, 1135]],
     [[1261, 1262, 1263, 1264, 1265],
      [1251, 1252, 1253, 1254, 1255],
      [1241, 1242, 1243, 1244, 1245],
      [1231, 1232, 1233, 1234, 1235]],
     [[1361, 1362, 1363, 1364, 1365],
      [1351, 1352, 1353, 1354, 1355],
      [1341, 1342, 1343, 1344, 1345],
      [1331, 1332, 1333, 1334, 1335]]],
    [[[2141, 2142, 2143, 2144, 2145],
      [2131, 2132, 2133, 2134, 2135],
      [2121, 2122, 2123, 2124, 2125],
      [2111, 2112, 2113, 2114, 2115]],
     [[2241, 2242, 2243, 2244, 2245],
      [2231, 2232, 2233, 2234, 2235],
      [2221, 2222, 2223, 2224, 2225],
      [2211, 2212, 2213, 2214, 2215]],
     [[2341, 2342, 2343, 2344, 2345],
      [2331, 2332, 2333, 2334, 2335],
      [2321, 2322, 2323, 2324, 2325],
      [2311, 2312, 2313, 2314, 2315]]]]> : tensor<2x3x4x5xi16>
  func.return %gathered, %expected : tensor<2x3x4x5xi16>, tensor<2x3x4x5xi16>
  // CHECK-NOT: tfl.gather
  // CHECK: [[CST:%.*]] = arith.constant dense<"0x89048A048B048C048D047F048004810482048304750476047704780479046B046C046D046E046F04ED04EE04EF04F004F104E304E404E504E604E704D904DA04DB04DC04DD04CF04D004D104D204D304510552055305540555054705480549054A054B053D053E053F0540054105330534053505360537055D085E085F08600861085308540855085608570849084A084B084C084D083F084008410842084308C108C208C308C408C508B708B808B908BA08BB08AD08AE08AF08B008B108A308A408A508A608A708250926092709280929091B091C091D091E091F09110912091309140915090709080909090A090B09"> : tensor<2x3x4x5xi16> 
  // If the return value is the same constant twice, the result is the same as expected
  // CHECK: return [[CST]], [[CST]]
}


// CHECK-LABEL: reverse_2_dims
func.func @reverse_2_dims() -> tensor<2x3x2xi32> {
  %input = "tfl.pseudo_const"() <{value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>}> : () -> tensor<2x3x2xi32>
  %axis = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %reverse = "tfl.reverse_v2"(%input, %axis) : (tensor<2x3x2xi32>, tensor<2xi32>) -> tensor<2x3x2xi32>
  return %reverse : tensor<2x3x2xi32>
}

// CHECK-LITERAL: %cst = arith.constant dense<[[[11, 12], [9, 10], [7, 8]], [[5, 6], [3, 4], [1, 2]]]> : tensor<2x3x2xi32>
// CHECK:         return %cst : tensor<2x3x2xi32>

// CHECK-LABEL: reverse_1_dim
func.func @reverse_1_dim() -> tensor<2x3x2xi32> {
  %input = "tfl.pseudo_const"() <{value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xi32>}> : () -> tensor<2x3x2xi32>
  %axis = "tfl.pseudo_const"() <{value = dense<[2]> : tensor<1xi32>}> : () -> tensor<1xi32>
  %reverse = "tfl.reverse_v2"(%input, %axis) : (tensor<2x3x2xi32>, tensor<1xi32>) -> tensor<2x3x2xi32>
  return %reverse : tensor<2x3x2xi32>
}

// CHECK-LITERAL: %cst = arith.constant dense<[[[2, 1], [4, 3], [6, 5]], [[8, 7], [10, 9], [12, 11]]]> : tensor<2x3x2xi32>
// CHECK:         return %cst : tensor<2x3x2xi32>

// CHECK-LABEL: @slice_no_op
func.func @slice_no_op() -> tensor<2x3x2xi32> {
  %cst_1 = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : tensor<2x3x2xi32>
  %cst_2 = arith.constant dense<0> : tensor<3xi32>
  %cst_3 = arith.constant dense<[2, 3, 2]> : tensor<3xi32>
  %0 = "tfl.slice"(%cst_1, %cst_2, %cst_3) : (tensor<2x3x2xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<2x3x2xi32>
  return %0 : tensor<2x3x2xi32>
}

// CHECK-LITERAL: %cst = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : tensor<2x3x2xi32>
// CHECK-NOT:     slice

// CHECK-LABEL: @slice_some_dims
func.func @slice_some_dims() -> tensor<2x2x1xi32> {
  %cst_1 = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : tensor<2x3x2xi32>
  %cst_2 = arith.constant dense<[0, 1, 1]> : tensor<3xi32>
  %cst_3 = arith.constant dense<[2, 2, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%cst_1, %cst_2, %cst_3) : (tensor<2x3x2xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<2x2x1xi32>
  return %0 : tensor<2x2x1xi32>
}

// CHECK-LITERAL: %cst = arith.constant dense<[[[3], [5]], [[9], [11]]]> : tensor<2x2x1xi32>
// CHECK-NOT:     slice

// CHECK-LABEL: @slice_all_dims
func.func @slice_all_dims() -> tensor<1x2x1xi32> {
  %cst_1 = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : tensor<2x3x2xi32>
  %cst_2 = arith.constant dense<[1, 1, 1]> : tensor<3xi32>
  %cst_3 = arith.constant dense<[1, 2, 1]> : tensor<3xi32>
  %0 = "tfl.slice"(%cst_1, %cst_2, %cst_3) : (tensor<2x3x2xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x2x1xi32>
  return %0 : tensor<1x2x1xi32>
}

// CHECK-LITERAL: %cst = arith.constant dense<[[[9], [11]]]> : tensor<1x2x1xi32>
// CHECK-NOT:     slice

// CHECK-LABEL: @slice_some_dims_i64
func.func @slice_some_dims_i64() -> tensor<2x2x1xi32> {
  %cst_1 = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : tensor<2x3x2xi32>
  %cst_2 = arith.constant dense<[0, 1, 1]> : tensor<3xi64>
  %cst_3 = arith.constant dense<[2, 2, 1]> : tensor<3xi64>
  %0 = "tfl.slice"(%cst_1, %cst_2, %cst_3) : (tensor<2x3x2xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x2x1xi32>
  return %0 : tensor<2x2x1xi32>
}

// CHECK-LITERAL: %cst = arith.constant dense<[[[3], [5]], [[9], [11]]]> : tensor<2x2x1xi32>
// CHECK-NOT:     slice

// CHECK-LABEL: @slice_big_float
func.func @slice_big_float() -> tensor<1x1x1792x256xf32> {
  %cst_1 = arith.constant dense<9.000000e+00> : tensor<2x1x1792x256xf32>
  %cst_2 = arith.constant dense<[1, 0, 0, 0]> : tensor<4xi32>
  %cst_3 = arith.constant dense<[1, 1, 1792, 256]> : tensor<4xi32>
  %0 = "tfl.slice"(%cst_1, %cst_2, %cst_3) : (tensor<2x1x1792x256xf32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x1x1792x256xf32>
  return %0 : tensor<1x1x1792x256xf32>
}

// CHECK-LITERAL: %cst = arith.constant dense<9.000000e+00> : tensor<1x1x1792x256xf32>
// CHECK-NOT:     slice



