// RUN: litert-opt %s -canonicalize | FILECHECK_OPTS="" FileCheck %s
// RUN: litert-opt %s --tfl-dense-to-dense-resource-elements -canonicalize | litert-opt --tfl-dense-resource-to-dense-elements | FILECHECK_OPTS="" FileCheck %s

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

// CHECK-LABEL: @reshape
func.func @reshape() -> tensor<4xi32> {
  %input = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %shape = arith.constant dense<[4]> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.reshape"(%input, %shape) : (tensor<2x2xi32>, tensor<1xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
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
