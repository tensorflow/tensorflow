// RUN: tf-opt %s -canonicalize | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: @add_float
func @add_float() -> (tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = constant dense<4.5> : tensor<f32>
  %1 = constant dense<1.5> : tensor<f32>

  %2 = constant dense< 3.5> : tensor<4xf32>
  %3 = constant dense<-0.5> : tensor<4xf32>

  // CHECK: %[[CST:.*]] = constant dense<3.500000e+00> : tensor<4xf32>
  // CHECK: %[[CST_0:.*]]  = constant dense<-5.000000e-01> : tensor<4xf32>
  // CHECK: %[[CST_1:.*]]  = constant dense<6.000000e+00> : tensor<f32>
  // CHECK: %[[CST_2:.*]]  = constant dense<4.000000e+00> : tensor<4xf32>
  // CHECK: %[[CST_3:.*]]  = constant dense<5.000000e+00> : tensor<4xf32>
  // CHECK: %[[CST_4:.*]]  = constant dense<3.000000e+00> : tensor<4xf32>
  // CHECK: %0 = tfl.add %[[CST]], %[[CST_0]] {fused_activation_function = "SIGN_BIT"} : tensor<4xf32>

  %5 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<  f32>) -> tensor<  f32>
  %6 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = "tfl.add"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<  f32>) -> tensor<4xf32>
  %8 = "tfl.add"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %9 = "tfl.add"(%2, %3) {fused_activation_function = "SIGN_BIT"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %5, %6, %7, %8, %9 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @add_int
func @add_int() -> (tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = constant dense<8> : tensor<i32>
  %1 = constant dense<1> : tensor<i32>

  %2 = constant dense< 4> : tensor<4xi32>
  %3 = constant dense<-2> : tensor<4xi32>

  // CHECK: %[[CST:.*]] = constant dense<9> : tensor<i32>
  // CHECK: %[[CST_0:.*]]  = constant dense<6> : tensor<4xi32>
  // CHECK: %[[CST_1:.*]]  = constant dense<5> : tensor<4xi32>
  // CHECK: %[[CST_2:.*]]  = constant dense<2> : tensor<4xi32>

  %5 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<  i32>) -> tensor<  i32>
  %6 = "tfl.add"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<4xi32>) -> tensor<4xi32>
  %7 = "tfl.add"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<  i32>) -> tensor<4xi32>
  %8 = "tfl.add"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %5, %6, %7, %8 : tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @sub_float
func @sub_float() -> (tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = constant dense<4.5> : tensor<f32>
  %1 = constant dense<1.5> : tensor<f32>

  %2 = constant dense< 3.5> : tensor<4xf32>
  %3 = constant dense<-0.5> : tensor<4xf32>

  // CHECK: %[[CST:.*]] = constant dense<3.000000e+00> : tensor<f32>
  // CHECK: %[[CST_0:.*]]  = constant dense<5.000000e+00> : tensor<4xf32>
  // CHECK: %[[CST_1:.*]]  = constant dense<2.000000e+00> : tensor<4xf32>
  // CHECK: %[[CST_2:.*]]  = constant dense<4.000000e+00> : tensor<4xf32>

  %5 = "tfl.sub"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<  f32>) -> tensor<  f32>
  %6 = "tfl.sub"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = "tfl.sub"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<  f32>) -> tensor<4xf32>
  %8 = "tfl.sub"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %5, %6, %7, %8 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @sub_int
func @sub_int() -> (tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = constant dense<8> : tensor<i32>
  %1 = constant dense<1> : tensor<i32>

  %2 = constant dense< 4> : tensor<4xi32>
  %3 = constant dense<-2> : tensor<4xi32>

  // CHECK: %[[CST:.*]] = constant dense<7> : tensor<i32>
  // CHECK: %[[CST_0:.*]]  = constant dense<10> : tensor<4xi32>
  // CHECK: %[[CST_1:.*]]  = constant dense<3> : tensor<4xi32>
  // CHECK: %[[CST_2:.*]]  = constant dense<6> : tensor<4xi32>

  %5 = "tfl.sub"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<  i32>) -> tensor<  i32>
  %6 = "tfl.sub"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<4xi32>) -> tensor<4xi32>
  %7 = "tfl.sub"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<  i32>) -> tensor<4xi32>
  %8 = "tfl.sub"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %5, %6, %7, %8 : tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @mul_float
func @mul_float() -> (tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %0 = constant dense<4.5> : tensor<f32>
  %1 = constant dense<1.5> : tensor<f32>

  %2 = constant dense< 3.5> : tensor<4xf32>
  %3 = constant dense<-0.5> : tensor<4xf32>

  // CHECK: %[[CST:.*]] = constant dense<6.750000e+00> : tensor<f32>
  // CHECK: %[[CST_0:.*]]  = constant dense<-2.250000e+00> : tensor<4xf32>
  // CHECK: %[[CST_1:.*]]  = constant dense<5.250000e+00> : tensor<4xf32>
  // CHECK: %[[CST_2:.*]]  = constant dense<-1.750000e+00> : tensor<4xf32>

  %5 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<  f32>) -> tensor<  f32>
  %6 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  f32>, tensor<4xf32>) -> tensor<4xf32>
  %7 = "tfl.mul"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<  f32>) -> tensor<4xf32>
  %8 = "tfl.mul"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %5, %6, %7, %8 : tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: @elementwise_unary_ops
func @elementwise_unary_ops() -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  %0 = constant dense<-1.0> : tensor<f32>
  %1 = constant dense<1.0> : tensor<f32>
  %2 = constant dense<1.0> : tensor<f32>
  %3 = constant dense<1.0> : tensor<f32>
  %4 = constant dense<4.0> : tensor<f32>
  %5 = constant dense<4.0> : tensor<f32>
  %6 = constant dense<2.0> : tensor<f32>

  // CHECK-DAG: [[cst0:%.*]] = constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: [[cst1:%.*]] = constant dense<0.841470957> : tensor<f32>
  // CHECK-DAG: [[cst2:%.*]] = constant dense<0.540302277> : tensor<f32>
  // CHECK-DAG: [[cst3:%.*]] = constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG: [[cst4:%.*]] = constant dense<2.000000e+00> : tensor<f32>
  // CHECK-DAG: [[cst5:%.*]] = constant dense<5.000000e-01> : tensor<f32>
  // CHECK-DAG: [[cst6:%.*]] = constant dense<4.000000e+00> : tensor<f32>
  // CHECK: return [[cst0]], [[cst1]], [[cst2]], [[cst3]], [[cst4]], [[cst5]], [[cst6]]

  %7 = "tfl.abs"(%0) : (tensor<f32>) -> tensor<f32>
  %8 = "tfl.sin"(%1) : (tensor<f32>) -> tensor<f32>
  %9 = "tfl.cos"(%2) : (tensor<f32>) -> tensor<f32>
  %10 = "tfl.log"(%3) : (tensor<f32>) -> tensor<f32>
  %11 = "tfl.sqrt"(%4) : (tensor<f32>) -> tensor<f32>
  %12 = "tfl.rsqrt"(%5) : (tensor<f32>) -> tensor<f32>
  %13 = "tfl.square"(%6) : (tensor<f32>) -> tensor<f32>

  return %7, %8, %9, %10, %11, %12, %13 : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// CHECK-LABEL: @mul_int
func @mul_int() -> (tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) {
  %0 = constant dense<8> : tensor<i32>
  %1 = constant dense<1> : tensor<i32>

  %2 = constant dense< 4> : tensor<4xi32>
  %3 = constant dense<-2> : tensor<4xi32>

  // CHECK-DAG: [[cst0:%.*]] = constant dense<8> : tensor<i32>
  // CHECK-DAG: [[cst1:%.*]] = constant dense<-16> : tensor<4xi32>
  // CHECK-DAG: [[cst2:%.*]] = constant dense<4> : tensor<4xi32>
  // CHECK-DAG: [[cst3:%.*]] = constant dense<-8> : tensor<4xi32>
  // CHECK: return [[cst0]], [[cst1]], [[cst2]], [[cst3]]

  %5 = "tfl.mul"(%0, %1) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<  i32>) -> tensor<  i32>
  %6 = "tfl.mul"(%0, %3) {fused_activation_function = "NONE"} : (tensor<  i32>, tensor<4xi32>) -> tensor<4xi32>
  %7 = "tfl.mul"(%2, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<  i32>) -> tensor<4xi32>
  %8 = "tfl.mul"(%2, %3) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %5, %6, %7, %8 : tensor<i32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>
}

// CHECK-LABEL: @add_dense_splat_int
func @add_dense_splat_int() -> tensor<4xi32> {
  %0 = constant dense<[-10, -1, 42, 100]> : tensor<4xi32>
  %1 = constant dense< 5> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %2 : tensor<4xi32>

// CHECK:  %[[CST:.*]] = constant dense<[-5, 4, 47, 105]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_splat_dense_int
func @add_splat_dense_int() -> tensor<4xi32> {
  %0 = constant dense< 5> : tensor<4xi32>
  %1 = constant dense<[-10, -1, 42, 100]> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %2 : tensor<4xi32>

// CHECK:  %[[CST:.*]] = constant dense<[-5, 4, 47, 105]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_int_same_shape
func @add_dense_dense_int_same_shape() -> tensor<4xi32> {
  %0 = constant dense<[15, 23, -44, -2]> : tensor<4xi32>
  %1 = constant dense<[-10, -1, 42, 100]> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  return %2 : tensor<4xi32>

// CHECK:  %[[CST:.*]] = constant dense<[5, 22, -2, 98]> : tensor<4xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_int_trailing_dim
func @add_dense_dense_int_trailing_dim() -> (tensor<2x2xi32>, tensor<2x2x2xi32>, tensor<2x2x2xi32>) {
  %cst_0 = constant dense<[10, 20]> : tensor<2xi32>
  %cst_1 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %cst_2 = constant dense<[[[1, 1], [2, 2]], [[3, 3], [4, 4]]]> : tensor<2x2x2xi32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<    2xi32>, tensor<  2x2xi32>) -> tensor<  2x2xi32>
  %1 = "tfl.add"(%cst_2, %cst_1) {fused_activation_function = "NONE"} : (tensor<2x2x2xi32>, tensor<  2x2xi32>) -> tensor<2x2x2xi32>
  %2 = "tfl.add"(%cst_0, %cst_2) {fused_activation_function = "NONE"} : (tensor<    2xi32>, tensor<2x2x2xi32>) -> tensor<2x2x2xi32>

  return %0, %1, %2 : tensor<2x2xi32>, tensor<2x2x2xi32>, tensor<2x2x2xi32>

// CHECK:  %[[CST:.*]] = constant dense<{{\[\[}}11, 22], [13, 24]]> : tensor<2x2xi32>
// CHECK:  %[[CST_0:.*]]  = constant dense<{{\[\[\[}}2, 3], [5, 6]], {{\[\[}}4, 5], [7, 8]]]> : tensor<2x2x2xi32>
// CHECK:  %[[CST_1:.*]]  = constant dense<{{\[\[\[}}11, 21], [12, 22]], {{\[\[}}13, 23], [14, 24]]]> : tensor<2x2x2xi32>
// CHECK:  return %[[CST]], %[[CST_0]], %[[CST_1]]
}

// CHECK-LABEL: @add_dense_dense_int_mixing_1_n
func @add_dense_dense_int_mixing_1_n() -> tensor<2x2xi32> {
  %cst_0 = constant dense<[[1, 2]]> : tensor<1x2xi32>
  %cst_1 = constant dense<[[3], [4]]> : tensor<2x1xi32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<2x2xi32>

  return %0 : tensor<2x2xi32>
// CHECK: %[[CST:.*]] = constant dense<{{\[\[}}4, 5], [5, 6]]> : tensor<2x2xi32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_splat_float
func @add_dense_splat_float() -> tensor<4xf32> {
  %0 = constant dense<[-10.0, -1.5, 42.0, 7.25]> : tensor<4xf32>
  %1 = constant dense< 3.5> : tensor<4xf32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %2 : tensor<4xf32>

// CHECK:  %[[CST:.*]] = constant dense<[-6.500000e+00, 2.000000e+00, 4.550000e+01, 1.075000e+01]> : tensor<4xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_splat_dense_float
func @add_splat_dense_float() -> tensor<4xf32> {
  %0 = constant dense< 3.5> : tensor<4xf32>
  %1 = constant dense<[-10.0, -1.5, 42.0, 7.25]> : tensor<4xf32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %2 : tensor<4xf32>

// CHECK:  %[[CST:.*]] = constant dense<[-6.500000e+00, 2.000000e+00, 4.550000e+01, 1.075000e+01]> : tensor<4xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_float_same_shape
func @add_dense_dense_float_same_shape() -> (tensor<4xf32>) {
  %0 = constant dense<[1.5, 2.3, -4.4, -2.0]> : tensor<4xf32>
  %1 = constant dense<[-10.4, -1.3, 42.4, 100.0]> : tensor<4xf32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %2 : tensor<4xf32>

// CHECK:  %[[CST:.*]] = constant dense<[-8.89999961, 1.000000e+00, 3.800000e+01, 9.800000e+01]> : tensor<4xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_float_trailing_dim
func @add_dense_dense_float_trailing_dim() -> (tensor<2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>) {
  %cst_0 = constant dense<[1., -4.]> : tensor<2xf32>
  %cst_1 = constant dense<[[-5.5, 1.5], [7.5, -4.5]]> : tensor<2x2xf32>
  %cst_2 = constant dense<[[[1., 1.], [2., 2.]], [[3., 3.], [4., 4.]]]> : tensor<2x2x2xf32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<    2xf32>, tensor<  2x2xf32>) -> tensor<  2x2xf32>
  %1 = "tfl.add"(%cst_2, %cst_1) {fused_activation_function = "NONE"} : (tensor<2x2x2xf32>, tensor<  2x2xf32>) -> tensor<2x2x2xf32>
  %2 = "tfl.add"(%cst_0, %cst_2) {fused_activation_function = "NONE"} : (tensor<    2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>

  return %0, %1, %2 : tensor<2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>

// CHECK:  %[[CST:.*]] = constant dense<{{\[\[}}-4.500000e+00, -2.500000e+00], [8.500000e+00, -8.500000e+00]]> : tensor<2x2xf32>
// CHECK:  %[[CST_0:.*]]  = constant dense<{{\[\[\[}}-4.500000e+00, 2.500000e+00], [9.500000e+00, -2.500000e+00]], {{\[\[}}-2.500000e+00, 4.500000e+00], [1.150000e+01, -5.000000e-01]]]> : tensor<2x2x2xf32>
// CHECK:  %[[CST_1:.*]]  = constant dense<{{\[\[\[}}2.000000e+00, -3.000000e+00], [3.000000e+00, -2.000000e+00]], {{\[\[}}4.000000e+00, -1.000000e+00], [5.000000e+00, 0.000000e+00]]]> : tensor<2x2x2xf32>
// CHECK:  return %[[CST]], %[[CST_0]], %[[CST_1]]
}

// CHECK-LABEL: @add_dense_dense_float_mixfng_1_n
func @add_dense_dense_float_mixfng_1_n() -> tensor<2x2xf32> {
  %cst_0 = constant dense<[[1.5, -2.5]]> : tensor<1x2xf32>
  %cst_1 = constant dense<[[-3.], [4.]]> : tensor<2x1xf32>

  %0 = "tfl.add"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>

  return %0 : tensor<2x2xf32>

// CHECK: %[[CST:.*]] = constant dense<{{\[\[}}-1.500000e+00, -5.500000e+00], [5.500000e+00, 1.500000e+00]]> : tensor<2x2xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @rank
func @rank() -> tensor<1xi32> {
  %cst = constant dense<[[1], [2]]> : tensor<2x1xi32>

  // CHECK: %[[CST:.*]] = constant dense<2> : tensor<1xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.rank"(%cst) : (tensor<2x1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: @rank_input_known_rank
func @rank_input_known_rank(%arg0 : tensor<2x1xi32>) -> tensor<1xi32> {
  // CHECK: %[[CST:.*]] = constant dense<2> : tensor<1xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.rank"(%arg0) : (tensor<2x1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: @reshape
func @reshape() -> tensor<4xi32> {
  %input = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %shape = constant dense<[4]> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.reshape"(%input, %shape) : (tensor<2x2xi32>, tensor<1xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @reshape_dynamic_output
func @reshape_dynamic_output() -> tensor<?xi32> {
  %input = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %shape = constant dense<[4]> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.reshape"(%input, %shape) : (tensor<2x2xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}


// CHECK-LABEL: @pseudo_const
func @pseudo_const() -> tensor<i32> {
  // CHECK: %[[CST:.*]] = constant dense<1> : tensor<i32>
  // CHECK: return %[[CST]]
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  return %0 : tensor<i32>
}


// CHECK-LABEL: @range_int
func @range_int() -> tensor<?xi32> {
  %cst = constant dense<0> : tensor<i32>
  %cst_1 = constant dense<4> : tensor<i32>
  %cst_2 = constant dense<1> : tensor<i32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[0, 1, 2, 3]> : tensor<4xi32>} : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: @range_float
func @range_float() -> tensor<?xf32> {
  %cst = constant dense<0.0> : tensor<f32>
  %cst_1 = constant dense<4.0> : tensor<f32>
  %cst_2 = constant dense<1.0> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>} : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// CHECK-LABEL: @range_float_neg_delta
func @range_float_neg_delta() -> tensor<?xf32> {
  %cst = constant dense<0.0> : tensor<f32>
  %cst_1 = constant dense<-4.0> : tensor<f32>
  %cst_2 = constant dense<-1.0> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[0.000000e+00, -1.000000e+00, -2.000000e+00, -3.000000e+00]> : tensor<4xf32>} : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @range_float_nonzero_base
func @range_float_nonzero_base() -> tensor<?xf32> {
  %cst = constant dense<2.0> : tensor<f32>
  %cst_1 = constant dense<7.0> : tensor<f32>
  %cst_2 = constant dense<1.5> : tensor<f32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[2.000000e+00, 3.500000e+00, 5.000000e+00, 6.500000e+00]> : tensor<4xf32>} : () -> tensor<?xf32>
  // CHECK: return %[[CST]]
  %0 = "tfl.range"(%cst, %cst_1, %cst_2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @transpose_no_fold
func @transpose_no_fold(%arg0 : tensor<2xi32>) -> tensor<2x2xi32> {
  %cst = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>

  // CHECK: tfl.transpose
  %0 = "tfl.transpose"(%cst, %arg0) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_1d
// Basic 1D identity
func @transpose_1d() -> tensor<3xi32> {
  %cst = constant dense<[1, 2, 3]> : tensor<3xi32>
  %cst_perm = constant dense<0> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = constant dense<{{\[}}1, 2, 3]> : tensor<3xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<3xi32>, tensor<1xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: @transpose_dynamic
func @transpose_dynamic() -> tensor<?xi32> {
  %cst = constant dense<[1, 2, 3]> : tensor<3xi32>
  %cst_perm = constant dense<0> : tensor<1xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<{{\[}}1, 2, 3]> : tensor<3xi32>} : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<3xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// CHECK-LABEL: @transpose_2d
func @transpose_2d() -> tensor<2x2xi32> {
  %cst = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %cst_perm = constant dense<[1, 0]> : tensor<2xi32>

  // CHECK: %[[CST:.*]] = constant dense<{{\[\[}}0, 2], {{\[}}1, 3]]> : tensor<2x2xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_2d_identity
func @transpose_2d_identity() -> tensor<2x2xi32> {
  %cst = constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
  %cst_perm = constant dense<[0, 1]> : tensor<2xi32>

  // CHECK: %[[CST:.*]] = constant dense<{{\[\[}}0, 1], {{\[}}2, 3]]> : tensor<2x2xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @transpose_3d
// A test case adopted from TransposeTest.Test3DInputConstTensor in
// tensorflow/lite/kernels/transpose_test.cc
func @transpose_3d() -> tensor<4x2x3xi32> {
  %cst = constant dense<[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]> : tensor<2x3x4xi32>
  %cst_perm = constant dense<[2, 0, 1]> : tensor<3xi32>

  // CHECK: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 4, 8], {{\[}}12, 16, 20]], {{\[\[}}1, 5, 9], {{\[}}13, 17, 21]], {{\[\[}}2, 6, 10], {{\[}}14, 18, 22]], {{\[\[}}3, 7, 11], {{\[}}15, 19, 23]]]> : tensor<4x2x3xi32>
  // CHECK: return %[[CST]]
  %0 = "tfl.transpose"(%cst, %cst_perm) : (tensor<2x3x4xi32>, tensor<3xi32>) -> tensor<4x2x3xi32>
  return %0 : tensor<4x2x3xi32>
}

// CHECK-LABEL: @ConstantFoldBinaryOpDynamicOutput
func @ConstantFoldBinaryOpDynamicOutput() -> tensor<?xi32> {
  %cst = constant dense<10> : tensor<i32>
  %cst_0 = "tfl.pseudo_const"() {value = dense<[5, 10]> : tensor<2xi32>} : () -> tensor<?xi32>
  %87 = "tfl.sub"(%cst_0, %cst) {fused_activation_function = "NONE"} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  return %87 : tensor<?xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[-5, 0]> : tensor<2xi32>} : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @add_dense_dense_int_same_shape_dynamic
func @add_dense_dense_int_same_shape_dynamic() -> tensor<?xi32> {
  %0 = constant dense<[15, 23, -44, -2]> : tensor<4xi32>
  %1 = constant dense<[-10, -1, 42, 100]> : tensor<4xi32>

  %2 = "tfl.add"(%0, %1) {fused_activation_function = "NONE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<?xi32>

  return %2 : tensor<?xi32>

  // CHECK: %[[CST:.*]] = "tfl.pseudo_const"() {value = dense<[5, 22, -2, 98]> : tensor<4xi32>} : () -> tensor<?xi32>
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @concat_2_tensors_1_empty
func @concat_2_tensors_1_empty() -> tensor<2xi32> {
  %1 = constant dense<1> : tensor<2xi32>
  %2 = constant dense<[]> : tensor<0xi32>
  %3 = "tfl.concatenation"(%1, %2) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<0xi32>) -> tensor<2xi32>
  return %3 : tensor<2xi32>

  // CHECK: %[[CST:.*]] = constant dense<1> : tensor<2xi32>
  // CHECK: return %[[CST]] : tensor<2xi32>
}

// CHECK-LABEL: @concat_3_tensors_1_empty
func @concat_3_tensors_1_empty() -> tensor<?xi32> {
  %0 = constant dense<1> : tensor<2xi32>
  %1 = constant dense<1> : tensor<2xi32>
  %2 = constant dense<[]> : tensor<0xi32>
  %3 = "tfl.concatenation"(%0, %1, %2) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>, tensor<0xi32>) -> tensor<?xi32>
  return %3 : tensor<?xi32>

  // CHECK: %0 = "tfl.concatenation"(%[[CST]], %[[CST]]) {axis = 0 : i32, fused_activation_function = "NONE"}
  // CHECK: return %0 : tensor<?xi32>
}

// CHECK-LABEL: @concatConstantTensorsFirstDim
func @concatConstantTensorsFirstDim() -> tensor<2x2x3xi32> {
  %cst_0 = constant dense<0> : tensor<1x2x3xi32>
  %cst_1 = constant dense<1> : tensor<1x2x3xi32>
  %0 = "tfl.concatenation"(%cst_0, %cst_1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> tensor<2x2x3xi32>
  return %0 : tensor<2x2x3xi32>

  // CHECK: %[[CST:.*]] = constant dense<[{{\[}}{{\[}}0, 0, 0], {{\[}}0, 0, 0]], {{\[}}{{\[}}1, 1, 1], {{\[}}1, 1, 1]]]> : tensor<2x2x3xi32>
  // CHECK-NOT: constant-dense
  // CHECK-NOT: "tfl.concatenation"
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @concatConstantTensorsMiddleDim
func @concatConstantTensorsMiddleDim() -> tensor<1x4x3xi32> {
  %cst_0 = constant dense<0> : tensor<1x2x3xi32>
  %cst_1 = constant dense<1> : tensor<1x2x3xi32>
  %0 = "tfl.concatenation"(%cst_0, %cst_1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> tensor<1x4x3xi32>
  return %0 : tensor<1x4x3xi32>

  // CHECK: %[[CST:.*]] = constant dense<[{{\[}}{{\[}}0, 0, 0], {{\[}}0, 0, 0], {{\[}}1, 1, 1], {{\[}}1, 1, 1]]]> : tensor<1x4x3xi32>
  // CHECK-NOT: constant-dense
  // CHECK-NOT: "tfl.concatenation"
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @concatConstantTensorsLastDim
func @concatConstantTensorsLastDim() -> tensor<1x2x6xi32> {
  %cst_0 = constant dense<0> : tensor<1x2x3xi32>
  %cst_1 = constant dense<1> : tensor<1x2x3xi32>
  %0 = "tfl.concatenation"(%cst_0, %cst_1) {axis = 2 : i32, fused_activation_function = "NONE"} : (tensor<1x2x3xi32>, tensor<1x2x3xi32>) -> tensor<1x2x6xi32>
  return %0 : tensor<1x2x6xi32>

  // CHECK: %[[CST:.*]] = constant dense<[{{\[}}{{\[}}0, 0, 0, 1, 1, 1], {{\[}}0, 0, 0, 1, 1, 1]]]> : tensor<1x2x6xi32>
  // CHECK-NOT: constant-dense
  // CHECK-NOT: "tfl.concatenation"
  // CHECK: return %[[CST]]
}

// CHECK-LABEL: @div_dense_dense_float_mixfng_1_n
func @div_dense_dense_float_mixfng_1_n() -> tensor<2x2xf32> {
  %cst_0 = constant dense<[[1.5, -2.5]]> : tensor<1x2xf32>
  %cst_1 = constant dense<[[-3.], [4.]]> : tensor<2x1xf32>

  %0 = "tfl.div"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2xf32>, tensor<2x1xf32>) -> tensor<2x2xf32>

  return %0 : tensor<2x2xf32>

// CHECK: %[[CST:.*]] = constant dense<{{\[\[}}-5.000000e-01, 0.833333313], [3.750000e-01, -6.250000e-01]]> : tensor<2x2xf32>
// CHECK:  return %[[CST]]
}

// CHECK-LABEL: @div_dense_different_rank
func @div_dense_different_rank() -> tensor<1x2x2xf32> {
  %cst_0 = constant dense<[[[1.0],[2.0]]]> : tensor<1x2x1xf32>
  %cst_1 = constant dense<[[2.0, 3.0]]> : tensor<1x2xf32>

  %0 = "tfl.div"(%cst_0, %cst_1) {fused_activation_function = "NONE"} : (tensor<1x2x1xf32>, tensor<1x2xf32>) -> tensor<1x2x2xf32>

  return %0 : tensor<1x2x2xf32>

// CHECK: %[[CST:.*]] = constant dense<[{{\[}}{{\[}}5.000000e-01, 0.333333343], [1.000000e+00, 0.666666686]]]> : tensor<1x2x2xf32>
// CHECK:  return %[[CST]]
}
