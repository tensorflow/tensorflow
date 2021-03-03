// RUN: mlir-hlo-opt -mhlo-legalize-to-std %s -o - | FileCheck %s

// CHECK-LABEL: func @binary_ops_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
func @binary_ops_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %0 = addf %arg0, %arg1 : tensor<4xf32>
  %0 = "mhlo.add"(%arg0, %arg1) {name = "add.3"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %1 = mulf %0, %arg1 : tensor<4xf32>
  %1 = "mhlo.multiply"(%0, %arg1) {name = "mul.4"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %2 = subf %1, %arg1 : tensor<4xf32>
  %2 = "mhlo.subtract"(%1, %arg1) {name = "sub.5"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %3 = divf %2, %arg1 : tensor<4xf32>
  %3 = "mhlo.divide"(%2, %arg1) {name = "div.6"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %4 = remf %3, %arg1 : tensor<4xf32>
  %4 = "mhlo.remainder"(%3, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   return %4 : tensor<4xf32>
  return %4 : tensor<4xf32>
}

// CHECK-LABEL: func @binary_ops_int(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
func @binary_ops_int(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT:   %0 = addi %arg0, %arg1 : tensor<4xi32>
  %0 = "mhlo.add"(%arg0, %arg1) {name = "add.3"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %1 = muli %0, %arg1 : tensor<4xi32>
  %1 = "mhlo.multiply"(%0, %arg1) {name = "mul.4"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %2 = subi %1, %arg1 : tensor<4xi32>
  %2 = "mhlo.subtract"(%1, %arg1) {name = "sub.5"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %3 = divi_signed %2, %arg1 : tensor<4xi32>
  %3 = "mhlo.divide"(%2, %arg1) {name = "div.6"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %4 = remi_signed %3, %arg1 : tensor<4xi32>
  %4 = "mhlo.remainder"(%3, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   return %4 : tensor<4xi32>
  return %4 : tensor<4xi32>
}

// CHECK-LABEL: func @unary_ops_float
func @unary_ops_float(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %0 = ceilf %arg0 : tensor<4xf32>
  %0 = "mhlo.ceil"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   return %0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @compare_int
func @compare_int(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>) {
  // CHECK-NEXT: %0 = cmpi eq, %arg0, %arg1 : tensor<4xi32>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %1 = cmpi ne, %arg0, %arg1 : tensor<4xi32>
  %1 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %2 = cmpi slt, %arg0, %arg1 : tensor<4xi32>
  %2 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %3 = cmpi sle, %arg0, %arg1 : tensor<4xi32>
  %3 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %4 = cmpi sgt, %arg0, %arg1 : tensor<4xi32>
  %4 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %5 = cmpi sge, %arg0, %arg1 : tensor<4xi32>
  %5 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: return %0, %1, %2, %3, %4, %5 : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
  return %0, %1, %2, %3, %4, %5 : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// CHECK-LABEL: func @compare_float
func @compare_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>) {
  // CHECK-NEXT: %0 = cmpf oeq, %arg0, %arg1 : tensor<4xf32>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %1 = cmpf une, %arg0, %arg1 : tensor<4xf32>
  %1 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "NE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %2 = cmpf olt, %arg0, %arg1 : tensor<4xf32>
  %2 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %3 = cmpf ole, %arg0, %arg1 : tensor<4xf32>
  %3 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %4 = cmpf ogt, %arg0, %arg1 : tensor<4xf32>
  %4 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GT"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %5 = cmpf oge, %arg0, %arg1 : tensor<4xf32>
  %5 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "GE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0, %1, %2, %3, %4, %5: tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// CHECK-LABEL: func @int_constant
func @int_constant() -> (tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>) {
  // CHECK-NEXT: [[CST0:%.+]] = constant {{.+}} : tensor<i32>
  %0 = "mhlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  // CHECK-NEXT: [[CST1:%.+]] = constant {{.+}} : tensor<2x3xi32>
  %1 = "mhlo.constant"() {value = dense<1> : tensor<2x3xi32>} : () -> (tensor<2x3xi32>)
  // CHECK-NEXT: [[CST2:%.+]] = constant {{.+}} : tensor<2x3xi32>
  %2 = "mhlo.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>} : () -> (tensor<2x3xi32>)
  // CHECK-NEXT: return [[CST0]], [[CST1]], [[CST2]] : tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
  return %0, %1, %2: tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
}

// CHECK-LABEL: func @float_constant
func @float_constant() -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK-NEXT: [[CST0:%.+]] = constant {{.+}} : tensor<f32>
  %0 = "mhlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
  // CHECK-NEXT: [[CST1:%.+]] = constant {{.+}} : tensor<2x3xf32>
  %1 = "mhlo.constant"() {value = dense<1.0> : tensor<2x3xf32>} : () -> (tensor<2x3xf32>)
  // CHECK-NEXT: [[CST2:%.+]] = constant {{.+}} : tensor<2x3xf32>
  %2 = "mhlo.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> (tensor<2x3xf32>)
  // CHECK-NEXT: return [[CST0]], [[CST1]], [[CST2]] : tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %0, %1, %2: tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>
}

// Test Iota lowering to constant
// CHECK-LABEL: func @iota.const.1() -> tensor<4xi32> {
func @iota.const.1() -> tensor<4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @iota.const.2() -> tensor<2x4xi32> {
func @iota.const.2() -> tensor<2x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[}}0, 0, 0, 0], [1, 1, 1, 1]]> : tensor<2x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// CHECK-LABEL: func @iota.const.3() -> tensor<2x4xi32> {
func @iota.const.3() -> tensor<2x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[}}0, 1, 2, 3], [0, 1, 2, 3]]> : tensor<2x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<2x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// CHECK-LABEL: func @iota.const.4() -> tensor<2x3x4xi32> {
func @iota.const.4() -> tensor<2x3x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0{{\]\]}}, {{\[\[}}1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]> : tensor<2x3x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<2x3x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func @iota.const.5() -> tensor<2x3x4xi32> {
func @iota.const.5() -> tensor<2x3x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2{{\]\]}}, {{\[\[}}0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]]> : tensor<2x3x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<2x3x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func @iota.const.6() -> tensor<2x3x4xi32> {
func @iota.const.6() -> tensor<2x3x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<{{\[\[\[}}0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3{{\]\]}}, {{\[\[}}0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]> : tensor<2x3x4xi32>
  %0 = "mhlo.iota"() {iota_dimension = 2 : i64} : () -> tensor<2x3x4xi32>
  // CHECK-NEXT: return %[[CST]] : tensor<2x3x4xi32>
  return %0 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func @iota.const.f32
func @iota.const.f32() -> tensor<4xf32> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xf32>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @iota.const.f64
func @iota.const.f64() -> tensor<4xf64> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf64>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xf64>
  // CHECK-NEXT: return %[[CST]] : tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK-LABEL: func @iota.const.bf16
func @iota.const.bf16() -> tensor<4xbf16> {
  // CHECK-NEXT: %[[CST:.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xbf16>
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xbf16>
  // CHECK-NEXT: return %[[CST]] : tensor<4xbf16>
  return %0 : tensor<4xbf16>
}

// CHECK-LABEL: func @iota.const.complex.f32
func @iota.const.complex.f32() -> tensor<4xcomplex<f32>> {
  // CHECK-NEXT: [[REAL:%.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf32>
  // CHECK-NEXT: [[IMAG:%.*]] = constant dense<0.000000e+00> : tensor<4xf32>
  // CHECK-NEXT: [[COMPLEX:%.*]] = "mhlo.complex"([[REAL]], [[IMAG]])
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xcomplex<f32>>
  // CHECK-NEXT: return [[COMPLEX]] : tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func @iota.const.complex.f64
func @iota.const.complex.f64() -> tensor<4xcomplex<f64>> {
  // CHECK-NEXT: [[REAL:%.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<4xf64>
  // CHECK-NEXT: [[IMAG:%.*]] = constant dense<0.000000e+00> : tensor<4xf64>
  // CHECK-NEXT: [[COMPLEX:%.*]] = "mhlo.complex"([[REAL]], [[IMAG]])
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xcomplex<f64>>
  // CHECK-NEXT: return [[COMPLEX]] : tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}
