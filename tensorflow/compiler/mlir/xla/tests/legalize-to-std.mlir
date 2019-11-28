// RUN: tf-opt -xla-legalize-to-std %s -o - | FileCheck %s

// CHECK-LABEL: func @binary_ops_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
func @binary_ops_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT:   %0 = addf %arg0, %arg1 : tensor<4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {name = "add.3"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %1 = mulf %0, %arg1 : tensor<4xf32>
  %1 = "xla_hlo.mul"(%0, %arg1) {name = "mul.4"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %2 = subf %1, %arg1 : tensor<4xf32>
  %2 = "xla_hlo.sub"(%1, %arg1) {name = "sub.5"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %3 = divf %2, %arg1 : tensor<4xf32>
  %3 = "xla_hlo.div"(%2, %arg1) {name = "div.6"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   %4 = remf %3, %arg1 : tensor<4xf32>
  %4 = "xla_hlo.remainder"(%3, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT:   return %4 : tensor<4xf32>
  return %4 : tensor<4xf32>
}

// CHECK-LABEL: func @binary_ops_int(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
func @binary_ops_int(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NEXT:   %0 = addi %arg0, %arg1 : tensor<4xi32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {name = "add.3"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %1 = muli %0, %arg1 : tensor<4xi32>
  %1 = "xla_hlo.mul"(%0, %arg1) {name = "mul.4"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %2 = subi %1, %arg1 : tensor<4xi32>
  %2 = "xla_hlo.sub"(%1, %arg1) {name = "sub.5"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %3 = divis %2, %arg1 : tensor<4xi32>
  %3 = "xla_hlo.div"(%2, %arg1) {name = "div.6"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   %4 = remis %3, %arg1 : tensor<4xi32>
  %4 = "xla_hlo.remainder"(%3, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>

  // CHECK-NEXT:   return %4 : tensor<4xi32>
  return %4 : tensor<4xi32>
}

// Broadcasting is not currently supported.
// TODO(suderman):Future pass should take all broadcasted binary ops and convert
// them to separate broadcast and binary op.
// CHECK-LABEL: func @binary_ops_broadcast(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>) -> tensor<4x4xf32> {
func @binary_ops_broadcast(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>) -> tensor<4x4xf32> {
  // CHECK-NEXT: %0 = "xla_hlo.add"(%arg0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "add.3"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %0 = "xla_hlo.add"(%arg0, %arg1) {
      name = "add.3", broadcast_dimensions = dense<1> : tensor<1xi64>} :
          (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>

  // CHECK-NEXT: %1 = "xla_hlo.mul"(%0, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "mul.4"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %1 = "xla_hlo.mul"(%0, %arg1) {
      name = "mul.4", broadcast_dimensions = dense<1> : tensor<1xi64>} :
          (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>

  // CHECK-NEXT: %2 = "xla_hlo.sub"(%1, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "sub.5"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %2 = "xla_hlo.sub"(%1, %arg1) {
      name = "sub.5", broadcast_dimensions = dense<1> : tensor<1xi64>} :
          (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>

  // CHECK-NEXT: %3 = "xla_hlo.div"(%2, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "div.6"} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %3 = "xla_hlo.div"(%2, %arg1) {
      name = "div.6", broadcast_dimensions = dense<1> : tensor<1xi64>} :
          (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>

  // CHECK-NEXT: %4 = "xla_hlo.remainder"(%3, %arg1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  %4 = "xla_hlo.remainder"(%3, %arg1) {
    broadcast_dimensions = dense<1> : tensor<1xi64>} :
          (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>

  // CHECK-NEXT: return %4 : tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}

// CHECK-LABEL: func @compare_int(%arg0: tensor<4xi32>) -> (tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>) {
func @compare_int(%arg0: tensor<4xi32>) -> (tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>) {
  // CHECK-NEXT: %0 = cmpi "eq", %arg0, %arg0 : tensor<4xi32>
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %1 = cmpi "ne", %arg0, %arg0 : tensor<4xi32>
  %1 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "NE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %2 = cmpi "slt", %arg0, %arg0 : tensor<4xi32>
  %2 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %3 = cmpi "sle", %arg0, %arg0 : tensor<4xi32>
  %3 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %4 = cmpi "sgt", %arg0, %arg0 : tensor<4xi32>
  %4 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: %5 = cmpi "sge", %arg0, %arg0 : tensor<4xi32>
  %5 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  // CHECK-NEXT: return %0, %1, %2, %3, %4, %5 : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
  return %0, %1, %2, %3, %4, %5 : tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// CHECK-LABEL: func @compare_float
func @compare_float(%arg0: tensor<4xf32>) -> (tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>,tensor<4xi1>) {
  // CHECK-NEXT: %0 = cmpf "oeq", %arg0, %arg0 : tensor<4xf32>
  %0 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %1 = cmpf "une", %arg0, %arg0 : tensor<4xf32>
  %1 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "NE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %2 = cmpf "olt", %arg0, %arg0 : tensor<4xf32>
  %2 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LT"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %3 = cmpf "ole", %arg0, %arg0 : tensor<4xf32>
  %3 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "LE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %4 = cmpf "ogt", %arg0, %arg0 : tensor<4xf32>
  %4 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GT"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  // CHECK-NEXT: %5 = cmpf "oge", %arg0, %arg0 : tensor<4xf32>
  %5 = "xla_hlo.compare"(%arg0, %arg0) {comparison_direction = "GE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0, %1, %2, %3, %4, %5: tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>, tensor<4xi1>
}

// CHECK-LABEL: func @int_constant
func @int_constant() -> (tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>) {
  // CHECK-NEXT: [[CST0:%.+]] = constant {{.+}} : tensor<i32>
  %0 = "xla_hlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  // CHECK-NEXT: [[CST1:%.+]] = constant {{.+}} : tensor<2x3xi32>
  %1 = "xla_hlo.constant"() {value = dense<1> : tensor<2x3xi32>} : () -> (tensor<2x3xi32>)
  // CHECK-NEXT: [[CST2:%.+]] = constant {{.+}} : tensor<2x3xi32>
  %2 = "xla_hlo.constant"() {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>} : () -> (tensor<2x3xi32>)
  // CHECK-NEXT: return [[CST0]], [[CST1]], [[CST2]] : tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
  return %0, %1, %2: tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
}

// CHECK-LABEL: func @float_constant
func @float_constant() -> (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK-NEXT: [[CST0:%.+]] = constant {{.+}} : tensor<f32>
  %0 = "xla_hlo.constant"() {value = dense<0.0> : tensor<f32>} : () -> (tensor<f32>)
  // CHECK-NEXT: [[CST1:%.+]] = constant {{.+}} : tensor<2x3xf32>
  %1 = "xla_hlo.constant"() {value = dense<1.0> : tensor<2x3xf32>} : () -> (tensor<2x3xf32>)
  // CHECK-NEXT: [[CST2:%.+]] = constant {{.+}} : tensor<2x3xf32>
  %2 = "xla_hlo.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> (tensor<2x3xf32>)
  // CHECK-NEXT: return [[CST0]], [[CST1]], [[CST2]] : tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>
  return %0, %1, %2: tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>
}

