// RUN: tf-opt %s -test-xla-lower-complex | FileCheck %s

// CHECK-LABEL: @add
func @add(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.add %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.add %arg1, %arg3
  %4 = "xla_hlo.add"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @add_broadcast
func @add_broadcast(%arg0 : tensor<1x2xf32>, %arg1 : tensor<1x2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.add"(%arg0, %arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL1:%.+]] = "xla_hlo.add"(%arg1, %arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %4 = "xla_hlo.add"(%2, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<1x2xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  return %5, %6 : tensor<1x2xf32>, tensor<1x2xf32>
}

// CHECK-LABEL: @add_unranked
func @add_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.add %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.add %arg1, %arg3
  %4 = "xla_hlo.add"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @sub
func @sub(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.subtract %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.subtract %arg1, %arg3
  %4 = "xla_hlo.subtract"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @sub_broadcast
func @sub_broadcast(%arg0 : tensor<1x2xf32>, %arg1 : tensor<1x2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.subtract"(%arg0, %arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL1:%.+]] = "xla_hlo.subtract"(%arg1, %arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %4 = "xla_hlo.subtract"(%2, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<1x2xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  return %5, %6 : tensor<1x2xf32>, tensor<1x2xf32>
}

// CHECK-LABEL: @sub_unranked
func @sub_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.subtract %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.subtract %arg1, %arg3
  %4 = "xla_hlo.subtract"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @mul
func @mul(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.multiply %arg1, %arg3
  // CHECK-DAG: [[VAL2:%.+]] = xla_hlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.multiply %arg0, %arg3
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = xla_hlo.add [[VAL3]], [[VAL4]]
  %4 = "xla_hlo.multiply"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return %2, %5 : tensor<2xf32>, tensor<2xf32>
  return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @mul_broadcast
func @mul_broadcast(%arg0 : tensor<1x2xf32>, %arg1 : tensor<1x2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.multiply"(%arg0, %arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL1:%.+]] = "xla_hlo.multiply"(%arg1, %arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL2:%.+]] = xla_hlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = "xla_hlo.multiply"(%arg0, %arg3) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL4:%.+]] = "xla_hlo.multiply"(%arg1, %arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL5:%.+]] = xla_hlo.add [[VAL3]], [[VAL4]]
  %4 = "xla_hlo.multiply"(%2, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<1x2xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)

  // CHECK: return %2, %5 : tensor<1x2xf32>, tensor<1x2xf32>
  return %5, %6 : tensor<1x2xf32>, tensor<1x2xf32>
}

// CHECK-LABEL: @mul_unranked
func @mul_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.multiply %arg1, %arg3
  // CHECK-DAG: [[VAL2:%.+]] = xla_hlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.multiply %arg0, %arg3
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = xla_hlo.add [[VAL3]], [[VAL4]]
  %4 = "xla_hlo.multiply"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = "xla_hlo.real"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return %2, %5 : tensor<*xf32>, tensor<*xf32>
  return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @div
func @div(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.neg"(%arg3)

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL2:%.+]] = xla_hlo.multiply %arg1, [[VAL0]]
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = xla_hlo.multiply %arg3, [[VAL0]]
  // CHECK-DAG: [[VAL6:%.+]] = xla_hlo.subtract [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = xla_hlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL8:%.+]] = xla_hlo.multiply %arg0, [[VAL0]]
  // CHECK-DAG: [[VAL9:%.+]] = xla_hlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = xla_hlo.divide [[VAL3]], [[VAL6]]
  // CHECK-DAG: [[VAL11:%.+]] = xla_hlo.divide [[VAL9]], [[VAL6]]
  %4 = "xla_hlo.divide"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  %5 = "xla_hlo.real"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: @div_broadcast
func @div_broadcast(%arg0 : tensor<1x2xf32>, %arg1 : tensor<1x2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<1x2xf32>) -> (tensor<1x2xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.neg"(%arg3)

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = "xla_hlo.multiply"(%arg0, %arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL2:%.+]] = "xla_hlo.multiply"(%arg1, [[VAL0]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = xla_hlo.multiply %arg3, [[VAL0]]
  // CHECK-DAG: [[VAL6:%.+]] = xla_hlo.subtract [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = "xla_hlo.multiply"(%arg1, %arg2)
  // CHECK-DAG: [[VAL8:%.+]] = "xla_hlo.multiply"(%arg0, [[VAL0]])
  // CHECK-DAG: [[VAL9:%.+]] = xla_hlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = "xla_hlo.divide"([[VAL3]], [[VAL6]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL11:%.+]] = "xla_hlo.divide"([[VAL9]], [[VAL6]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  %4 = "xla_hlo.divide"(%2, %3) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<1x2xcomplex<f32>>)

  %5 = "xla_hlo.real"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<1x2xcomplex<f32>>) -> (tensor<1x2xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  return %5, %6 : tensor<1x2xf32>, tensor<1x2xf32>
}

// -----

// CHECK-LABEL: @div_unranked
func @div_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "xla_hlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.neg"(%arg3)

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL2:%.+]] = xla_hlo.multiply %arg1, [[VAL0]]
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = xla_hlo.multiply %arg3, [[VAL0]]
  // CHECK-DAG: [[VAL6:%.+]] = xla_hlo.subtract [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = xla_hlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL8:%.+]] = xla_hlo.multiply %arg0, [[VAL0]]
  // CHECK-DAG: [[VAL9:%.+]] = xla_hlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = xla_hlo.divide [[VAL3]], [[VAL6]]
  // CHECK-DAG: [[VAL11:%.+]] = xla_hlo.divide [[VAL9]], [[VAL6]]
  %4 = "xla_hlo.divide"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)

  %5 = "xla_hlo.real"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = "xla_hlo.imag"(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @abs
func @abs(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> (tensor<2xf32>) {
  %0 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = xla_hlo.multiply %arg0, %arg0
  // CHECK-DAG: [[VAL1:%.+]] = xla_hlo.multiply %arg1, %arg1
  // CHECK-DAG: [[VAL2:%.+]] = xla_hlo.add [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = "xla_hlo.sqrt"([[VAL2]])
  %1 = "xla_hlo.abs"(%0) : (tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %2 = "xla_hlo.real"(%1) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL3]]
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: @exp
func @exp(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %0 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.exp"(%arg0)
  // CHECK-DAG: [[VAL1:%.+]] = "xla_hlo.cos"(%arg1)
  // CHECK-DAG: [[VAL2:%.+]] = "xla_hlo.sin"(%arg1)
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.multiply [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply [[VAL0]], [[VAL2]]
  %1 = "xla_hlo.exp"(%0) : (tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %2 = "xla_hlo.real"(%1) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %3 = "xla_hlo.imag"(%1) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL3]], [[VAL4]]
  return %2, %3 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @exp_unranked
func @exp_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0 = "xla_hlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = "xla_hlo.exp"(%arg0)
  // CHECK-DAG: [[VAL1:%.+]] = "xla_hlo.cos"(%arg1)
  // CHECK-DAG: [[VAL2:%.+]] = "xla_hlo.sin"(%arg1)
  // CHECK-DAG: [[VAL3:%.+]] = xla_hlo.multiply [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL4:%.+]] = xla_hlo.multiply [[VAL0]], [[VAL2]]
  %1 = "xla_hlo.exp"(%0) : (tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %2 = "xla_hlo.real"(%1) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %3 = "xla_hlo.imag"(%1) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL3]], [[VAL4]]
  return %2, %3 : tensor<*xf32>, tensor<*xf32>
}
