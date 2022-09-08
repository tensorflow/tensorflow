// RUN: mlir-hlo-opt %s -chlo-legalize-to-hlo -mhlo-test-lower-complex | FileCheck %s

// CHECK-LABEL: @add
func.func @add(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.add %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.add %arg1, %arg3
  %4 = "mhlo.add"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = mhlo.real(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = mhlo.imag(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @add_unranked
func.func @add_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.add %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.add %arg1, %arg3
  %4 = "mhlo.add"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = mhlo.real(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = mhlo.imag(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @sub
func.func @sub(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.subtract %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.subtract %arg1, %arg3
  %4 = "mhlo.subtract"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = mhlo.real(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = mhlo.imag(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @sub_unranked
func.func @sub_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.subtract %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.subtract %arg1, %arg3
  %4 = "mhlo.subtract"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = mhlo.real(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = mhlo.imag(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL0]], [[VAL1]]
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @mul
func.func @mul(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.multiply %arg1, %arg3
  // CHECK-DAG: [[VAL2:%.+]] = mhlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = mhlo.multiply %arg0, %arg3
  // CHECK-DAG: [[VAL4:%.+]] = mhlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = mhlo.add [[VAL3]], [[VAL4]]
  %4 = "mhlo.multiply"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)
  %5 = mhlo.real(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = mhlo.imag(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return %2, %5 : tensor<2xf32>, tensor<2xf32>
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @mul_unranked
func.func @mul_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.multiply %arg1, %arg3
  // CHECK-DAG: [[VAL2:%.+]] = mhlo.subtract [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = mhlo.multiply %arg0, %arg3
  // CHECK-DAG: [[VAL4:%.+]] = mhlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = mhlo.add [[VAL3]], [[VAL4]]
  %4 = "mhlo.multiply"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)
  %5 = mhlo.real(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = mhlo.imag(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return %2, %5 : tensor<*xf32>, tensor<*xf32>
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @div
func.func @div(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>, %arg3 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.negate %arg3

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL2:%.+]] = mhlo.multiply %arg1, [[VAL0]]
  // CHECK-DAG: [[VAL3:%.+]] = mhlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = mhlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = mhlo.multiply %arg3, %arg3
  // CHECK-DAG: [[VAL6:%.+]] = mhlo.add [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = mhlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL8:%.+]] = mhlo.multiply %arg0, [[VAL0]]
  // CHECK-DAG: [[VAL9:%.+]] = mhlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = mhlo.divide [[VAL3]], [[VAL6]]
  // CHECK-DAG: [[VAL11:%.+]] = mhlo.divide [[VAL9]], [[VAL6]]
  %4 = "mhlo.divide"(%2, %3) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  %5 = mhlo.real(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %6 = mhlo.imag(%4) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  func.return %5, %6 : tensor<2xf32>, tensor<2xf32>
}

// -----

// CHECK-LABEL: @div_unranked
func.func @div_unranked(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>, %arg2 : tensor<*xf32>, %arg3 : tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %2 = "mhlo.complex"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)
  %3 = "mhlo.complex"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.negate %arg3

  // Compute the numerator's real component:
  //   numerator.real = lhs.real * rhs.real  lhs.imag * rhs.imag
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.multiply %arg0, %arg2
  // CHECK-DAG: [[VAL2:%.+]] = mhlo.multiply %arg1, [[VAL0]]
  // CHECK-DAG: [[VAL3:%.+]] = mhlo.subtract [[VAL1]], [[VAL2]]

  // Compute the real valued denominator as rhs * con(rhs):
  //   denominator = rhs.real * rhs.real + rhs.imag * rhs.imag
  // CHECK-DAG: [[VAL4:%.+]] = mhlo.multiply %arg2, %arg2
  // CHECK-DAG: [[VAL5:%.+]] = mhlo.multiply %arg3, %arg3
  // CHECK-DAG: [[VAL6:%.+]] = mhlo.add [[VAL4]], [[VAL5]]

  // Compute the numerator's imaginary component:
  //   numerator.imag = lhs.imag * rhs.real - lhs.real * rhs.imag
  // CHECK-DAG: [[VAL7:%.+]] = mhlo.multiply %arg1, %arg2
  // CHECK-DAG: [[VAL8:%.+]] = mhlo.multiply %arg0, [[VAL0]]
  // CHECK-DAG: [[VAL9:%.+]] = mhlo.add [[VAL8]], [[VAL7]]

  // Divide the numerator by the real valued denominator.
  // CHECK-DAG: [[VAL10:%.+]] = mhlo.divide [[VAL3]], [[VAL6]]
  // CHECK-DAG: [[VAL11:%.+]] = mhlo.divide [[VAL9]], [[VAL6]]

  %4 = "mhlo.divide"(%2, %3) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)

  %5 = mhlo.real(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)
  %6 = mhlo.imag(%4) : (tensor<*xcomplex<f32>>) -> (tensor<*xf32>)

  // CHECK: return [[VAL10]], [[VAL11]]
  func.return %5, %6 : tensor<*xf32>, tensor<*xf32>
}

// CHECK-LABEL: @abs
func.func @abs(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> (tensor<2xf32>) {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[VAL0:%.+]] = mhlo.multiply %arg0, %arg0
  // CHECK-DAG: [[VAL1:%.+]] = mhlo.multiply %arg1, %arg1
  // CHECK-DAG: [[VAL2:%.+]] = mhlo.add [[VAL0]], [[VAL1]]
  // CHECK-DAG: [[VAL3:%.+]] = mhlo.sqrt [[VAL2]]
  %1 = "mhlo.abs"(%0) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: return [[VAL3]]
  func.return %1 : tensor<2xf32>
}

// CHECK-LABEL: @exp
func.func @exp(%arg0 : tensor<2xf32>, %arg1 : tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>)

  // CHECK-DAG: [[EXP:%.+]] = mhlo.exponential %arg0
  // CHECK-DAG: [[COS:%.+]] = mhlo.cosine %arg1
  // CHECK-DAG: [[SIN:%.+]] = mhlo.sine %arg1
  // CHECK-DAG: [[OUTR:%.+]] = mhlo.multiply [[COS]], [[EXP]]
  // CHECK-DAG: [[OUTI:%.+]] = mhlo.multiply [[SIN]], [[EXP]]
  %1 = mhlo.exponential(%0) : (tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  %2 = mhlo.real(%1) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)
  %3 = mhlo.imag(%1) : (tensor<2xcomplex<f32>>) -> (tensor<2xf32>)

  // CHECK: [[OUTR]], [[OUTI]]
  func.return %2, %3 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @exp_complex
func.func @exp_complex(%arg0 : tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>) {
  // CHECK-DAG: [[REAL:%.+]] = mhlo.real(%arg0)
  // CHECK-DAG: [[IMAG:%.+]] = mhlo.imag(%arg0)
  // CHECK-DAG: [[EXP:%.+]] = mhlo.exponential [[REAL]]
  // CHECK-DAG: [[COS:%.+]] = mhlo.cosine [[IMAG]]
  // CHECK-DAG: [[SIN:%.+]] = mhlo.sine [[IMAG]]
  // CHECK-DAG: [[OUTR:%.+]] = mhlo.multiply [[COS]], [[EXP]]
  // CHECK-DAG: [[OUTI:%.+]] = mhlo.multiply [[SIN]], [[EXP]]
  // CHECK-DAG: [[OUT:%.+]] = mhlo.complex([[OUTR]], [[OUTI]])
  %0 = mhlo.exponential(%arg0) : (tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>)

  // CHECK: [[OUT]]
  func.return %0 : tensor<2xcomplex<f32>>
}

// CHECK-LABEL: @exp_unranked
func.func @exp_unranked(%arg0 : tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>) {
  // CHECK-DAG: [[REAL:%.+]] = mhlo.real(%arg0)
  // CHECK-DAG: [[IMAG:%.+]] = mhlo.imag(%arg0)
  // CHECK-DAG: [[EXP:%.+]] = mhlo.exponential [[REAL]]
  // CHECK-DAG: [[COS:%.+]] = mhlo.cosine [[IMAG]]
  // CHECK-DAG: [[SIN:%.+]] = mhlo.sine [[IMAG]]
  // CHECK-DAG: [[OUTR:%.+]] = mhlo.multiply [[COS]], [[EXP]]
  // CHECK-DAG: [[OUTI:%.+]] = mhlo.multiply [[SIN]], [[EXP]]
  // CHECK-DAG: [[OUT:%.+]] = mhlo.complex([[OUTR]], [[OUTI]])
  %0 = mhlo.exponential(%arg0) : (tensor<*xcomplex<f32>>) -> (tensor<*xcomplex<f32>>)

  // CHECK: [[OUT]]
  func.return %0 : tensor<*xcomplex<f32>>
}

// CHECK-LABEL: @compare_eq
// CHECK: ([[LHS:%.+]]: tensor<2xcomplex<f32>>, [[RHS:%.+]]: tensor<2xcomplex<f32>>)
func.func @compare_eq(%lhs : tensor<2xcomplex<f32>>, %rhs: tensor<2xcomplex<f32>>) -> (tensor<2xi1>) {
  // CHECK-DAG: [[REAL_LHS:%.+]] = mhlo.real([[LHS]])
  // CHECK-DAG: [[REAL_RHS:%.+]] = mhlo.real([[RHS]])
  // CHECK-DAG: [[OUTR:%.+]] = mhlo.compare EQ, [[REAL_LHS]], [[REAL_RHS]]
  // CHECK-DAG: [[IMAG_LHS:%.+]] = mhlo.imag([[LHS]])
  // CHECK-DAG: [[IMAG_RHS:%.+]] = mhlo.imag([[RHS]])
  // CHECK-DAG: [[OUTI:%.+]] = mhlo.compare EQ, [[IMAG_LHS]], [[IMAG_RHS]]
  // CHECK-DAG: [[OUT:%.+]] = mhlo.and [[OUTR]], [[OUTI]]
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xi1>

  // CHECK: return [[OUT]]
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: @compare_ne
// CHECK: ([[LHS:%.+]]: tensor<2xcomplex<f32>>, [[RHS:%.+]]: tensor<2xcomplex<f32>>)
func.func @compare_ne(%lhs : tensor<2xcomplex<f32>>, %rhs: tensor<2xcomplex<f32>>) -> (tensor<2xi1>) {
  // CHECK-DAG: [[REAL_LHS:%.+]] = mhlo.real([[LHS]])
  // CHECK-DAG: [[REAL_RHS:%.+]] = mhlo.real([[RHS]])
  // CHECK-DAG: [[OUTR:%.+]] = mhlo.compare NE, [[REAL_LHS]], [[REAL_RHS]]
  // CHECK-DAG: [[IMAG_LHS:%.+]] = mhlo.imag([[LHS]])
  // CHECK-DAG: [[IMAG_RHS:%.+]] = mhlo.imag([[RHS]])
  // CHECK-DAG: [[OUTI:%.+]] = mhlo.compare NE, [[IMAG_LHS]], [[IMAG_RHS]]
  // CHECK-DAG: [[OUT:%.+]] = mhlo.or [[OUTR]], [[OUTI]]
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction NE>} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xi1>

  // CHECK: return [[OUT]]
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: @sin
func.func @sin(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  // CHECK-DAG: %[[TWO:.+]] = mhlo.constant dense<2.000000e+00>
  // CHECK-DAG: %[[SIN:.+]] = mhlo.sine %arg0
  // CHECK-DAG: %[[EXP:.+]] = mhlo.exponential %arg1
  // CHECK-DAG: %[[NEG:.+]] = mhlo.negate %arg1
  // CHECK-DAG: %[[NEXP:.+]] = mhlo.exponential %[[NEG]]
  // CHECK-DAG: %[[ADD:.+]] = mhlo.add %[[EXP]], %[[NEXP]]
  // CHECK-DAG: %[[MUL:.+]] = mhlo.multiply %[[SIN]], %[[ADD]]
  // CHECK-DAG: %[[RDIV:.+]] = mhlo.divide %[[MUL]], %[[TWO]]
  // CHECK-DAG: %[[COS:.+]] = mhlo.cosine %arg0
  // CHECK-DAG: %[[SUB:.+]] = mhlo.subtract %[[EXP]], %[[NEXP]]
  // CHECK-DAG: %[[IMUL:.+]] = mhlo.multiply %[[COS]], %[[SUB]]
  // CHECK-DAG: %[[IDIV:.+]] = mhlo.divide %[[IMUL]], %[[TWO]]
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xcomplex<f32>>)
  %1 = mhlo.sine(%0) : (tensor<10xcomplex<f32>>) -> (tensor<10xcomplex<f32>>)
  %2 = mhlo.real(%1) : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)
  %3 = mhlo.imag(%1) : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)

  // CHECK: return %[[RDIV]], %[[IDIV]]
  func.return %2, %3 : tensor<10xf32>, tensor<10xf32>
}

// CHECK-LABEL: @cos
func.func @cos(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  // CHECK-DAG: %[[TWO:.+]] = mhlo.constant dense<2.000000e+00>
  // CHECK-DAG: %[[COS:.+]] = mhlo.cosine %arg0
  // CHECK-DAG: %[[EXP:.+]] = mhlo.exponential %arg1
  // CHECK-DAG: %[[NEG:.+]] = mhlo.negate %arg1
  // CHECK-DAG: %[[NEXP:.+]] = mhlo.exponential %[[NEG]]
  // CHECK-DAG: %[[ADD:.+]] = mhlo.add %[[EXP]], %[[NEXP]]
  // CHECK-DAG: %[[MUL:.+]] = mhlo.multiply %[[COS]], %[[ADD]]
  // CHECK-DAG: %[[RDIV:.+]] = mhlo.divide %[[MUL]], %[[TWO]]
  // CHECK-DAG: %[[SIN:.+]] = mhlo.sine %arg0
  // CHECK-DAG: %[[SUB:.+]] = mhlo.subtract %[[NEXP]], %[[EXP]]
  // CHECK-DAG: %[[IMUL:.+]] = mhlo.multiply %[[SIN]], %[[SUB]]
  // CHECK-DAG: %[[IDIV:.+]] = mhlo.divide %[[IMUL]], %[[TWO]]
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> (tensor<10xcomplex<f32>>)
  %1 = mhlo.cosine(%0) : (tensor<10xcomplex<f32>>) -> (tensor<10xcomplex<f32>>)
  %2 = mhlo.real(%1) : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)
  %3 = mhlo.imag(%1) : (tensor<10xcomplex<f32>>) -> (tensor<10xf32>)

  // CHECK: return %[[RDIV]], %[[IDIV]]
  func.return %2, %3 : tensor<10xf32>, tensor<10xf32>
}
