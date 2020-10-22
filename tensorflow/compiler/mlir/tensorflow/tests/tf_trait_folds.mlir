// RUN: tf-opt %s -tf-standard-pipeline | FileCheck %s

// CHECK-LABEL: func @testSingleConj
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<complex<f32>>)
func @testSingleConj(%arg0: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: [[CONJ:%.+]] = "tf.Conj"([[ARG0]])
  %0 = "tf.Conj"(%arg0) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  // CHECK: return [[CONJ]]
  return %0: tensor<complex<f32>>
}

// CHECK-LABEL: func @testDoubleConj
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<complex<f32>>)
func @testDoubleConj(%arg0: tensor<complex<f32>>) -> tensor<complex<f32>> {
  %0 = "tf.Conj"(%arg0) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  %1 = "tf.Conj"(%0) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  // CHECK: return [[ARG0]]
  return %1: tensor<complex<f32>>
}

// CHECK-LABEL: func @testTripleConj
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<complex<f32>>)
func @testTripleConj(%arg0: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: [[CONJ:%.+]] = "tf.Conj"([[ARG0]])
  %0 = "tf.Conj"(%arg0) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  %1 = "tf.Conj"(%0) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  %2 = "tf.Conj"(%1) : (tensor<complex<f32>>) -> tensor<complex<f32>>
  // CHECK: return [[CONJ]]
  return %2: tensor<complex<f32>>
}

// CHECK-LABEL: func @testSingleReciprocal
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testSingleReciprocal(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[RECIPROCAL:%.+]] = "tf.Reciprocal"([[ARG0]])
  %0 = "tf.Reciprocal"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[RECIPROCAL]]
  return %0: tensor<i32>
}

// CHECK-LABEL: func @testDoubleReciprocal
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testDoubleReciprocal(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Reciprocal"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Reciprocal"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[ARG0]]
  return %1: tensor<i32>
}

// CHECK-LABEL: func @testTripleReciprocal
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testTripleReciprocal(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[RECIPROCAL:%.+]] = "tf.Reciprocal"([[ARG0]])
  %0 = "tf.Reciprocal"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Reciprocal"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Reciprocal"(%1) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[RECIPROCAL]]
  return %2: tensor<i32>
}

// CHECK-LABEL: func @testSingleInvert
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testSingleInvert(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[INVERT:%.+]] = "tf.Invert"([[ARG0]])
  %0 = "tf.Invert"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[INVERT]]
  return %0: tensor<i32>
}

// CHECK-LABEL: func @testDoubleInvert
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testDoubleInvert(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Invert"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Invert"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[ARG0]]
  return %1: tensor<i32>
}

// CHECK-LABEL: func @testTripleInvert
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testTripleInvert(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[INVERT:%.+]] = "tf.Invert"([[ARG0]])
  %0 = "tf.Invert"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Invert"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Invert"(%1) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[INVERT]]
  return %2: tensor<i32>
}

// CHECK-LABEL: func @testSingleNeg
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testSingleNeg(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[NEG:%.+]] = "tf.Neg"([[ARG0]])
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[NEG]]
  return %0: tensor<i32>
}

// CHECK-LABEL: func @testDoubleNeg
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testDoubleNeg(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[ARG0]]
  return %1: tensor<i32>
}

// CHECK-LABEL: func @testTripleNeg
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testTripleNeg(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[NEG:%.+]] = "tf.Neg"([[ARG0]])
  %0 = "tf.Neg"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Neg"(%0) : (tensor<i32>) -> tensor<i32>
  %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[NEG]]
  return %2: tensor<i32>
}

// CHECK-LABEL: func @testSingleLogicalNot
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testSingleLogicalNot(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: [[LNOT:%.+]] = "tf.LogicalNot"([[ARG0]])
  %0 = "tf.LogicalNot"(%arg0) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[LNOT]]
  return %0: tensor<i1>
}

// CHECK-LABEL: func @testDoubleLogicalNot
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testDoubleLogicalNot(%arg0: tensor<i1>) -> tensor<i1> {
  %0 = "tf.LogicalNot"(%arg0) : (tensor<i1>) -> tensor<i1>
  %1 = "tf.LogicalNot"(%0) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[ARG0]]
  return %1: tensor<i1>
}

// CHECK-LABEL: func @testTripleLogicalNot
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testTripleLogicalNot(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: [[LNOT:%.+]] = "tf.LogicalNot"([[ARG0]])
  %0 = "tf.LogicalNot"(%arg0) : (tensor<i1>) -> tensor<i1>
  %1 = "tf.LogicalNot"(%0) : (tensor<i1>) -> tensor<i1>
  %2 = "tf.LogicalNot"(%1) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[LNOT]]
  return %2: tensor<i1>
}