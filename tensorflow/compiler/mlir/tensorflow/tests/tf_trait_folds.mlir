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

// CHECK-LABEL: func @testSingleAbs
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testSingleAbs(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[ABS:%.+]] = "tf.Abs"([[ARG0]])
  %0 = "tf.Abs"(%arg0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[ABS]]
  return %0: tensor<i8>
}

// CHECK-LABEL: func @testDoubleAbs
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testDoubleAbs(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[ABS:%.+]] = "tf.Abs"([[ARG0]])
  %0 = "tf.Abs"(%arg0) : (tensor<i8>) -> tensor<i8>
  %1 = "tf.Abs"(%0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[ABS]]
  return %1: tensor<i8>
}

// CHECK-LABEL: func @testSingleCeil
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f1>)
func @testSingleCeil(%arg0: tensor<f1>) -> tensor<f1> {
  // CHECK: [[CEIL:%.+]] = "tf.Ceil"([[ARG0]])
  %0 = "tf.Ceil"(%arg0) : (tensor<f1>) -> tensor<f1>
  // CHECK: return [[CEIL]]
  return %0: tensor<f1>
}

// CHECK-LABEL: func @testDoubleCeil
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f1>)
func @testDoubleCeil(%arg0: tensor<f1>) -> tensor<f1> {
  // CHECK: [[CEIL:%.+]] = "tf.Ceil"([[ARG0]])
  %0 = "tf.Ceil"(%arg0) : (tensor<f1>) -> tensor<f1>
  %1 = "tf.Ceil"(%0) : (tensor<f1>) -> tensor<f1>
  // CHECK: return [[CEIL]]
  return %1: tensor<f1>
}

// CHECK-LABEL: func @testSingleFloor
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f1>)
func @testSingleFloor(%arg0: tensor<f1>) -> tensor<f1> {
  // CHECK: [[FLOOR:%.+]] = "tf.Floor"([[ARG0]])
  %0 = "tf.Floor"(%arg0) : (tensor<f1>) -> tensor<f1>
  // CHECK: return [[FLOOR]]
  return %0: tensor<f1>
}

// CHECK-LABEL: func @testDoubleFloor
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f1>)
func @testDoubleFloor(%arg0: tensor<f1>) -> tensor<f1> {
  // CHECK: [[FLOOR:%.+]] = "tf.Floor"([[ARG0]])
  %0 = "tf.Floor"(%arg0) : (tensor<f1>) -> tensor<f1>
  %1 = "tf.Floor"(%0) : (tensor<f1>) -> tensor<f1>
  // CHECK: return [[FLOOR]]
  return %1: tensor<f1>
}

// CHECK-LABEL: func @testSingleOnesLike
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testSingleOnesLike(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[ONESLIKE:%.+]] = "tf.OnesLike"([[ARG0]])
  %0 = "tf.OnesLike"(%arg0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[ONESLIKE]]
  return %0: tensor<i8>
}

// CHECK-LABEL: func @testDoubleOnesLike
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testDoubleOnesLike(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[ONESLIKE:%.+]] = "tf.OnesLike"([[ARG0]])
  %0 = "tf.OnesLike"(%arg0) : (tensor<i8>) -> tensor<i8>
  %1 = "tf.OnesLike"(%0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[ONESLIKE]]
  return %1: tensor<i8>
}

// CHECK-LABEL: func @testSingleRelu
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testSingleRelu(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[RELU:%.+]] = "tf.Relu"([[ARG0]])
  %0 = "tf.Relu"(%arg0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[RELU]]
  return %0: tensor<i8>
}

// CHECK-LABEL: func @testDoubleRelu
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testDoubleRelu(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[RELU:%.+]] = "tf.Relu"([[ARG0]])
  %0 = "tf.Relu"(%arg0) : (tensor<i8>) -> tensor<i8>
  %1 = "tf.Relu"(%0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[RELU]]
  return %1: tensor<i8>
}

// CHECK-LABEL: func @testSingleRelu6
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testSingleRelu6(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: [[RELU6:%.+]] = "tf.Relu6"([[ARG0]])
  %0 = "tf.Relu6"(%arg0) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[RELU6]]
  return %0: tensor<i1>
}

// CHECK-LABEL: func @testDoubleRelu6
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testDoubleRelu6(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: [[RELU6:%.+]] = "tf.Relu6"([[ARG0]])
  %0 = "tf.Relu6"(%arg0) : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Relu6"(%0) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[RELU6]]
  return %1: tensor<i1>
}

// CHECK-LABEL: func @testSingleRint
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f1>)
func @testSingleRint(%arg0: tensor<f1>) -> tensor<f1> {
  // CHECK: [[RINT:%.+]] = "tf.Rint"([[ARG0]])
  %0 = "tf.Rint"(%arg0) : (tensor<f1>) -> tensor<f1>
  // CHECK: return [[RINT]]
  return %0: tensor<f1>
}

// CHECK-LABEL: func @testDoubleRint
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<f1>)
func @testDoubleRint(%arg0: tensor<f1>) -> tensor<f1> {
  // CHECK: [[RINT:%.+]] = "tf.Rint"([[ARG0]])
  %0 = "tf.Rint"(%arg0) : (tensor<f1>) -> tensor<f1>
  %1 = "tf.Rint"(%0) : (tensor<f1>) -> tensor<f1>
  // CHECK: return [[RINT]]
  return %1: tensor<f1>
}

// CHECK-LABEL: func @testSingleRound
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testSingleRound(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[ROUND:%.+]] = "tf.Round"([[ARG0]])
  %0 = "tf.Round"(%arg0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[ROUND]]
  return %0: tensor<i8>
}

// CHECK-LABEL: func @testDoubleRound
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i8>)
func @testDoubleRound(%arg0: tensor<i8>) -> tensor<i8> {
  // CHECK: [[ROUND:%.+]] = "tf.Round"([[ARG0]])
  %0 = "tf.Round"(%arg0) : (tensor<i8>) -> tensor<i8>
  %1 = "tf.Round"(%0) : (tensor<i8>) -> tensor<i8>
  // CHECK: return [[ROUND]]
  return %1: tensor<i8>
}

// CHECK-LABEL: func @testSingleSign
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testSingleSign(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[SIGN:%.+]] = "tf.Sign"([[ARG0]])
  %0 = "tf.Sign"(%arg0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[SIGN]]
  return %0: tensor<i32>
}

// CHECK-LABEL: func @testDoubleSign
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i32>)
func @testDoubleSign(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: [[SIGN:%.+]] = "tf.Sign"([[ARG0]])
  %0 = "tf.Sign"(%arg0) : (tensor<i32>) -> tensor<i32>
  %1 = "tf.Sign"(%0) : (tensor<i32>) -> tensor<i32>
  // CHECK: return [[SIGN]]
  return %1: tensor<i32>
}

// CHECK-LABEL: func @testSingleZerosLike
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testSingleZerosLike(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: [[ZEROSLIKE:%.+]] = "tf.ZerosLike"([[ARG0]])
  %0 = "tf.ZerosLike"(%arg0) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[ZEROSLIKE]]
  return %0: tensor<i1>
}

// CHECK-LABEL: func @testDoubleZerosLike
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<i1>)
func @testDoubleZerosLike(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: [[ZEROSLIKE:%.+]] = "tf.ZerosLike"([[ARG0]])
  %0 = "tf.ZerosLike"(%arg0) : (tensor<i1>) -> tensor<i1>
  %1 = "tf.ZerosLike"(%0) : (tensor<i1>) -> tensor<i1>
  // CHECK: return [[ZEROSLIKE]]
  return %1: tensor<i1>
}
