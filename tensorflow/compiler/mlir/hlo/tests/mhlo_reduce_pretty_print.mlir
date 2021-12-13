// RUN: mlir-hlo-opt %s  -mlir-print-debuginfo -mlir-print-local-scope \
// RUN: | FileCheck %s
// RUN: mlir-hlo-opt %s  -mlir-print-debuginfo -mlir-print-local-scope \
// RUN: | mlir-hlo-opt -mlir-print-debuginfo -mlir-print-local-scope \
// RUN: | FileCheck %s

// The lit-tests below tests the printing and parsing of the "pretty-printed"
// version of mhlo.reduce op.

// The test case is eligible for pretty-printing reduce-op.

// CHECK-LABEL:  func @reduce_one_op_all_locs_same
// CHECK-NEXT:     mhlo.reduce %arg{{[0-9]+}}, %arg{{[0-9]+}} applies mhlo.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")

func @reduce_one_op_all_locs_same(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>) -> (tensor<?xf32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32> loc("foo"), %arg3: tensor<f32> loc("foo")):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32> loc("foo")
    "mhlo.return"(%1) : (tensor<f32>) -> () loc("foo")
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")

  return %0: tensor<?xf32>
}

// The test case is not eligible for pretty-printing reduce-op. The location of
// reduce-op is different.

// CHECK-LABEL:  func @reduce_one_op_all_locs_not_same_1
// CHECK-NEXT:     mhlo.reduce %arg{{[0-9]+}}, %arg{{[0-9]+}}
// CHECK-NEXT:     ^bb0(%arg[[x:[0-9]+]]: tensor<f32> loc("foo"), %arg[[y:[0-9]+]]: tensor<f32> loc("foo")):
// CHECK-NEXT:       mhlo.add %arg[[x]], %arg[[y]] : tensor<f32> loc("foo")
// CHECK-NEXT:       "mhlo.return"(%{{[0-9]+}}) : (tensor<f32>) -> () loc("foo")
// CHECK-NEXT:     {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("not_foo")

func @reduce_one_op_all_locs_not_same_1(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>) -> (tensor<?xf32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32> loc("foo"), %arg3: tensor<f32> loc("foo")):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32> loc("foo")
    "mhlo.return"(%1) : (tensor<f32>) -> () loc("foo")
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("not_foo")

  return %0: tensor<?xf32>
}

// The test case is not eligible for pretty-printing reduce-op. The location of
// block-arguments are different.

// CHECK-LABEL:  func @reduce_one_op_all_locs_not_same_2
// CHECK-NEXT:     mhlo.reduce
// CHECK-NEXT:     ^bb0

func @reduce_one_op_all_locs_not_same_2(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>) -> (tensor<?xf32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32> loc("foo"), %arg3: tensor<f32> loc("not_foo")):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32> loc("foo")
    "mhlo.return"(%1) : (tensor<f32>) -> () loc("foo")
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")

  return %0: tensor<?xf32>
}


// The test case is not eligible for pretty-printing reduce-op. More than two
// block-arguments which are not perfectly forwarded to inner-op.

// CHECK-LABEL:  func @reduce_one_op_more_than_two_block_args
// CHECK-NEXT:     mhlo.reduce
// CHECK-NEXT:     ^bb0

func @reduce_one_op_more_than_two_block_args(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>) {
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ( {
  ^bb0(%arg4: tensor<f32> loc("foo"), %arg5: tensor<f32> loc("foo"), %arg6: tensor<f32> loc("foo"), %arg7: tensor<f32> loc("foo")):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32> loc("foo")
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> () loc("foo")
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)  loc("foo")

  return %0#0: tensor<?xf32>
}

// The test case is not eligible for pretty-printing reduce-op because of
// non-commutative inner-op.

// CHECK-LABEL:  func @reduce_non_commutative_inner_op
// CHECK-NEXT:     mhlo.reduce
// CHECK-NEXT:     ^bb0

func @reduce_non_commutative_inner_op(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) -> (tensor<?xf32>) {
    %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32> loc("foo"), %arg3: tensor<f32> loc("foo")):
    %1 = "mhlo.divide"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32> loc("foo")
    "mhlo.return"(%1) : (tensor<f32>) -> () loc("foo")
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")

  return %0: tensor<?xf32>
}

// The test case is not eligible for pretty-printing reduce-op because of
// non-binary inner-op.

// CHECK-LABEL:  func @reduce_non_binary_inner_op
// CHECK-NEXT:     mhlo.reduce
// CHECK-NEXT:     ^bb0

func @reduce_non_binary_inner_op(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>) -> (tensor<?xf32>) {
    %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32> loc("foo"), %arg3: tensor<f32> loc("foo")):
    %1 = "mhlo.reshape"(%arg2) : (tensor<f32>) -> tensor<f32> loc("foo")
    "mhlo.return"(%1) : (tensor<f32>) -> () loc("foo")
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")

  return %0: tensor<?xf32>
}

// The test case is not eligible for pretty-printing reduce-op. More than one
// inner-op.

// CHECK-LABEL:  func @reduce_more_than_one_inner_op
// CHECK-NEXT:     mhlo.reduce
// CHECK-NEXT:     ^bb0

func @reduce_more_than_one_inner_op(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) ->
  (tensor<8xf32>, tensor<8xi32>) {
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ( {
  ^bb0(%arg4: tensor<f32> loc("foo"), %arg5: tensor<i32> loc("foo"), %arg6: tensor<f32> loc("foo"), %arg7: tensor<i32> loc("foo")):
    %1 = mhlo.add %arg4, %arg6 : tensor<f32> loc("foo")
    %2 = mhlo.add %arg5, %arg7 : tensor<i32> loc("foo")
    %3 = "mhlo.tuple"(%1, %2) : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>> loc("foo")
    "mhlo.return"(%3) : (tuple<tensor<f32>, tensor<i32>>) -> () loc("foo")
  }) {dimensions = dense<0> : tensor<1xi64>}
    : (tensor<1x8xf32>, tensor<1x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<8xf32>, tensor<8xi32>) loc("foo")

  return %0#0, %0#1 : tensor<8xf32>, tensor<8xi32>
}

// The test case is eligible for pretty-printing reduce-op with complex types.

// CHECK-LABEL:  func @reduce_complex_type
// CHECK:          mhlo.reduce %arg0, %arg1 applies mhlo.add across dimensions = [1] : (tensor<1x2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<1xcomplex<f32>> loc("foo")

func @reduce_complex_type(%arg0: tensor<1x2xcomplex<f32>>, %arg1 : tensor<complex<f32>>) -> (tensor<1xcomplex<f32>>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<complex<f32>> loc("foo"), %arg3: tensor<complex<f32>> loc("foo")):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>> loc("foo")
    "mhlo.return"(%1) : (tensor<complex<f32>>) -> () loc("foo")
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<1xcomplex<f32>> loc("foo")

  return %0: tensor<1xcomplex<f32>>
}

// The test case is not eligible for pretty-printing reduce-op. During parsing
// of the pretty-printed version, we follow the rule of using the element-type
// of reduce-op's `first` input-operand to re-create the type of inner-op. The
// rule is based on the assumption that the pretty-prining  will happen only
// when the above rule is obeyed.  The following tests breaks that rule.

// CHECK-LABEL:  func @reduce_innerop_type_not_trivially_derived
// CHECK-NEXT:     mhlo.reduce
// CHECK-NEXT:     ^bb0

func @reduce_innerop_type_not_trivially_derived(%arg0: tensor<4x4xf32>, %arg1 : tensor<4xf32>) ->
    (tensor<4xf32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "mhlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>

  return %0: tensor<4xf32>
}
