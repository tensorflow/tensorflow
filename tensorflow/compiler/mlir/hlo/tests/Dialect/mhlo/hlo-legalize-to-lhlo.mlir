// RUN: mlir-hlo-opt -hlo-legalize-to-lhlo -buffer-hoisting \
// RUN: -buffer-deallocation -split-input-file -cse %s \
// RUN: | FileCheck %s

// CHECK-LABEL: func @attrs
func.func @attrs_copy(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.exponential"(%operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  func.return %result : tensor<2x2xf32>
}

// -----

func.func @return_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  func.return %arg0 : tensor<4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
// CHECK-NEXT: return %[[ARG0]]

// -----

// CHECK-LABEL: func @func_op_long
func.func @func_op_long(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %1 = mhlo.maximum %arg0, %arg1 : tensor<4xf32>
  %2 = mhlo.add %arg0, %1 : tensor<4xf32>
  %3 = mhlo.minimum %arg0, %arg1 : tensor<4xf32>
  %4 = mhlo.subtract %arg1, %3 : tensor<4xf32>
  %5 = mhlo.multiply %2, %4 : tensor<4xf32>
  func.return %5 : tensor<4xf32>
}
//       CHECK: (%[[NEW_ARG0:.*]]: memref<4xf32>, %[[NEW_ARG1:.*]]: memref<4xf32>) -> memref<4xf32>
//  CHECK-NEXT: %[[MAX_RESULT:.*]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.maximum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MAX_RESULT]])
//  CHECK-NEXT: %[[ADD_RESULT:.*]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.add"(%[[NEW_ARG0]], %[[MAX_RESULT]], %[[ADD_RESULT]])
//  CHECK-NEXT: memref.dealloc %[[MAX_RESULT]] : memref<4xf32>
//  CHECK-NEXT: %[[MIN_RESULT:.*]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.minimum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MIN_RESULT]])
//  CHECK-NEXT: %[[SUB_RESULT:.*]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.subtract"(%[[NEW_ARG1]], %[[MIN_RESULT]], %[[SUB_RESULT]])
//  CHECK-NEXT: memref.dealloc %[[MIN_RESULT]] : memref<4xf32>
//  CHECK-NEXT: %[[MUL_RESULT:.*]] = memref.alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.multiply"(%[[ADD_RESULT]], %[[SUB_RESULT]], %[[MUL_RESULT]])
//  CHECK-NEXT: memref.dealloc %[[SUB_RESULT]] : memref<4xf32>
//  CHECK-NEXT: memref.dealloc %[[ADD_RESULT]] : memref<4xf32>
//  CHECK-NEXT: return %[[MUL_RESULT]] : memref<4xf32>

// -----

// CHECK-LABEL: func @fusion
func.func @fusion(%multiplier: tensor<2x2xf32>, %summand_1: tensor<2x2xf32>,
             %summand_2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: (%{{.*}}: {{.*}}, {{.*}}: {{.*}}, {{.*}}: {{.*}})
  // CHECK-NEXT:  %[[ADD_RESULT:.*]] = memref.alloc() : memref<2x2xf32>
  %sum = "mhlo.add"(%summand_1, %summand_2)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "lmhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  // CHECK-NEXT:  %[[MUL_RESULT:.*]] = memref.alloc() : memref<2x2xf32>
  %result = "mhlo.multiply"(%sum, %multiplier)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "lmhlo.multiply"(%[[ADD_RESULT]], %{{.*}}, %[[MUL_RESULT]])
  // CHECK-NEXT:  memref.dealloc %[[ADD_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT:  return %[[MUL_RESULT]] : memref<2x2xf32>
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @copy
func.func @copy(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.copy"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // TODO(herhut): An explicit copy should not be removed.
  // TODO-CHECK: "lmhlo.copy"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @exp
func.func @exp(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.exponential"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @expm1
func.func @expm1(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.exponential_minus_one"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential_minus_one"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @log
func.func @log(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.log"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.log"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @select
func.func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.select"(%pred, %lhs, %rhs)
      : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @compare
func.func @compare(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %result = "mhlo.compare"(%lhs, %rhs)
      {comparison_direction = #mhlo<"comparison_direction EQ">}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // CHECK: "lmhlo.compare"(%{{.*}}, %{{.*}}, %{{.*}}) {comparison_direction = #mhlo<"comparison_direction EQ">}
  func.return %result : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @broadcast
func.func @broadcast(%operand: tensor<5xf32>) -> tensor<10x5xf32> {
  %result = "mhlo.broadcast_in_dim"(%operand)
      {broadcast_dimensions = dense<1> : tensor<1xi64>}
        : (tensor<5xf32>) -> tensor<10x5xf32>
  // CHECK: "lmhlo.broadcast_in_dim"(%{{.*}}, %{{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  func.return %result : tensor<10x5xf32>
}

// -----

// CHECK-LABEL: func @complex
func.func @complex(%real: tensor<2x2xf32>, %imag: tensor<2x2xf32>)
    -> tensor<2x2xcomplex<f32>> {
  %result = "mhlo.complex"(%real, %imag)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  // CHECK: "lmhlo.complex"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_dyn
func.func @complex_dyn(%real: tensor<?xf32>, %imag: tensor<?xf32>)
    -> tensor<?xcomplex<f32>> {
  %result = "mhlo.complex"(%real, %imag)
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xcomplex<f32>>
  // CHECK: "lmhlo.complex"(%{{.*}}, %{{.*}})
  func.return %result : tensor<?xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @real
func.func @real(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32> {
  %result = "mhlo.real"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.real"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @real_dyn
func.func @real_dyn(%operand: tensor<?xcomplex<f32>>) -> tensor<?xf32> {
  %result = "mhlo.real"(%operand)
      : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: "lmhlo.real"(%{{.*}}, %{{.*}})
  func.return %result : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @imag
func.func @imag(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32> {
  %result = "mhlo.imag"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.imag"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @gather
func.func @gather(%operand: tensor<13x7xf32>, %idxs: tensor<5xi32>)
    -> tensor<5x7xf32> {
  %result = "mhlo.gather"(%operand, %idxs) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 1,
      offset_dims = [1],
      start_index_map = [0],
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 7]> : tensor<2xi64>
  } : (tensor<13x7xf32>, tensor<5xi32>) -> tensor<5x7xf32>
  // CHECK: "lmhlo.gather"(%{{.*}}, %{{.*}}, %{{.*}})
  func.return %result : tensor<5x7xf32>
}

// -----

// CHECK-LABEL: func @imag_dyn
func.func @imag_dyn(%operand: tensor<?xcomplex<f32>>) -> tensor<?xf32> {
  %result = "mhlo.imag"(%operand)
      : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: "lmhlo.imag"(%{{.*}}, %{{.*}})
  func.return %result : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @iota
// TODO(herhut): Dummy should not be required here.
func.func @iota(%dummy: tensor<?xf32>) -> tensor<10xi32> {
  %result = "mhlo.iota"()
      {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  // CHECK: "lmhlo.iota"(%{{.*}}) {iota_dimension = 0 : i64}
  func.return %result : tensor<10xi32>
}

// -----

// CHECK-LABEL: func @abs
func.func @abs(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.abs"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.abs"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @and
func.func @and(%operand0: tensor<2x2xi32>, %operand1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.and"(%operand0, %operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.and"(%{{.*}}, %{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @ceil
func.func @ceil(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.ceil"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.ceil"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @convert
func.func @convert(%operand: tensor<2x2xf32>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.convert"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @cos
func.func @cos(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.cosine"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.cosine"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
func.func @floor(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.floor"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.floor"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @neg
func.func @neg(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.negate"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.negate"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @not
func.func @not(%operand: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.not"(%operand)
      : (tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.not"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @or
func.func @or(%operand0: tensor<2x2xi32>, %operand1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.or"(%operand0, %operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.or"(%{{.*}}, %{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @rsqrt
func.func @rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.rsqrt"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @sign
func.func @sign(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.sign"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.sign"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @sqrt
func.func @sqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.sqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.sqrt"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @shift_left
func.func @shift_left(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.shift_left"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_left"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @shift_right_arithmetic
func.func @shift_right_arithmetic(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_arithmetic"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_right_arithmetic"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @shift_right_logical
func.func @shift_right_logical(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_logical"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_right_logical"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @tanh
func.func @tanh(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.tanh"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.tanh"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @remainder
func.func @remainder(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>)
    -> tensor<2x2xf32> {
  %result = "mhlo.remainder"(%lhs, %rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.remainder"(%{{.*}}, %{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @xor
func.func @xor(%operand0: tensor<2x2xi32>, %operand1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.xor"(%operand0, %operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.xor"(%{{.*}}, %{{.*}})
  func.return %result : tensor<2x2xi32>
}

// -----

// Dynamic shape binary element-wise operation.
// CHECK-LABEL: func @add_dyn
func.func @add_dyn(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = "mhlo.add"(%lhs, %rhs)
      : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT:.*]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[C0:.*]] = arith.constant 0
  // CHECK: %[[EE0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<2xindex>
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK: %[[EE1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xindex>
  // CHECK: %[[RESULT:.*]] = memref.alloc(%[[EE0]], %[[EE1]])
  // CHECK: "lmhlo.add"(%arg0, %arg1, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  func.return %result : tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
}

// -----

// Dynamic shape unary element-wise operation.
// CHECK-LABEL: func @tanh_dyn
func.func @tanh_dyn(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = "mhlo.tanh"(%arg0)
      : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT:.*]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[C0:.*]] = arith.constant 0
  // CHECK: %[[EE0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<2xindex>
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK: %[[EE1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xindex>
  // CHECK: %[[RESULT:.*]] = memref.alloc(%[[EE0]], %[[EE1]])
  // CHECK: "lmhlo.tanh"(%arg0, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  func.return %result : tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
}

// -----

// CHECK-LABEL: func @dot
func.func @dot(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-SAME: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc
//      CHECK: "lmhlo.dot"(%[[ARG0]], %[[ARG0]], %[[ALLOC]]) {
//      CHECK:  dot_dimension_numbers =
//      CHECK-NOT:    lhs_batching_dimensions =
//      CHECK-NOT:    rhs_batching_dimensions =
//      CHECK-SAME:   lhs_contracting_dimensions = [1]
//      CHECK-SAME:   rhs_contracting_dimensions = [0]
  %dot = "mhlo.dot"(%arg0, %arg0)
          : (tensor<1024x1024xf32>, tensor<1024x1024xf32>)
              -> tensor<1024x1024xf32>
// CHECK: return %[[ALLOC]]
  func.return %dot : tensor<1024x1024xf32>
}

// -----

// CHECK-LABEL: func @conv
func.func @conv(%input: tensor<3x2x4x3xf32>, %filter : tensor<2x2x3x4xf32>)
    -> tensor<2x1x2x3xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<2x1x2x3xf32>
  // CHECK: lmhlo.convolution(%{{.+}}, %{{.+}}, %[[OUT]])
  // CHECK-SAME{LITERAL}: window = {stride = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
  %out = "mhlo.convolution"(%filter, %input) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 2]> : tensor<2xi64>,
    window_strides = dense<[2, 1]> : tensor<2xi64>
  } : (tensor<2x2x3x4xf32>, tensor<3x2x4x3xf32>) -> tensor<2x1x2x3xf32>
  func.return %out : tensor<2x1x2x3xf32>
}

// -----

// CHECK-LABEL: func @reduce
func.func @reduce(%arg0: tensor<1x8xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<1xf32>
  // CHECK:  "lmhlo.reduce"(%{{.+}}, %{{.+}}, %[[OUT]]) ({
  // CHECK:  ^bb0(%[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<f32>,
  // CHECK-SAME:  %[[ARG3:.*]]: memref<f32>):
  // CHECK:    %[[TMP:.*]] = memref.alloc() : memref<f32>
  // CHECK:    "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[TMP]])
  // CHECK:    "lmhlo.copy"(%[[TMP]], %[[ARG3]])
  // CHECK:    "lmhlo.terminator"() : () -> ()
  // CHECK:  }) {dimensions = dense<1> : tensor<1xi64>}
  // CHECK-SAME: : (memref<1x8xf32>, memref<f32>, memref<1xf32>) -> ()
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @reduce_multiple_operand
func.func @reduce_multiple_operand(%arg0: tensor<1x8xf32>, %arg1: tensor<1x8xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> 
  (tensor<1xf32>, tensor<1xi32>) {
  // CHECK: %[[OUT_F:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: %[[OUT_I:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: "lmhlo.reduce"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[OUT_F]], %[[OUT_I]]) ({
  // CHECK:  ^bb0(%[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<i32>, %[[ARG3:.*]]: memref<f32>, %[[ARG4:.*]]: memref<i32>,
  // CHECK-SAME:  %[[ARG5:.*]]: memref<f32>, %[[ARG6:.*]]: memref<i32>):
  // CHECK:    %[[TMP_OUT0:.*]] = memref.alloc() : memref<f32>
  // CHECK:    "lmhlo.add"(%[[ARG1]], %[[ARG3]], %[[TMP_OUT0]])
  // CHECK:    %[[TMP_OUT1:.*]] = memref.alloc() : memref<i32>
  // CHECK:    "lmhlo.add"(%[[ARG2]], %[[ARG4]], %[[TMP_OUT1]])
  // CHECK:    "lmhlo.copy"(%[[TMP_OUT0]], %[[ARG5]])
  // CHECK:    "lmhlo.copy"(%[[TMP_OUT1]], %[[ARG6]])
  // CHECK:    "lmhlo.terminator"() : () -> ()
  // CHECK:  }) {dimensions = dense<1> : tensor<1xi64>}
  // CHECK-SAME: : (memref<1x8xf32>, memref<1x8xi32>, memref<f32>, memref<i32>, memref<1xf32>, memref<1xi32>) -> ()
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = mhlo.add %arg4, %arg6 : tensor<f32>
    %2 = mhlo.add %arg5, %arg7 : tensor<i32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} 
    : (tensor<1x8xf32>, tensor<1x8xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  func.return %0#0, %0#1 : tensor<1xf32>, tensor<1xi32>
}

// -----

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<1x17x17x64xf32>, %arg1: tensor<f32>) -> tensor<1x8x8x64xf32> {
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<1x8x8x64xf32>
  // CHECK:  "lmhlo.reduce_window"(%{{.+}}, %{{.+}}, %[[OUT]]) ({
  // CHECK:  ^bb0(%[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<f32>,
  // CHECK-SAME:  %[[ARG3:.*]]: memref<f32>):
  // CHECK:    %[[TMP:.*]] = memref.alloc() : memref<f32>
  // CHECK:    "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[TMP]])
  // CHECK:    "lmhlo.copy"(%[[TMP]], %[[ARG3]])
  // CHECK:    "lmhlo.terminator"() : () -> ()
  // CHECK:  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}
  // CHECK-SAME: : (memref<1x17x17x64xf32>, memref<f32>, memref<1x8x8x64xf32>) -> ()
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} 
    : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}

// -----

// CHECK-LABEL: func @reduce_window_multiple_operand
func.func @reduce_window_multiple_operand(%arg0: tensor<1x17x17x64xf32>, %arg1: tensor<1x17x17x64xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> 
  (tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>) {
  // CHECK: %[[OUT_F:.*]] = memref.alloc() : memref<1x8x8x64xf32>
  // CHECK: %[[OUT_I:.*]] = memref.alloc() : memref<1x8x8x64xi32>
  // CHECK: "lmhlo.reduce_window"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[OUT_F]], %[[OUT_I]]) ({
  // CHECK:  ^bb0(%[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<i32>, %[[ARG3:.*]]: memref<f32>, %[[ARG4:.*]]: memref<i32>,
  // CHECK-SAME:  %[[ARG5:.*]]: memref<f32>, %[[ARG6:.*]]: memref<i32>):
  // CHECK:    %[[TMP_OUT0:.*]] = memref.alloc() : memref<f32>
  // CHECK:    "lmhlo.add"(%[[ARG1]], %[[ARG3]], %[[TMP_OUT0]])
  // CHECK:    %[[TMP_OUT1:.*]] = memref.alloc() : memref<i32>
  // CHECK:    "lmhlo.add"(%[[ARG2]], %[[ARG4]], %[[TMP_OUT1]])
  // CHECK:    "lmhlo.copy"(%[[TMP_OUT0]], %[[ARG5]])
  // CHECK:    "lmhlo.copy"(%[[TMP_OUT1]], %[[ARG6]])
  // CHECK:    "lmhlo.terminator"() : () -> ()
  // CHECK:  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}
  // CHECK-SAME: : (memref<1x17x17x64xf32>, memref<1x17x17x64xi32>, memref<f32>, memref<i32>, memref<1x8x8x64xf32>, memref<1x8x8x64xi32>) -> ()
  %0:2 = "mhlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ( {
  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = mhlo.add %arg4, %arg6 : tensor<f32>
    %2 = mhlo.add %arg5, %arg7 : tensor<i32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>}
    : (tensor<1x17x17x64xf32>, tensor<1x17x17x64xi32>, tensor<f32>, tensor<i32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>)
  func.return %0#0, %0#1 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xi32>
}

// -----

// CHECK-LABEL: func @transpose
func.func @transpose(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.transpose"(%operand) {permutation = dense<[1, 0]> : tensor<2xi64>}
              : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.transpose"(%{{.*}}, %{{.*}}) {permutation = dense<[1, 0]> : tensor<2xi64>}
  func.return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call
// CHECK-SAME:([[ARG0:%.*]]: memref<2x2xf32>, [[ARG1:%.*]]: memref<2x3xf32>)
func.func @custom_call(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x4xf16> {
  // CHECK: "lmhlo.custom_call"([[ARG0]], [[ARG1]], %{{.*}}) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>}
  %result = "mhlo.custom_call"(%arg0, %arg1)
              {backend_config = "", call_target_name = "foo", has_side_effect = false}
              : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<4x4xf16>
  func.return %result : tensor<4x4xf16>
}

// -----

// CHECK-LABEL: func @custom_call_multiout
// CHECK-SAME:([[ARG0:%.*]]: memref<2x2xf32>, [[ARG1:%.*]]: memref<2x3xf32>)
func.func @custom_call_multiout(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x4xf16> {
  // CHECK: "lmhlo.custom_call"([[ARG0]], [[ARG1]], %{{.*}}, %{{.*}}) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = dense<2> : vector<2xi32>}
  %temp:2 = "mhlo.custom_call"(%arg0, %arg1)
                   {backend_config = "", call_target_name = "foo", has_side_effect = false}
                   : (tensor<2x2xf32>, tensor<2x3xf32>) -> (tensor<4x4xf16>, tensor<4x4xf16>)
  %result = "mhlo.add"(%temp#0, %temp#1) : (tensor<4x4xf16>, tensor<4x4xf16>) -> tensor<4x4xf16>
  func.return %result : tensor<4x4xf16>
}

// -----

// CHECK-LABEL: func @isfinite
func.func @isfinite(%arg0: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: "lmhlo.is_finite"(%{{.*}}, %{{.*}})
  %result = "mhlo.is_finite"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %result : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @zero_inputs
func.func @zero_inputs() -> tensor<100x100xf32> {
  // CHECK: "lmhlo.constant"(%{{.*}})
  %0 = mhlo.constant dense<0.000000e+00> : tensor<100x100xf32>
  func.return %0 : tensor<100x100xf32>
}

// -----

// CHECK-LABEL: func @clamp
func.func @clamp(%lb : tensor<4xf32>, %x : tensor<4xf32>, %ub : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: "lmhlo.clamp"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  %0 = "mhlo.clamp"(%lb, %x, %ub) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func @clamp_broadcast
func.func @clamp_broadcast(%min: tensor<f32>, %value: tensor<4xf32>, %max: tensor<f32>) -> tensor<4xf32> {
  // CHECK: "lmhlo.clamp"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<f32>, memref<4xf32>, memref<f32>, memref<4xf32>) -> ()
  %0 = "mhlo.clamp"(%min, %value, %max) : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}


