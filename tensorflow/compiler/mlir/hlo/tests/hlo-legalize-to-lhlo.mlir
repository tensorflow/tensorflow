// RUN: mlir-hlo-opt -hlo-legalize-to-lhlo -buffer-hoisting \
// RUN: -buffer-deallocation -split-input-file -cse %s \
// RUN: | FileCheck %s

// CHECK-LABEL: func @attrs
func @attrs_copy(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.exponential"(%operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  return %result : tensor<2x2xf32>
}

// -----

func @return_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
// CHECK-NEXT: return %[[ARG0]]

// -----

// CHECK-LABEL: func @func_op_long
func @func_op_long(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %1 = mhlo.maximum %arg0, %arg1 : tensor<4xf32>
  %2 = mhlo.add %arg0, %1 : tensor<4xf32>
  %3 = mhlo.minimum %arg0, %arg1 : tensor<4xf32>
  %4 = mhlo.subtract %arg1, %3 : tensor<4xf32>
  %5 = mhlo.multiply %2, %4 : tensor<4xf32>
  return %5 : tensor<4xf32>
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
func @fusion(%multiplier: tensor<2x2xf32>, %summand_1: tensor<2x2xf32>,
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
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @copy
func @copy(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.copy"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // TODO(herhut): An explicit copy should not be removed.
  // TODO-CHECK: "lmhlo.copy"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @exp
func @exp(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.exponential"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @expm1
func @expm1(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.exponential_minus_one"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential_minus_one"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @log
func @log(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.log"(%operand) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.log"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @select
func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.select"(%pred, %lhs, %rhs)
      : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @compare
func @compare(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %result = "mhlo.compare"(%lhs, %rhs)
      {comparison_direction = "EQ"}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // CHECK: "lmhlo.compare"(%{{.*}}, %{{.*}}, %{{.*}}) {comparison_direction = "EQ"}
  return %result : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @broadcast
func @broadcast(%operand: tensor<5xf32>) -> tensor<10x5xf32> {
  %result = "mhlo.broadcast_in_dim"(%operand)
      {broadcast_dimensions = dense<1> : tensor<1xi64>}
        : (tensor<5xf32>) -> tensor<10x5xf32>
  // CHECK: "lmhlo.broadcast_in_dim"(%{{.*}}, %{{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  return %result : tensor<10x5xf32>
}

// -----

// CHECK-LABEL: func @complex
func @complex(%real: tensor<2x2xf32>, %imag: tensor<2x2xf32>)
    -> tensor<2x2xcomplex<f32>> {
  %result = "mhlo.complex"(%real, %imag)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  // CHECK: "lmhlo.complex"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_dyn
func @complex_dyn(%real: tensor<?xf32>, %imag: tensor<?xf32>)
    -> tensor<?xcomplex<f32>> {
  %result = "mhlo.complex"(%real, %imag)
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xcomplex<f32>>
  // CHECK: "lmhlo.complex"(%{{.*}}, %{{.*}})
  return %result : tensor<?xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @real
func @real(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32> {
  %result = "mhlo.real"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.real"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @real_dyn
func @real_dyn(%operand: tensor<?xcomplex<f32>>) -> tensor<?xf32> {
  %result = "mhlo.real"(%operand)
      : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: "lmhlo.real"(%{{.*}}, %{{.*}})
  return %result : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @imag
func @imag(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32> {
  %result = "mhlo.imag"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.imag"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @gather
func @gather(%operand: tensor<13x7xf32>, %idxs: tensor<5xi32>)
    -> tensor<5x7xf32> {
  %result =
    "mhlo.gather"(%operand, %idxs)
      { dimension_numbers =
        { collapsed_slice_dims = dense<0> : tensor<1xi64>
        , index_vector_dim = 1 : i64
        , offset_dims = dense<1> : tensor<1xi64>
        , start_index_map = dense<0> : tensor<1xi64> }
      , indices_are_sorted = false
      , name = "gather.71"
      , slice_sizes = dense<[1, 7]> : tensor<2xi64> }
      : (tensor<13x7xf32>, tensor<5xi32>) -> tensor<5x7xf32>
  // CHECK: "lmhlo.gather"(%{{.*}}, %{{.*}}, %{{.*}})
  return %result : tensor<5x7xf32>
}

// -----

// CHECK-LABEL: func @imag_dyn
func @imag_dyn(%operand: tensor<?xcomplex<f32>>) -> tensor<?xf32> {
  %result = "mhlo.imag"(%operand)
      : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: "lmhlo.imag"(%{{.*}}, %{{.*}})
  return %result : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @iota
// TODO(herhut): Dummy should not be required here.
func @iota(%dummy: tensor<?xf32>) -> tensor<10xi32> {
  %result = "mhlo.iota"()
      {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  // CHECK: "lmhlo.iota"(%{{.*}}) {iota_dimension = 0 : i64}
  return %result : tensor<10xi32>
}

// -----

// CHECK-LABEL: func @abs
func @abs(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.abs"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.abs"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @and
func @and(%operand0: tensor<2x2xi32>, %operand1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.and"(%operand0, %operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.and"(%{{.*}}, %{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @ceil
func @ceil(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.ceil"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.ceil"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @convert
func @convert(%operand: tensor<2x2xf32>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.convert"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @cos
func @cos(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.cosine"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.cosine"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
func @floor(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.floor"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.floor"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @neg
func @neg(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.negate"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.negate"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @not
func @not(%operand: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.not"(%operand)
      : (tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.not"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @or
func @or(%operand0: tensor<2x2xi32>, %operand1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.or"(%operand0, %operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.or"(%{{.*}}, %{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @rsqrt
func @rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.rsqrt"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @sign
func @sign(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.sign"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.sign"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @sqrt
func @sqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.sqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.sqrt"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @shift_left
func @shift_left(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.shift_left"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_left"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @shift_right_arithmetic
func @shift_right_arithmetic(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_arithmetic"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_right_arithmetic"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @shift_right_logical
func @shift_right_logical(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_logical"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_right_logical"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @tanh
func @tanh(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.tanh"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.tanh"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @remainder
func @remainder(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>)
    -> tensor<2x2xf32> {
  %result = "mhlo.remainder"(%lhs, %rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.remainder"(%{{.*}}, %{{.*}}, %{{.*}})
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @xor
func @xor(%operand0: tensor<2x2xi32>, %operand1: tensor<2x2xi32>)
    -> tensor<2x2xi32> {
  %result = "mhlo.xor"(%operand0, %operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.xor"(%{{.*}}, %{{.*}})
  return %result : tensor<2x2xi32>
}

// -----

// Dynamic shape binary element-wise operation.
// CHECK-LABEL: func @add_dyn
func @add_dyn(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = "mhlo.add"(%lhs, %rhs)
      : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT:.*]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[C0:.*]] = constant 0
  // CHECK: %[[EE0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<2xindex>
  // CHECK: %[[C1:.*]] = constant 1
  // CHECK: %[[EE1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xindex>
  // CHECK: %[[RESULT:.*]] = memref.alloc(%[[EE0]], %[[EE1]])
  // CHECK: "lmhlo.add"(%arg0, %arg1, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %result : tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
}

// -----

// Dynamic shape unary element-wise operation.
// CHECK-LABEL: func @tanh_dyn
func @tanh_dyn(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = "mhlo.tanh"(%arg0)
      : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[INPUT:.*]] : tensor<?x?xf32> -> tensor<2xindex>
  // CHECK: %[[C0:.*]] = constant 0
  // CHECK: %[[EE0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<2xindex>
  // CHECK: %[[C1:.*]] = constant 1
  // CHECK: %[[EE1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xindex>
  // CHECK: %[[RESULT:.*]] = memref.alloc(%[[EE0]], %[[EE1]])
  // CHECK: "lmhlo.tanh"(%arg0, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return %result : tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
}

// -----

// CHECK-LABEL: func @dot
func @dot(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-SAME: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc
//      CHECK: "lmhlo.dot"(%[[ARG0]], %[[ARG0]], %[[ALLOC]]) {
//        dot_dimension_numbers = {
//          lhs_batching_dimensions = dense<> : tensor<0xi64>,
//          lhs_contracting_dimensions = dense<1> : tensor<1xi64>,
//          rhs_batching_dimensions = dense<> : tensor<0xi64>,
//          rhs_contracting_dimensions = dense<0> : tensor<1xi64>}}
//        : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
  %dot = "mhlo.dot"(%arg0, %arg0)
          : (tensor<1024x1024xf32>, tensor<1024x1024xf32>)
              -> tensor<1024x1024xf32>
// CHECK: return %[[ALLOC]]
  return %dot : tensor<1024x1024xf32>
}

// -----

// CHECK-LABEL: func @conv
func @conv(%input: tensor<3x5x5x3xf32>, %filter : tensor<2x2x3x4xf32>)
    -> tensor<3x5x5x4xf32> {
  %c0 = constant 0 : index
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<3x5x5x4xf32>
  // CHECK: lmhlo.convolution(%{{.+}}, %{{.+}}, %[[OUT]])
  // CHECK-SAME{LITERAL}: window = {stride = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
  %out = "mhlo.convolution"(%filter, %input) {
    batch_group_count = 1 : i64,
    dimension_numbers = {
      input_batch_dimension = 0 : i64,
      input_feature_dimension = 3 : i64,
      input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>,
      kernel_input_feature_dimension = 2 : i64,
      kernel_output_feature_dimension = 3 : i64,
      kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>,
      output_batch_dimension = 0 : i64,
      output_feature_dimension = 3 : i64,
      output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>
    },
    feature_group_count = 1 : i64,
    padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<[1, 2]> : tensor<2xi64>,
    window_strides = dense<[2, 1]> : tensor<2xi64>
  } : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  return %out : tensor<3x5x5x4xf32>
}

// -----

// CHECK-LABEL: func @reduce
func @reduce(%arg0: tensor<1x8xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<1xf32>
  // CHECK:  "lmhlo.reduce"(%{{.+}}, %{{.+}}, %[[OUT]]) ( {
  // CHECK:  ^bb0(%[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<f32>,
  // CHECK-SAME:  %[[ARG3:.*]]: memref<f32>):
  // CHECK:    %[[TMP:.*]] = memref.alloc() : memref<f32>
  // CHECK:    "lmhlo.add"(%[[ARG1]], %[[ARG2]], %[[TMP]])
  // CHECK:    "lmhlo.copy"(%[[TMP]], %[[ARG3]])
  // CHECK:    "lmhlo.terminator"() : () -> ()
  // CHECK:  }) {dimensions = dense<1> : tensor<1xi64>}
  // CHECK-SAME: : (memref<1x8xf32>, memref<f32>, memref<1xf32>) -> ()
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: func @transpose
func @transpose(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = "mhlo.transpose"(%operand) {permutation = dense<[1, 0]> : tensor<2xi64>}
              : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.transpose"(%{{.*}}, %{{.*}}) {permutation = dense<[1, 0]> : tensor<2xi64>}
  return %result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @custom_call
// CHECK-SAME:([[ARG0:%.*]]: memref<2x2xf32>, [[ARG1:%.*]]: memref<2x3xf32>)
func @custom_call(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x4xf16> {
  // CHECK: "lmhlo.custom_call"([[ARG0]], [[ARG1]], %{{.*}}) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>}
  %result = "mhlo.custom_call"(%arg0, %arg1)
              {backend_config = "", call_target_name = "foo", has_side_effect = false}
              : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<4x4xf16>
  return %result : tensor<4x4xf16>
}

// -----

// CHECK-LABEL: func @custom_call_multiout
// CHECK-SAME:([[ARG0:%.*]]: memref<2x2xf32>, [[ARG1:%.*]]: memref<2x3xf32>)
func @custom_call_multiout(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<4x4xf16> {
  // CHECK: "lmhlo.custom_call"([[ARG0]], [[ARG1]], %{{.*}}, %{{.*}}) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = dense<2> : vector<2xi32>}
  %temp:2 = "mhlo.custom_call"(%arg0, %arg1)
                   {backend_config = "", call_target_name = "foo", has_side_effect = false}
                   : (tensor<2x2xf32>, tensor<2x3xf32>) -> (tensor<4x4xf16>, tensor<4x4xf16>)
  %result = "mhlo.add"(%temp#0, %temp#1) : (tensor<4x4xf16>, tensor<4x4xf16>) -> tensor<4x4xf16>
  return %result : tensor<4x4xf16>
}

// -----

// CHECK-LABEL: func @isfinite
func @isfinite(%arg0: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: "lmhlo.is_finite"(%{{.*}}, %{{.*}})
  %result = "mhlo.is_finite"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  return %result : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: func @zero_inputs
func @zero_inputs() -> tensor<100x100xf32> {
  // CHECK: "lmhlo.constant"(%{{.*}})
  %0 = mhlo.constant dense<0.000000e+00> : tensor<100x100xf32>
  return %0 : tensor<100x100xf32>
}


