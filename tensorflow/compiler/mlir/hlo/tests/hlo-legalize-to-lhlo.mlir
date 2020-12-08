// RUN: mlir-hlo-opt -hlo-legalize-to-lhlo -buffer-hoisting \
// RUN: -buffer-deallocation -split-input-file -cse %s -o - \
// RUN: | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: func @attrs
func @attrs_copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.exponential"(%tensor_operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

func @return_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
//  CHECK-NOT: "lmhlo.copy"
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
//  CHECK-NEXT: %[[MAX_RESULT:.*]] = alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.maximum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MAX_RESULT]])
//  CHECK-NEXT: %[[ADD_RESULT:.*]] = alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.add"(%[[NEW_ARG0]], %[[MAX_RESULT]], %[[ADD_RESULT]])
//  CHECK-NEXT: dealloc %[[MAX_RESULT]] : memref<4xf32>
//  CHECK-NEXT: %[[MIN_RESULT:.*]] = alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.minimum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MIN_RESULT]])
//  CHECK-NEXT: %[[SUB_RESULT:.*]] = alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.subtract"(%[[NEW_ARG1]], %[[MIN_RESULT]], %[[SUB_RESULT]])
//  CHECK-NEXT: dealloc %[[MIN_RESULT]] : memref<4xf32>
//  CHECK-NEXT: %[[MUL_RESULT:.*]] = alloc() : memref<4xf32>
//  CHECK-NEXT: "lmhlo.multiply"(%[[ADD_RESULT]], %[[SUB_RESULT]], %[[MUL_RESULT]])
//  CHECK-NEXT: dealloc %[[SUB_RESULT]] : memref<4xf32>
//  CHECK-NEXT: dealloc %[[ADD_RESULT]] : memref<4xf32>
//  CHECK-NEXT: return %[[MUL_RESULT]] : memref<4xf32>

// -----

// CHECK-LABEL: func @fusion
func @fusion(%multiplier: memref<2x2xf32>, %summand_1: memref<2x2xf32>,
             %summand_2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  // CHECK: (%{{.*}}: {{.*}}, {{.*}}: {{.*}}, {{.*}}: {{.*}}, %[[RESULT:.*]]: {{.*}})
  // CHECK-NEXT:  %[[ADD_RESULT:.*]] = alloc() : memref<2x2xf32>
  %tensor_summand_1 = tensor_load %summand_1 : memref<2x2xf32>
  %tensor_summand_2 = tensor_load %summand_2 : memref<2x2xf32>
  %sum = "mhlo.add"(%tensor_summand_1, %tensor_summand_2)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "lmhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  // CHECK-NEXT:  %[[MUL_RESULT:.*]] = alloc() : memref<2x2xf32>
  %tensor_multiplier = tensor_load %multiplier : memref<2x2xf32>
  %tensor_result = "mhlo.multiply"(%sum, %tensor_multiplier)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "lmhlo.multiply"(%[[ADD_RESULT]], %{{.*}}, %[[MUL_RESULT]])
  // CHECK-NEXT:  dealloc %[[ADD_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT: "lmhlo.copy"(%[[MUL_RESULT]], %[[RESULT]])
  tensor_store %tensor_result, %result : memref<2x2xf32>
  // CHECK-NEXT:  dealloc %[[MUL_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK-LABEL: func @copy
func @copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.copy"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.copy"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @exp
func @exp(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.exponential"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.exponential"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @log
func @log(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.log"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.log"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @select
func @select(%pred: memref<2x2xi1>, %lhs: memref<2x2xf32>,
             %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_pred = tensor_load %pred : memref<2x2xi1>
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "mhlo.select"(%tensor_pred, %tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @compare
func @compare(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xi1>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "mhlo.compare"(%tensor_lhs, %tensor_rhs)
      {comparison_direction = "EQ"}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // CHECK: "lmhlo.compare"(%{{.*}}, %{{.*}}, %{{.*}}) {comparison_direction = "EQ"}
  tensor_store %tensor_result, %result : memref<2x2xi1>
  return
}

// -----

// CHECK-LABEL: func @broadcast
func @broadcast(%operand: memref<5xf32>, %result: memref<10x5xf32>) {
  %tensor_operand = tensor_load %operand : memref<5xf32>
  %tensor_result = "mhlo.broadcast_in_dim"(%tensor_operand)
      {broadcast_dimensions = dense<1> : tensor<1xi64>}
        : (tensor<5xf32>) -> tensor<10x5xf32>
  // CHECK: "lmhlo.broadcast_in_dim"(%{{.*}}, %{{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<10x5xf32>
  return
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + d2 * s2)>

// CHECK-LABEL: func @dyn_broadcast
func @dyn_broadcast(%operand: memref<?x?xf32>) -> index {
  // CHECK-SAME: %[[OPERAND:.*]]: memref<?x?xf32>
  %tensor_operand = tensor_load %operand : memref<?x?xf32>
  %c1 = constant 1 : i64
  %shape = tensor_from_elements %c1, %c1, %c1 : tensor<3xi64>
  %tensor_result = "mhlo.dynamic_broadcast_in_dim"(%tensor_operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  %rank = rank %tensor_result : tensor<?x?x?xf32>
  return %rank : index
}
// CHECK: %[[SHAPE:.*]] = tensor_from_elements
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[EL0:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C0]]] : tensor<3xi64>
// CHECK: %[[SIZE_0:.*]] = index_cast %[[EL0]] : i64 to index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[EL1:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C1]]] : tensor<3xi64>
// CHECK: %[[SIZE_1:.*]] = index_cast %[[EL1]] : i64 to index
// CHECK: %[[C2:.*]] = constant 2 : index
// CHECK: %[[EL2:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C2]]] : tensor<3xi64>
// CHECK: %[[SIZE_2:.*]] = index_cast %[[EL2]] : i64 to index
// CHECK: %[[RESULT:.*]] = alloc(%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]) : memref<?x?x?xf32>
// CHECK: %[[OPER_DIM_1:.*]] = dim %[[OPERAND]], %[[C1]] : memref<?x?xf32>
// CHECK: %[[OP_STRIDE_0:.*]] = muli %[[C1]], %[[OPER_DIM_1]] : index
// CHECK: %[[OPER_DIM_0:.*]] = dim %[[OPERAND]], %[[C0]] : memref<?x?xf32>
// CHECK: %[[EXPAND_1:.*]] = cmpi "slt", %[[OPER_DIM_0]], %[[SIZE_1]] : index
// CHECK: %[[STRIDE_1:.*]] = select %[[EXPAND_1]], %[[C0]], %[[OP_STRIDE_0]] : index
// CHECK: %[[EXPAND_2:.*]] = cmpi "slt", %[[OPER_DIM_1]], %[[SIZE_2]] : index
// CHECK: %[[STRIDE_2:.*]] = select %[[EXPAND_2]], %[[C0]], %[[C1]] : index
// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref_reinterpret_cast %[[OPERAND]] to offset: [0], sizes: {{\[}}%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]], strides: {{\[}}%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]]: memref<?x?xf32> to memref<?x?x?xf32, #map>
// CHECK: "lmhlo.copy"(%[[TRANSFORMED_MEMREF]], %[[RESULT]]) : (memref<?x?x?xf32, #map>, memref<?x?x?xf32>) -> ()
// CHECK: dealloc %[[RESULT]] : memref<?x?x?xf32>

// -----

// CHECK-LABEL: func @complex
func @complex(%real: memref<2x2xf32>,
              %imag: memref<2x2xf32>,
              %result: memref<2x2xcomplex<f32>>) {
  %tensor_real = tensor_load %real : memref<2x2xf32>
  %tensor_imag = tensor_load %imag : memref<2x2xf32>
  %tensor_result = "mhlo.complex"(%tensor_real, %tensor_imag)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  // CHECK: "lmhlo.complex"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func @complex_dyn
func @complex_dyn(%real: memref<?xf32>,
                  %imag: memref<?xf32>,
                  %result: memref<?xcomplex<f32>>) {
  %tensor_real = tensor_load %real : memref<?xf32>
  %tensor_imag = tensor_load %imag : memref<?xf32>
  %tensor_result = "mhlo.complex"(%tensor_real, %tensor_imag)
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xcomplex<f32>>
  // CHECK: "lmhlo.complex"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<?xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func @real
func @real(%operand: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xcomplex<f32>>
  %tensor_result = "mhlo.real"(%tensor_operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.real"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @real_dyn
func @real_dyn(%operand: memref<?xcomplex<f32>>, %result: memref<?xf32>) {
  %tensor_operand = tensor_load %operand : memref<?xcomplex<f32>>
  %tensor_result = "mhlo.real"(%tensor_operand)
      : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: "lmhlo.real"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func @imag
func @imag(%operand: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xcomplex<f32>>
  %tensor_result = "mhlo.imag"(%tensor_operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.imag"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @gather
func @gather(%operand: memref<13x7xf32>, %idxs: memref<5xi32>, %result: memref<5x7xf32>) {
  %tensor_operand = tensor_load %operand : memref<13x7xf32>
  %tensor_idxs = tensor_load %idxs : memref<5xi32>
  %tensor_result =
    "mhlo.gather"(%tensor_operand, %tensor_idxs)
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
  tensor_store %tensor_result, %result : memref<5x7xf32>
  return
}

// -----

// CHECK-LABEL: func @imag_dyn
func @imag_dyn(%operand: memref<?xcomplex<f32>>, %result: memref<?xf32>) {
  %tensor_operand = tensor_load %operand : memref<?xcomplex<f32>>
  %tensor_result = "mhlo.imag"(%tensor_operand)
      : (tensor<?xcomplex<f32>>) -> tensor<?xf32>
  // CHECK: "lmhlo.imag"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func @iota
func @iota(%result: memref<10xi32>) {
  %tensor_result = "mhlo.iota"()
      {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  // CHECK: "lmhlo.iota"(%{{.*}}) {iota_dimension = 0 : i64}
  tensor_store %tensor_result, %result : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abs
func @abs(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.abs"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.abs"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @and
func @and(%operand0: memref<2x2xi32>, %operand1: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  %tensor_operand0 = tensor_load %operand0 : memref<2x2xi32>
  %tensor_operand1 = tensor_load %operand1 : memref<2x2xi32>
  %tensor_result = "mhlo.and"(%tensor_operand0, %tensor_operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.and"(%{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @ceil
func @ceil(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.ceil"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.ceil"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @convert
func @convert(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.convert"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.copy"(%{{.*}}, %{{.*}})
  // CHECK-NOT: tensor_store
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @cos
func @cos(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.cosine"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.cosine"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @floor
func @floor(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.floor"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.floor"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @neg
func @neg(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.negate"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.negate"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @not
func @not(%operand: memref<2x2xi32>, %result: memref<2x2xi32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xi32>
  %tensor_result = "mhlo.not"(%tensor_operand)
      : (tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.not"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @or
func @or(%operand0: memref<2x2xi32>, %operand1: memref<2x2xi32>,
         %result: memref<2x2xi32>) {
  %tensor_operand0 = tensor_load %operand0 : memref<2x2xi32>
  %tensor_operand1 = tensor_load %operand1 : memref<2x2xi32>
  %tensor_result = "mhlo.or"(%tensor_operand0, %tensor_operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.or"(%{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @rsqrt
func @rsqrt(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.rsqrt"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.rsqrt"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @sign
func @sign(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.sign"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.sign"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @sqrt
func @sqrt(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.sqrt"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.sqrt"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @shift_left
func @shift_left(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
                 %result: memref<2x2xi32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xi32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xi32>
  %tensor_result = "mhlo.shift_left"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_left"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @shift_right_arithmetic
func @shift_right_arithmetic(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
                             %result: memref<2x2xi32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xi32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xi32>
  %tensor_result = "mhlo.shift_right_arithmetic"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_right_arithmetic"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @shift_right_logical
func @shift_right_logical(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
                          %result: memref<2x2xi32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xi32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xi32>
  %tensor_result = "mhlo.shift_right_logical"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.shift_right_logical"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// CHECK-LABEL: func @tanh
func @tanh(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.tanh"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.tanh"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @remainder
func @remainder(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
                %result: memref<2x2xf32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "mhlo.remainder"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.remainder"(%{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @xor
func @xor(%operand0: memref<2x2xi32>, %operand1: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  %tensor_operand0 = tensor_load %operand0 : memref<2x2xi32>
  %tensor_operand1 = tensor_load %operand1 : memref<2x2xi32>
  %tensor_result = "mhlo.xor"(%tensor_operand0, %tensor_operand1)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // CHECK: "lmhlo.xor"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xi32>
  return
}

// -----

// Dynamic shape binary element-wise operation.
// CHECK-LABEL: func @add_dyn
func @add_dyn(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) {
  %result = "mhlo.add"(%lhs, %rhs)
      : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[DIM0:.*]] = dim %arg0, %[[C0]] : memref<?x?xf32>
  // CHECK: %[[IC0:.*]] = index_cast %[[DIM0]] : index to i64
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[DIM1:.*]] = dim %arg0, %[[C1]] : memref<?x?xf32>
  // CHECK: %[[IC1:.*]] = index_cast %[[DIM1]] : index to i64
  // CHECK: %[[SHAPE:.*]] = tensor_from_elements %[[IC0]], %[[IC1]] : tensor<2xi64>
  // CHECK: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<2xi64>
  // CHECK: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // CHECK: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<2xi64>
  // CHECK: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // CHECK: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // CHECK: "lmhlo.add"(%arg0, %arg1, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

// Dynamic shape unary element-wise operation.
// CHECK-LABEL: func @tanh_dyn
func @tanh_dyn(%arg0: tensor<?x?xf32>) {
  %result = "mhlo.tanh"(%arg0)
      : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[DIM0:.*]] = dim %arg0, %[[C0]] : memref<?x?xf32>
  // CHECK: %[[IC0:.*]] = index_cast %[[DIM0]] : index to i64
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[DIM1:.*]] = dim %arg0, %[[C1]] : memref<?x?xf32>
  // CHECK: %[[IC1:.*]] = index_cast %[[DIM1]] : index to i64
  // CHECK: %[[SHAPE:.*]] = tensor_from_elements %[[IC0]], %[[IC1]] : tensor<2xi64>
  // CHECK: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<2xi64>
  // CHECK: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // CHECK: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<2xi64>
  // CHECK: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // CHECK: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // CHECK: "lmhlo.tanh"(%arg0, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @dot
func @dot(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-SAME: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
// CHECK-NEXT: %[[ALLOC:.*]] = alloc
//      CHECK: "lmhlo.dot"(%[[ARG0]], %[[ARG0]], %[[ALLOC]]) {
//        dot_dimension_numbers = {
//          lhs_batching_dimensions = dense<> : tensor<0xi64>,
//          lhs_contracting_dimensions = dense<1> : tensor<1xi64>,
//          rhs_batching_dimensions = dense<> : tensor<0xi64>,
//          rhs_contracting_dimensions = dense<0> : tensor<1xi64>}}
//        : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
  %dot = "mhlo.dot"(%arg0, %arg0)
          : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK: return %[[ALLOC]]
  return %dot : tensor<1024x1024xf32>
}

// -----

// CHECK-LABEL: func @conv
func @conv(%input: tensor<3x5x5x3xf32>, %filter : tensor<2x2x3x4xf32>) -> tensor<3x5x5x4xf32> {
  %c0 = constant 0 : index
  // CHECK: %[[OUT:.*]] = alloc() : memref<3x5x5x4xf32>
  // CHECK: "lmhlo.convolution"(%{{.+}}, %{{.+}}, %[[OUT]])
  // CHECK-SAME: padding = dense<[
  // CHECK-SAME:                  [0, 1], [0, 1]]> : tensor<2x2xi64>
  // CHECK-SAME: rhs_dilation = dense<[1, 2]>
  // CHECK-SAME: window_strides = dense<[2, 1]>
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
  // CHECK: %[[OUT:.*]] = alloc() : memref<1xf32>
  // CHECK:  "lmhlo.reduce"(%{{.+}}, %{{.+}}, %[[OUT]]) ( {
  // CHECK:  ^bb0(%[[ARG1:.*]]: memref<f32>, %[[ARG2:.*]]: memref<f32>,
  // CHECK-SAME:  %[[ARG3:.*]]: memref<f32>):
  // CHECK:    %[[TMP:.*]] = alloc() : memref<f32>
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
func @transpose(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "mhlo.transpose"(%tensor_operand) {permutation = dense<[1, 0]> : tensor<2xi64>}
                    : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "lmhlo.transpose"(%{{.*}}, %{{.*}}) {permutation = dense<[1, 0]> : tensor<2xi64>}
  // CHECK-NOT: tensor_store
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @custom_call
// CHECK-SAME:([[ARG0:%.*]]: memref<2x2xf32>, [[ARG1:%.*]]: memref<2x3xf32>, [[RESULT:%.*]]: memref<4x4xf16>)
func @custom_call(%arg0: memref<2x2xf32>, %arg1: memref<2x3xf32>, %result: memref<4x4xf16>) {
  %arg0_tensor = tensor_load %arg0 : memref<2x2xf32>
  %arg1_tensor = tensor_load %arg1 : memref<2x3xf32>
  // CHECK: "lmhlo.custom_call"([[ARG0]], [[ARG1]], %{{.*}}) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = dense<[2, 1]> : vector<2xi32>}
  %result_tensor = "mhlo.custom_call"(%arg0_tensor, %arg1_tensor)
                   {backend_config = "", call_target_name = "foo", has_side_effect = false}
                   : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<4x4xf16>
  tensor_store %result_tensor, %result: memref<4x4xf16>
  return
}

// ----

// CHECK-LABEL: func @custom_call_multiout
// CHECK-SAME:([[ARG0:%.*]]: memref<2x2xf32>, [[ARG1:%.*]]: memref<2x3xf32>, [[RESULT:%.*]]: memref<4x4xf16>)
func @custom_call_multiout(%arg0: memref<2x2xf32>, %arg1: memref<2x3xf32>, %result: memref<4x4xf16>) {
  %arg0_tensor = tensor_load %arg0 : memref<2x2xf32>
  %arg1_tensor = tensor_load %arg1 : memref<2x3xf32>
  // CHECK: "lmhlo.custom_call"([[ARG0]], [[ARG1]], %{{.*}}, %{{.*}}) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = dense<2> : vector<2xi32>}
  %temp:2 = "mhlo.custom_call"(%arg0_tensor, %arg1_tensor)
                   {backend_config = "", call_target_name = "foo", has_side_effect = false}
                   : (tensor<2x2xf32>, tensor<2x3xf32>) -> (tensor<4x4xf16>, tensor<4x4xf16>)
  %result_tensor = "mhlo.add"(%temp#0, %temp#1) : (tensor<4x4xf16>, tensor<4x4xf16>) -> tensor<4x4xf16>
  tensor_store %result_tensor, %result: memref<4x4xf16>
  return
}

// ----

// CHECK-LABEL: func @isfinite
func @isfinite(%arg0: memref<2x2xf32>, %result: memref<2x2xi1>) {
  %arg0_tensor = tensor_load %arg0 : memref<2x2xf32>
  // CHECK: "lmhlo.is_finite"(%{{.*}}, %{{.*}})
  %result_tensor = "mhlo.is_finite"(%arg0_tensor) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  tensor_store %result_tensor, %result: memref<2x2xi1>
  return
}

// -----

// Test that assuming ops propagate memref types.
// CHECK-LABEL: func @shape_assuming_memref
func @shape_assuming_memref(%arg0: tensor<?xf16>) -> tensor<?xf16> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
  %1 = shape.const_witness true
  // CHECK: shape.assuming %{{.*}} -> (memref<?xf16>)
  %2 = shape.assuming %1 -> (tensor<?xf16>) {
    %3 = shape.shape_of %arg0 : tensor<?xf16> -> tensor<?xindex>
    %4 = tensor_cast %3 : tensor<?xindex> to tensor<1xindex>
    %5 = "mhlo.dynamic_broadcast_in_dim"(%0, %4) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f16>, tensor<1xindex>) -> tensor<?xf16>
    %6 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %4) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf16>, tensor<1xindex>) -> tensor<?xf16>
    // CHECK: "lmhlo.maximum"(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?xf16>, memref<?xf16>, memref<?xf16>) -> ()
    %7 = mhlo.maximum %5, %6 : tensor<?xf16>
    // CHECK: shape.assuming_yield %{{.*}} : memref<?xf16>
    shape.assuming_yield %7 : tensor<?xf16>
  }
  return %2 : tensor<?xf16>
}
