// RUN: xla-opt -hlo-legalize-to-lhlo -buffer-placement -split-input-file %s -o - | FileCheck --check-prefixes=PRE,BOTH %s
// RUN: xla-opt -hlo-legalize-to-lhlo=results-escape-function=true -buffer-placement -split-input-file %s -o - | FileCheck --check-prefixes=ESC,BOTH %s

// BOTH-LABEL: func @attrs
func @attrs_copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exponential"(%tensor_operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.exponential"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

func @return_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
//      PRE: (%[[ARG0:.*]]: [[TYPE:.*]], %[[RESULT:.*]]: [[TYPE]])
// PRE-NEXT: "xla_lhlo.copy"(%[[ARG0]], %[[RESULT]]) : ([[TYPE]], [[TYPE]]) -> ()
// PRE-NEXT: return
//      ESC: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
//  ESC-NOT: "xla_lhlo.copy"
// ESC-NEXT: return %[[ARG0]]

// -----

// BOTH-LABEL: func @func_op_long
func @func_op_long(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %1 = xla_hlo.maximum %arg0, %arg1 : tensor<4xf32>
  %2 = xla_hlo.add %arg0, %1 : tensor<4xf32>
  %3 = xla_hlo.minimum %arg0, %arg1 : tensor<4xf32>
  %4 = xla_hlo.subtract %arg1, %3 : tensor<4xf32>
  %5 = xla_hlo.multiply %2, %4 : tensor<4xf32>
  return %5 : tensor<4xf32>
}
//        PRE: (%[[NEW_ARG0:.*]]: memref<4xf32>, %[[NEW_ARG1:.*]]: memref<4xf32>, %[[RESULT:.*]]: memref<4xf32>)
//        ESC: (%[[NEW_ARG0:.*]]: memref<4xf32>, %[[NEW_ARG1:.*]]: memref<4xf32>) -> memref<4xf32>
//  BOTH-NEXT: %[[MAX_RESULT:.*]] = alloc() : memref<4xf32>
//  BOTH-NEXT: "xla_lhlo.maximum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MAX_RESULT]])
//  BOTH-NEXT: %[[ADD_RESULT:.*]] = alloc() : memref<4xf32>
//  BOTH-NEXT: "xla_lhlo.add"(%[[NEW_ARG0]], %[[MAX_RESULT]], %[[ADD_RESULT]])
//  BOTH-NEXT: dealloc %[[MAX_RESULT]] : memref<4xf32>
//  BOTH-NEXT: %[[MIN_RESULT:.*]] = alloc() : memref<4xf32>
//  BOTH-NEXT: "xla_lhlo.minimum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MIN_RESULT]])
//  BOTH-NEXT: %[[SUB_RESULT:.*]] = alloc() : memref<4xf32>
//  BOTH-NEXT: "xla_lhlo.subtract"(%[[NEW_ARG1]], %[[MIN_RESULT]], %[[SUB_RESULT]])
//  BOTH-NEXT: dealloc %[[MIN_RESULT]] : memref<4xf32>
//  BOTH-NEXT: %[[MUL_RESULT:.*]] = alloc() : memref<4xf32>
//  BOTH-NEXT: "xla_lhlo.multiply"(%[[ADD_RESULT]], %[[SUB_RESULT]], %[[MUL_RESULT]])
//  BOTH-NEXT: dealloc %[[SUB_RESULT]] : memref<4xf32>
//  BOTH-NEXT: dealloc %[[ADD_RESULT]] : memref<4xf32>
//   PRE-NEXT: "xla_lhlo.copy"(%[[MUL_RESULT]], %[[RESULT]]) : (memref<4xf32>, memref<4xf32>) -> ()
//   PRE-NEXT: dealloc %[[MUL_RESULT]] : memref<4xf32>
//   PRE-NEXT: return
//   ESC-NEXT: return %[[MUL_RESULT]] : memref<4xf32>

// -----

// BOTH-LABEL: func @fusion
func @fusion(%multiplier: memref<2x2xf32>, %summand_1: memref<2x2xf32>,
             %summand_2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  // BOTH: (%{{.*}}: {{.*}}, {{.*}}: {{.*}}, {{.*}}: {{.*}}, %[[RESULT:.*]]: {{.*}})
  // BOTH-NEXT:  %[[ADD_RESULT:.*]] = alloc() : memref<2x2xf32>
  %tensor_summand_1 = tensor_load %summand_1 : memref<2x2xf32>
  %tensor_summand_2 = tensor_load %summand_2 : memref<2x2xf32>
  %sum = "xla_hlo.add"(%tensor_summand_1, %tensor_summand_2)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH-NEXT: "xla_lhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  // BOTH-NEXT:  %[[MUL_RESULT:.*]] = alloc() : memref<2x2xf32>
  %tensor_multiplier = tensor_load %multiplier : memref<2x2xf32>
  %tensor_result = "xla_hlo.multiply"(%sum, %tensor_multiplier)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH-NEXT: "xla_lhlo.multiply"(%[[ADD_RESULT]], %{{.*}}, %[[MUL_RESULT]])
  // BOTH-NEXT:  dealloc %[[ADD_RESULT]] : memref<2x2xf32>
  // BOTH-NEXT: "xla_lhlo.copy"(%[[MUL_RESULT]], %[[RESULT]])
  tensor_store %tensor_result, %result : memref<2x2xf32>
  // BOTH-NEXT:  dealloc %[[MUL_RESULT]] : memref<2x2xf32>
  // BOTH-NEXT:  return
  return
}

// -----

// BOTH-LABEL: func @copy
func @copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.copy"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.copy"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @exp
func @exp(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exponential"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.exponential"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @log
func @log(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.log"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.log"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @select
func @select(%pred: memref<2x2xi1>, %lhs: memref<2x2xf32>,
             %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_pred = tensor_load %pred : memref<2x2xi1>
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.select"(%tensor_pred, %tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @compare
func @compare(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xi1>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.compare"(%tensor_lhs, %tensor_rhs)
      {comparison_direction = "EQ"}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // BOTH: "xla_lhlo.compare"(%{{.*}}, %{{.*}}, %{{.*}}) {comparison_direction = "EQ"}
  tensor_store %tensor_result, %result : memref<2x2xi1>
  return
}

// -----

// BOTH-LABEL: func @broadcast
func @broadcast(%operand: memref<5xf32>, %result: memref<10x5xf32>) {
  %tensor_operand = tensor_load %operand : memref<5xf32>
  %tensor_result = "xla_hlo.broadcast_in_dim"(%tensor_operand)
      {broadcast_dimensions = dense<1> : tensor<1xi64>}
        : (tensor<5xf32>) -> tensor<10x5xf32>
  // BOTH: "xla_lhlo.broadcast_in_dim"(%{{.*}}, %{{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<10x5xf32>
  return
}

// -----

func @external_func() -> tensor<3xi64>

// BOTH: #[[MAP:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>

// BOTH-LABEL: func @dyn_broadcast
func @dyn_broadcast(%operand: memref<?x?xf32>) {
  // BOTH-SAME: (%[[OPERAND:.*]]: memref<?x?xf32>)
  %tensor_operand = tensor_load %operand : memref<?x?xf32>
  %shape = call @external_func() : () -> tensor<3xi64>
  %tensor_result = "xla_hlo.dynamic_broadcast_in_dim"(%tensor_operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  // BOTH: %[[SHAPE:.*]] = call @external_func()
  // BOTH: %[[C0:.*]] = constant 0 : index
  // BOTH: %[[EL0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<3xi64>
  // BOTH: %[[IC0:.*]]  = index_cast %[[EL0]] : i64 to index
  // BOTH: %[[C1:.*]] = constant 1 : index
  // BOTH: %[[EL1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<3xi64>
  // BOTH: %[[IC1:.*]]  = index_cast %[[EL1]] : i64 to index
  // BOTH: %[[C2:.*]] = constant 2 : index
  // BOTH: %[[EL2:.*]] = extract_element %[[SHAPE]][%[[C2]]] : tensor<3xi64>
  // BOTH: %[[IC2:.*]]  = index_cast %[[EL2]] : i64 to index
  // BOTH: %[[RESULT:.*]] = alloc(%[[IC0]], %[[IC1]], %[[IC2]])

  // BOTH: %[[C0_:.*]] = constant 0 : index
  // BOTH: %[[C1_:.*]] = constant 1 : index

  // BOTH: %[[C1__:.*]] = constant 1 : index
  // BOTH: %[[EL1_:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C1__]]] : tensor<3xi64>
  // BOTH: %[[C0___:.*]] = constant 0 : index
  // BOTH: %[[OPERAND_DIM_0:.*]] = dim %[[OPERAND]], %[[C0___]] : memref<?x?xf32>
  // BOTH: %[[RESULT_DIM_1:.*]] = index_cast %[[EL1_]] : i64 to index
  // BOTH: %[[EXPAND_0:.*]] = cmpi "slt", %[[OPERAND_DIM_0]], %[[RESULT_DIM_1]]
  // BOTH: %[[STRIDE_0:.*]] = select %[[EXPAND_0]], %[[C0_]], %[[C1_]] : index

  // BOTH: %[[C2_:.*]] = constant 2 : index
  // BOTH: %[[EL2_:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C2_]]] : tensor<3xi64>
  // BOTH: %[[C1___:.*]] = constant 1 : index
  // BOTH: %[[OPERAND_DIM_1:.*]] = dim %[[OPERAND]], %[[C1___]] : memref<?x?xf32>
  // BOTH: %[[RESULT_DIM_2:.*]] = index_cast %[[EL2_]] : i64 to index
  // BOTH: %[[EXPAND_1:.*]] = cmpi "slt", %[[OPERAND_DIM_1]], %[[RESULT_DIM_2]]
  // BOTH: %[[STRIDE_1:.*]] = select %[[EXPAND_1]], %[[C0_]], %[[C1_]] : index

  // BOTH: %[[TRANSFORMED_MEMREF:.*]] = xla_lhlo.dynamic_memref_cast
  // BOTH-SAME: %[[OPERAND]](%[[RESULT_DIM_1]], %[[RESULT_DIM_2]])
  // BOTH-SAME: {{\[}}%[[STRIDE_0]], %[[STRIDE_1]]]
  // BOTH-SAME: : memref<?x?xf32> -> memref<?x?xf32, #map0>

  // BOTH: "xla_lhlo.broadcast_in_dim"(%[[TRANSFORMED_MEMREF]], %[[RESULT]]) {
  // BOTH-SAME:   broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  // BOTH-SAME: } : (memref<?x?xf32, #[[MAP]]>, memref<?x?x?xf32>) -> ()

  // Do not store the value back to avoid the tensor-store being rewritten to
  // a copy into the pre-allocated argument.
  return
}

// -----

// BOTH-LABEL: func @complex
func @complex(%real: memref<2x2xf32>,
              %imag: memref<2x2xf32>,
              %result: memref<2x2xcomplex<f32>>) {
  %tensor_real = tensor_load %real : memref<2x2xf32>
  %tensor_imag = tensor_load %imag : memref<2x2xf32>
  %tensor_result = "xla_hlo.complex"(%tensor_real, %tensor_imag)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  // BOTH: "xla_lhlo.complex"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xcomplex<f32>>
  return
}

// -----

// BOTH-LABEL: func @real
func @real(%operand: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xcomplex<f32>>
  %tensor_result = "xla_hlo.real"(%tensor_operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.real"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @imag
func @imag(%operand: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xcomplex<f32>>
  %tensor_result = "xla_hlo.imag"(%tensor_operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.imag"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @iota
func @iota(%result: memref<10xi32>) {
  %tensor_result = "xla_hlo.iota"()
      {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  // BOTH: "xla_lhlo.iota"(%{{.*}}) {iota_dimension = 0 : i64}
  tensor_store %tensor_result, %result : memref<10xi32>
  return
}

// -----

// BOTH-LABEL: func @abs
func @abs(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.abs"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.abs"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @ceil
func @ceil(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.ceil"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.ceil"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @convert
func @convert(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.convert"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.copy"(%{{.*}}, %{{.*}})
  // BOTH-NOT: tensor_store
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @cos
func @cos(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.cosine"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.cosine"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @neg
func @neg(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.negate"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.negate"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @rsqrt
func @rsqrt(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.rsqrt"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.rsqrt"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @sign
func @sign(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.sign"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.sign"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @sqrt
func @sqrt(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.sqrt"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.sqrt"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @tanh
func @tanh(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.tanh"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.tanh"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// BOTH-LABEL: func @remainder
func @remainder(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.remainder"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // BOTH: "xla_lhlo.remainder"(%{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// Dynamic shape binary element-wise operation.
// BOTH-LABEL: func @add_dyn
func @add_dyn(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) {
  %result = "xla_hlo.add"(%lhs, %rhs)
      : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // BOTH: %[[C0:.*]] = constant 0 : index
  // BOTH: %[[DIM0:.*]] = dim %arg0, %[[C0]] : memref<?x?xf32>
  // BOTH: %[[IC0:.*]] = index_cast %[[DIM0]] : index to i64
  // BOTH: %[[C1:.*]] = constant 1 : index
  // BOTH: %[[DIM1:.*]] = dim %arg0, %[[C1]] : memref<?x?xf32>
  // BOTH: %[[IC1:.*]] = index_cast %[[DIM1]] : index to i64
  // BOTH: %[[SHAPE:.*]] = tensor_from_elements(%[[IC0]], %[[IC1]]) : tensor<2xi64>
  // BOTH: %[[C0_:.*]] = constant 0 : index
  // BOTH: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0_]]] : tensor<2xi64>
  // BOTH: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // BOTH: %[[C1_:.*]] = constant 1 : index
  // BOTH: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1_]]] : tensor<2xi64>
  // BOTH: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // BOTH: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // BOTH: "xla_lhlo.add"(%arg0, %arg1, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

// Dynamic shape unary element-wise operation.
// BOTH-LABEL: func @tanh_dyn
func @tanh_dyn(%arg0: tensor<?x?xf32>) {
  %result = "xla_hlo.tanh"(%arg0)
      : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // BOTH: %[[C0:.*]] = constant 0 : index
  // BOTH: %[[DIM0:.*]] = dim %arg0, %[[C0]] : memref<?x?xf32>
  // BOTH: %[[IC0:.*]] = index_cast %[[DIM0]] : index to i64
  // BOTH: %[[C1:.*]] = constant 1 : index
  // BOTH: %[[DIM1:.*]] = dim %arg0, %[[C1]] : memref<?x?xf32>
  // BOTH: %[[IC1:.*]] = index_cast %[[DIM1]] : index to i64
  // BOTH: %[[SHAPE:.*]] = tensor_from_elements(%[[IC0]], %[[IC1]]) : tensor<2xi64>
  // BOTH: %[[C0_:.*]] = constant 0 : index
  // BOTH: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0_]]] : tensor<2xi64>
  // BOTH: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // BOTH: %[[C1_:.*]] = constant 1 : index
  // BOTH: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1_]]] : tensor<2xi64>
  // BOTH: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // BOTH: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // BOTH: "xla_lhlo.tanh"(%arg0, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

// BOTH-LABEL: func @dot
func @dot(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
//  PRE-SAME: (%[[ARG0:.*]]: [[TYPE:.*]], %[[RESULT:.*]]: [[TYPE]])
//  ESC-SAME: (%[[ARG0:.*]]: [[TYPE:.*]]) -> [[TYPE]]
// BOTH-NEXT: %[[ALLOC:.*]] = alloc
//      BOTH: "xla_lhlo.dot"(%[[ARG0]], %[[ARG0]], %[[ALLOC]]) : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
  %dot = "xla_hlo.dot"(%arg0, %arg0)
          : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// PRE: "xla_lhlo.copy"(%[[ALLOC]], %[[RESULT]])
// ESC: return %[[ALLOC]]
  return %dot : tensor<1024x1024xf32>
}

// -----

// BOTH-LABEL: func @conv
func @conv(%input: tensor<3x5x5x3xf32>, %filter : tensor<2x2x3x4xf32>) -> tensor<3x5x5x4xf32> {
  %c0 = constant 0 : index
  // BOTH: %[[OUT:.*]] = alloc() : memref<3x5x5x4xf32>
  // BOTH: "xla_lhlo.convolution"(%{{.+}}, %{{.+}}, %[[OUT]])
  // BOTH-SAME: padding = dense<[
  // BOTH-SAME:                  [0, 1], [0, 1]]> : tensor<2x2xi64>
  // BOTH-SAME: rhs_dilation = dense<[1, 2]>
  // BOTH-SAME: window_strides = dense<[2, 1]>
  %out = "xla_hlo.convolution"(%filter, %input) {
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
