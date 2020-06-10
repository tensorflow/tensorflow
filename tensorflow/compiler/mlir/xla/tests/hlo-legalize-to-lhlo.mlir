// RUN: xla-opt -hlo-legalize-to-lhlo -buffer-placement -split-input-file %s -o - | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: func @attrs
func @attrs_copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exponential"(%tensor_operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.exponential"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

func @return_func(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}
//      CHECK: (%[[ARG0:.*]]: [[TYPE:.*]], %[[RESULT:.*]]: [[TYPE]])
// CHECK-NEXT: "xla_lhlo.copy"(%[[ARG0]], %[[RESULT]]) : ([[TYPE]], [[TYPE]]) -> ()
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @func_op_long
func @func_op_long(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %1 = xla_hlo.maximum %arg0, %arg1 : tensor<4xf32>
  %2 = xla_hlo.add %arg0, %1 : tensor<4xf32>
  %3 = xla_hlo.minimum %arg0, %arg1 : tensor<4xf32>
  %4 = xla_hlo.subtract %arg1, %3 : tensor<4xf32>
  %5 = xla_hlo.multiply %2, %4 : tensor<4xf32>
  return %5 : tensor<4xf32>
}
//      CHECK: (%[[NEW_ARG0:.*]]: memref<4xf32>, %[[NEW_ARG1:.*]]: memref<4xf32>, %[[RESULT:.*]]: memref<4xf32>)
// CHECK-NEXT: %[[MAX_RESULT:.*]] = alloc() : memref<4xf32>
// CHECK-NEXT: "xla_lhlo.maximum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MAX_RESULT]])
// CHECK-NEXT: %[[ADD_RESULT:.*]] = alloc() : memref<4xf32>
// CHECK-NEXT: "xla_lhlo.add"(%[[NEW_ARG0]], %[[MAX_RESULT]], %[[ADD_RESULT]])
// CHECK-NEXT: dealloc %[[MAX_RESULT]] : memref<4xf32>
// CHECK-NEXT: %[[MIN_RESULT:.*]] = alloc() : memref<4xf32>
// CHECK-NEXT: "xla_lhlo.minimum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MIN_RESULT]])
// CHECK-NEXT: %[[SUB_RESULT:.*]] = alloc() : memref<4xf32>
// CHECK-NEXT: "xla_lhlo.subtract"(%[[NEW_ARG1]], %[[MIN_RESULT]], %[[SUB_RESULT]])
// CHECK-NEXT: dealloc %[[MIN_RESULT]] : memref<4xf32>
// CHECK-NEXT: %[[MUL_RESULT:.*]] = alloc() : memref<4xf32>
// CHECK-NEXT: "xla_lhlo.multiply"(%[[ADD_RESULT]], %[[SUB_RESULT]], %[[MUL_RESULT]])
// CHECK-NEXT: dealloc %[[SUB_RESULT]] : memref<4xf32>
// CHECK-NEXT: dealloc %[[ADD_RESULT]] : memref<4xf32>
// CHECK-NEXT: "xla_lhlo.copy"(%[[MUL_RESULT]], %[[RESULT]]) : (memref<4xf32>, memref<4xf32>) -> ()
// CHECK-NEXT: dealloc %[[MUL_RESULT]] : memref<4xf32>
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @fusion
func @fusion(%multiplier: memref<2x2xf32>, %summand_1: memref<2x2xf32>,
             %summand_2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  // CHECK: (%{{.*}}: {{.*}}, {{.*}}: {{.*}}, {{.*}}: {{.*}}, %[[RESULT:.*]]: {{.*}})
  // CHECK-NEXT:  %[[ADD_RESULT:.*]] = alloc() : memref<2x2xf32>
  %tensor_summand_1 = tensor_load %summand_1 : memref<2x2xf32>
  %tensor_summand_2 = tensor_load %summand_2 : memref<2x2xf32>
  %sum = "xla_hlo.add"(%tensor_summand_1, %tensor_summand_2)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  // CHECK-NEXT:  %[[MUL_RESULT:.*]] = alloc() : memref<2x2xf32>
  %tensor_multiplier = tensor_load %multiplier : memref<2x2xf32>
  %tensor_result = "xla_hlo.multiply"(%sum, %tensor_multiplier)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.multiply"(%[[ADD_RESULT]], %{{.*}}, %[[MUL_RESULT]])
  // CHECK-NEXT:  dealloc %[[ADD_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.copy"(%[[MUL_RESULT]], %[[RESULT]])
  tensor_store %tensor_result, %result : memref<2x2xf32>
  // CHECK-NEXT:  dealloc %[[MUL_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
  "xla_lhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @copy
func @copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.copy"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.copy"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @exp
func @exp(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exponential"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.exponential"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @log
func @log(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.log"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.log"(%{{.*}}, %{{.*}})
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
  %tensor_result = "xla_hlo.select"(%tensor_pred, %tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @compare
func @compare(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xi1>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.compare"(%tensor_lhs, %tensor_rhs)
      {comparison_direction = "EQ"}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // CHECK: "xla_lhlo.compare"(%{{.*}}, %{{.*}}, %{{.*}}) {comparison_direction = "EQ"}
  tensor_store %tensor_result, %result : memref<2x2xi1>
  return
}

// -----

// CHECK-LABEL: func @broadcast
func @broadcast(%operand: memref<5xf32>, %result: memref<10x5xf32>) {
  %tensor_operand = tensor_load %operand : memref<5xf32>
  %tensor_result = "xla_hlo.broadcast_in_dim"(%tensor_operand)
      {broadcast_dimensions = dense<1> : tensor<1xi64>}
        : (tensor<5xf32>) -> tensor<10x5xf32>
  // CHECK: "xla_lhlo.broadcast_in_dim"(%{{.*}}, %{{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<10x5xf32>
  return
}

// -----

func @external_func() -> tensor<3xi64>

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0 + d1 * s1)>

// CHECK-LABEL: func @dyn_broadcast
func @dyn_broadcast(%operand: memref<?x?xf32>) {
  // CHECK-SAME: (%[[OPERAND:.*]]: memref<?x?xf32>)
  %tensor_operand = tensor_load %operand : memref<?x?xf32>
  %shape = call @external_func() : () -> tensor<3xi64>
  %tensor_result = "xla_hlo.dynamic_broadcast_in_dim"(%tensor_operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  // CHECK: %[[SHAPE:.*]] = call @external_func()
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[EL0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<3xi64>
  // CHECK: %[[IC0:.*]]  = index_cast %[[EL0]] : i64 to index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[EL1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<3xi64>
  // CHECK: %[[IC1:.*]]  = index_cast %[[EL1]] : i64 to index
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %[[EL2:.*]] = extract_element %[[SHAPE]][%[[C2]]] : tensor<3xi64>
  // CHECK: %[[IC2:.*]]  = index_cast %[[EL2]] : i64 to index
  // CHECK: %[[RESULT:.*]] = alloc(%[[IC0]], %[[IC1]], %[[IC2]])

  // CHECK: %[[C0_:.*]] = constant 0 : index
  // CHECK: %[[C1_:.*]] = constant 1 : index

  // CHECK: %[[C1__:.*]] = constant 1 : index
  // CHECK: %[[EL1_:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C1__]]] : tensor<3xi64>
  // CHECK: %[[OPERAND_DIM_0:.*]] = dim %[[OPERAND]], 0 : memref<?x?xf32>
  // CHECK: %[[RESULT_DIM_1:.*]] = index_cast %[[EL1_]] : i64 to index
  // CHECK: %[[EXPAND_0:.*]] = cmpi "slt", %[[OPERAND_DIM_0]], %[[RESULT_DIM_1]]
  // CHECK: %[[STRIDE_0:.*]] = select %[[EXPAND_0]], %[[C0_]], %[[C1_]] : index

  // CHECK: %[[C2_:.*]] = constant 2 : index
  // CHECK: %[[EL2_:.*]] = extract_element %[[SHAPE]]{{\[}}%[[C2_]]] : tensor<3xi64>
  // CHECK: %[[OPERAND_DIM_1:.*]] = dim %[[OPERAND]], 1 : memref<?x?xf32>
  // CHECK: %[[RESULT_DIM_2:.*]] = index_cast %[[EL2_]] : i64 to index
  // CHECK: %[[EXPAND_1:.*]] = cmpi "slt", %[[OPERAND_DIM_1]], %[[RESULT_DIM_2]]
  // CHECK: %[[STRIDE_1:.*]] = select %[[EXPAND_1]], %[[C0_]], %[[C1_]] : index

  // CHECK: %[[TRANSFORMED_MEMREF:.*]] = xla_lhlo.dynamic_memref_cast
  // CHECK-SAME: %[[OPERAND]](%[[RESULT_DIM_1]], %[[RESULT_DIM_2]])
  // CHECK-SAME: {{\[}}%[[STRIDE_0]], %[[STRIDE_1]]]
  // CHECK-SAME: : memref<?x?xf32> -> memref<?x?xf32, #map0>

  // CHECK: "xla_lhlo.broadcast_in_dim"(%[[TRANSFORMED_MEMREF]], %[[RESULT]]) {
  // CHECK-SAME:   broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  // CHECK-SAME: } : (memref<?x?xf32, #[[MAP]]>, memref<?x?x?xf32>) -> ()

  // Do not store the value back to avoid the tensor-store being rewritten to
  // a copy into the pre-allocated argument.
  return
}

// -----

// CHECK-LABEL: func @complex
func @complex(%real: memref<2x2xf32>,
              %imag: memref<2x2xf32>,
              %result: memref<2x2xcomplex<f32>>) {
  %tensor_real = tensor_load %real : memref<2x2xf32>
  %tensor_imag = tensor_load %imag : memref<2x2xf32>
  %tensor_result = "xla_hlo.complex"(%tensor_real, %tensor_imag)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  // CHECK: "xla_lhlo.complex"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xcomplex<f32>>
  return
}

// -----

// CHECK-LABEL: func @real
func @real(%operand: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xcomplex<f32>>
  %tensor_result = "xla_hlo.real"(%tensor_operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.real"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @imag
func @imag(%operand: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xcomplex<f32>>
  %tensor_result = "xla_hlo.imag"(%tensor_operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.imag"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @iota
func @iota(%result: memref<10xi32>) {
  %tensor_result = "xla_hlo.iota"()
      {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  // CHECK: "xla_lhlo.iota"(%{{.*}}) {iota_dimension = 0 : i64}
  tensor_store %tensor_result, %result : memref<10xi32>
  return
}

// -----

// CHECK-LABEL: func @abs
func @abs(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.abs"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.abs"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @ceil
func @ceil(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.ceil"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.ceil"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @convert
func @convert(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.convert"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.copy"(%{{.*}}, %{{.*}})
  // CHECK-NOT: tensor_store
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @cos
func @cos(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.cosine"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.cosine"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @neg
func @neg(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.negate"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.negate"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @rsqrt
func @rsqrt(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.rsqrt"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.rsqrt"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @sign
func @sign(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.sign"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.sign"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @sqrt
func @sqrt(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.sqrt"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.sqrt"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @tanh
func @tanh(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.tanh"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.tanh"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @remainder
func @remainder(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.remainder"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.remainder"(%{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// Dynamic shape binary element-wise operation.
// CHECK-LABEL: func @add_dyn
func @add_dyn(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) {
  %result = "xla_hlo.add"(%lhs, %rhs)
      : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[DIM0:.*]] = dim %arg0, 0 : memref<?x?xf32>
  // CHECK: %[[IC0:.*]] = index_cast %[[DIM0]] : index to i64
  // CHECK: %[[DIM1:.*]] = dim %arg0, 1 : memref<?x?xf32>
  // CHECK: %[[IC1:.*]] = index_cast %[[DIM1]] : index to i64
  // CHECK: %[[SHAPE:.*]] = tensor_from_elements(%[[IC0]], %[[IC1]]) : tensor<2xi64>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<2xi64>
  // CHECK: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // CHECK: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<2xi64>
  // CHECK: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // CHECK: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // CHECK: "xla_lhlo.add"(%arg0, %arg1, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

// Dynamic shape unary element-wise operation.
// CHECK-LABEL: func @tanh_dyn
func @tanh_dyn(%arg0: tensor<?x?xf32>) {
  %result = "xla_hlo.tanh"(%arg0)
      : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: %[[DIM0:.*]] = dim %arg0, 0 : memref<?x?xf32>
  // CHECK: %[[IC0:.*]] = index_cast %[[DIM0]] : index to i64
  // CHECK: %[[DIM1:.*]] = dim %arg0, 1 : memref<?x?xf32>
  // CHECK: %[[IC1:.*]] = index_cast %[[DIM1]] : index to i64
  // CHECK: %[[SHAPE:.*]] = tensor_from_elements(%[[IC0]], %[[IC1]]) : tensor<2xi64>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<2xi64>
  // CHECK: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // CHECK: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<2xi64>
  // CHECK: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // CHECK: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // CHECK: "xla_lhlo.tanh"(%arg0, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @dot
func @dot(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
// CHECK-SAME: (%[[ARG0:.*]]: [[TYPE:.*]],
// CHECK-SAME:  %[[RESULT:.*]]: [[TYPE]])
// CHECK: "xla_lhlo.dot"(%[[ARG0]], %[[ARG0]], %{{.*}}) : ([[TYPE]], [[TYPE]], [[TYPE]]) -> ()
  %dot = "xla_hlo.dot"(%arg0, %arg0)
          : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %dot : tensor<1024x1024xf32>
}

// -----

// CHECK-LABEL: func @conv
func @conv(%input: tensor<3x5x5x3xf32>, %filter : tensor<2x2x3x4xf32>) -> tensor<3x5x5x4xf32> {
  %c0 = constant 0 : index
  // CHECK: %[[OUT:.*]] = alloc() : memref<3x5x5x4xf32>
  // CHECK: "xla_lhlo.convolution"(%{{.+}}, %{{.+}}, %[[OUT]])
  // CHECK-SAME: padding = dense<[
  // CHECK-SAME:                  [0, 1], [0, 1]]> : tensor<2x2xi64>
  // CHECK-SAME: rhs_dilation = dense<[1, 2]>
  // CHECK-SAME: window_strides = dense<[2, 1]>
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
