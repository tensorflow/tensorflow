// RUN: tf-opt -hlo-legalize-to-lhlo %s -o - | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: func @attrs
func @attrs_copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exp"(%tensor_operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.exp"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @func_op_long
func @func_op_long(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: (%[[NEW_ARG0:.*]]: memref<4xf32>, %[[NEW_ARG1:.*]]: memref<4xf32>, %[[RESULT:.*]]: memref<4xf32>)
  // CHECK-NEXT: %[[MUL_RESULT:.*]] = alloc() {temp = true} : memref<4xf32>
  // CHECK-NEXT: %[[SUB_RESULT:.*]] = alloc() {temp = true} : memref<4xf32>
  // CHECK-NEXT: %[[MIN_RESULT:.*]] = alloc() {temp = true} : memref<4xf32>
  // CHECK-NEXT: %[[ADD_RESULT:.*]] = alloc() {temp = true} : memref<4xf32>
  // CHECK-NEXT: %[[MAX_RESULT:.*]] = alloc() {temp = true} : memref<4xf32>
  %1 = xla_hlo.maximum %arg0, %arg1 : tensor<4xf32>
  // CHECK-NEXT: "xla_lhlo.maximum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MAX_RESULT]])
  %2 = xla_hlo.add %arg0, %1 : tensor<4xf32>
  // CHECK-NEXT: "xla_lhlo.add"(%[[NEW_ARG0]], %[[MAX_RESULT]], %[[ADD_RESULT]])
  %3 = xla_hlo.minimum %arg0, %arg1 : tensor<4xf32>
  // CHECK-NEXT: "xla_lhlo.minimum"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[MIN_RESULT]])
  %4 = xla_hlo.subtract %arg1, %3 : tensor<4xf32>
  // CHECK-NEXT: "xla_lhlo.subtract"(%[[NEW_ARG1]], %[[MIN_RESULT]], %[[SUB_RESULT]])
  %5 = xla_hlo.multiply %2, %4 : tensor<4xf32>
  // CHECK-NEXT: "xla_lhlo.multiply"(%[[ADD_RESULT]], %[[SUB_RESULT]], %[[MUL_RESULT]])
  // CHECK-NEXT: dealloc %[[MAX_RESULT]] : memref<4xf32>
  // CHECK-NEXT: dealloc %[[ADD_RESULT]] : memref<4xf32>
  // CHECK-NEXT: dealloc %[[MIN_RESULT]] : memref<4xf32>
  // CHECK-NEXT: dealloc %[[SUB_RESULT]] : memref<4xf32>
  // CHECK-NEXT: "xla_lhlo.copy"(%[[MUL_RESULT]], %[[RESULT]]) : (memref<4xf32>, memref<4xf32>) -> ()
  // CHECK-NEXT: dealloc %[[MUL_RESULT]] : memref<4xf32>
  return %5 : tensor<4xf32>
  // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
}

// -----

// CHECK-LABEL: func @fusion
func @fusion(%multiplier: memref<2x2xf32>, %summand_1: memref<2x2xf32>,
             %summand_2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  // CHECK: (%{{.*}}: {{.*}}, {{.*}}: {{.*}}, {{.*}}: {{.*}}, %[[RESULT:.*]]: {{.*}})
  // CHECK-NEXT:  %[[MUL_RESULT:.*]] = alloc() {temp = true} : memref<2x2xf32>
  // CHECK-NEXT:  %[[ADD_RESULT:.*]] = alloc() {temp = true} : memref<2x2xf32>
  %tensor_summand_1 = tensor_load %summand_1 : memref<2x2xf32>
  %tensor_summand_2 = tensor_load %summand_2 : memref<2x2xf32>
  %sum = "xla_hlo.add"(%tensor_summand_1, %tensor_summand_2)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  %tensor_multiplier = tensor_load %multiplier : memref<2x2xf32>
  %tensor_result = "xla_hlo.multiply"(%sum, %tensor_multiplier)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.multiply"(%[[ADD_RESULT]], %{{.*}}, %[[MUL_RESULT]])
  // CHECK-NEXT: "xla_lhlo.copy"(%[[MUL_RESULT]], %[[RESULT]])
  tensor_store %tensor_result, %result : memref<2x2xf32>
  // CHECK-NEXT:  dealloc %[[ADD_RESULT]] : memref<2x2xf32>
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
  %tensor_result = "xla_hlo.exp"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.exp"(%{{.*}}, %{{.*}})
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

// CHECK-LABEL: func @dyn_broadcast
func @dyn_broadcast(%operand: memref<?x?xf32>) {
  %tensor_operand = tensor_load %operand : memref<?x?xf32>
  %shape = "compute.shape"() : () -> tensor<3xi64>
  %tensor_result = "xla_hlo.dynamic_broadcast_in_dim"(%tensor_operand, %shape)
      {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  // CHECK: %[[SHAPE:.*]] = "compute.shape"()
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
  // CHECK-NEXT: "xla_lhlo.broadcast_in_dim"(%{{.*}}, %[[RESULT]]) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
  // Do not store the value back to avoid the tensor-store being rewritten to
  // a copy into the pre-allocated argument.
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
  // CHECK: xla_lhlo.terminator
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @cos
func @cos(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.cos"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.cos"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// -----

// CHECK-LABEL: func @neg
func @neg(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.neg"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: "xla_lhlo.neg"(%{{.*}}, %{{.*}})
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
  // CHECK: %[[SHAPE:.*]] = "xla_hlo.scalars_to_dimension_tensor"(%[[IC0]], %[[IC1]]) : (i64, i64) -> tensor<2xi64>
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
  // CHECK: %[[SHAPE:.*]] = "xla_hlo.scalars_to_dimension_tensor"(%[[IC0]], %[[IC1]]) : (i64, i64) -> tensor<2xi64>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[EE0:.*]] = extract_element %[[SHAPE]][%[[C0]]] : tensor<2xi64>
  // CHECK: %[[ICS0:.*]] = index_cast %[[EE0]] : i64 to index
  // CHECK: %[[EE1:.*]] = extract_element %[[SHAPE]][%[[C1]]] : tensor<2xi64>
  // CHECK: %[[ICS1:.*]] = index_cast %[[EE1]] : i64 to index
  // CHECK: %[[RESULT:.*]] = alloc(%[[ICS0]], %[[ICS1]])
  // CHECK: "xla_lhlo.tanh"(%arg0, %[[RESULT]]) : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}
