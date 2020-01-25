// RUN: tf-opt -hlo-legalize-to-lhlo %s -o - | FileCheck %s --dump-input=always

// CHECK-LABEL: func @attrs
func @attrs_copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exp"(%tensor_operand)
      {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.exp"(%{{.*}}, %{{.*}}) {some_attr_1 = "exp.1", some_attr_2 = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @fusion
func @fusion(%multiplier: memref<2x2xf32>, %summand_1: memref<2x2xf32>,
             %summand_2: memref<2x2xf32>, %result: memref<2x2xf32>) {
  // CHECK-NEXT:  %[[ADD_RESULT:.*]] = alloc() {temp = true} : memref<2x2xf32>
  %tensor_summand_1 = tensor_load %summand_1 : memref<2x2xf32>
  %tensor_summand_2 = tensor_load %summand_2 : memref<2x2xf32>
  %sum = "xla_hlo.add"(%tensor_summand_1, %tensor_summand_2)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.add"(%{{.*}}, %{{.*}}, %[[ADD_RESULT]])
  %tensor_multiplier = tensor_load %multiplier : memref<2x2xf32>
  %tensor_result = "xla_hlo.mul"(%sum, %tensor_multiplier)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.mul"(%[[ADD_RESULT]], %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  // CHECK-NEXT:  dealloc %[[ADD_RESULT]] : memref<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()
  "xla_lhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @copy
func @copy(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.copy"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.copy"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @exp
func @exp(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.exp"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.exp"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @select
func @select(%pred: memref<2x2xi1>, %lhs: memref<2x2xf32>,
             %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_pred = tensor_load %pred : memref<2x2xi1>
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.select"(%tensor_pred, %tensor_lhs, %tensor_rhs)
      : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.select"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @compare
func @compare(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xi1>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.compare"(%tensor_lhs, %tensor_rhs)
      {comparison_direction = "EQ"}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  // CHECK-NEXT: "xla_lhlo.compare"(%{{.*}}, %{{.*}}, %{{.*}}) {comparison_direction = "EQ"}
  tensor_store %tensor_result, %result : memref<2x2xi1>
  return
}

// CHECK-LABEL: func @broadcast
func @broadcast(%operand: memref<5xf32>, %result: memref<10x5xf32>) {
  %tensor_operand = tensor_load %operand : memref<5xf32>
  %tensor_result = "xla_hlo.broadcast_in_dim"(%tensor_operand)
      {broadcast_dimensions = dense<1> : tensor<1xi64>}
        : (tensor<5xf32>) -> tensor<10x5xf32>
  // CHECK-NEXT: "xla_lhlo.broadcast_in_dim"(%{{.*}}, %{{.*}}) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  tensor_store %tensor_result, %result : memref<10x5xf32>
  return
}

// CHECK-LABEL: func @iota
func @iota(%result: memref<10xi32>) {
  %tensor_result = "xla_hlo.iota"()
      {iota_dimension = 0 : i64} : () -> tensor<10xi32>
  // CHECK-NEXT: "xla_lhlo.iota"(%{{.*}}) {iota_dimension = 0 : i64}
  tensor_store %tensor_result, %result : memref<10xi32>
  return
}

// CHECK-LABEL: func @abs
func @abs(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.abs"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.abs"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @ceil
func @ceil(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.ceil"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.ceil"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @convert
func @convert(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.convert"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: return
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @cos
func @cos(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.cos"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.cos"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @neg
func @neg(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.neg"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.neg"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @sign
func @sign(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.sign"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.sign"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @tanh
func @tanh(%operand: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_operand = tensor_load %operand : memref<2x2xf32>
  %tensor_result = "xla_hlo.tanh"(%tensor_operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.tanh"(%{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}

// CHECK-LABEL: func @remainder
func @remainder(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  %tensor_lhs = tensor_load %lhs : memref<2x2xf32>
  %tensor_rhs = tensor_load %rhs : memref<2x2xf32>
  %tensor_result = "xla_hlo.remainder"(%tensor_lhs, %tensor_rhs)
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-NEXT: "xla_lhlo.remainder"(%{{.*}}, %{{.*}}, %{{.*}})
  tensor_store %tensor_result, %result : memref<2x2xf32>
  return
}
