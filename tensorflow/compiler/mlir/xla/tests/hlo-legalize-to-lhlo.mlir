// RUN: tf-opt -hlo-legalize-to-lhlo -lhlo-deallocs-insertion -lhlo-redundant-copies-removal %s -o - | FileCheck %s --dump-input=always

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

// CHECK-LABEL: func @buffer_assignment_on_func_with_branches_using_live_analysis
func @buffer_assignment_on_func_with_branches_using_live_analysis(%arg0: tensor<10xf32>, %arg1: tensor<11x11xf32>) -> tensor<10xf32> {
  br ^bb1(%arg0, %arg1: tensor<10xf32>, tensor<11x11xf32>)
^bb1(%arg2: tensor<10xf32>, %arg3: tensor<11x11xf32>):
  %0 = "xla_hlo.exp"(%arg2) : (tensor<10xf32>) -> tensor<10xf32>
  br ^bb2(%0: tensor<10xf32>)
^bb2(%arg4: tensor<10xf32>):
  %2 = "xla_hlo.exp"(%arg4) : (tensor<10xf32>) -> tensor<10xf32>
  return %2 : tensor<10xf32>
}
//      CHECK: (%[[ARG0:.*]]: memref<10xf32>, %[[ARG1:.*]]: memref<11x11xf32>, %[[RESULT:.*]]: memref<10xf32>)
// CHECK-NEXT: %[[BB1_ARG0:.*]] = alloc() {temp = true} : memref<10xf32>
// CHECK-NEXT: "xla_lhlo.copy"(%[[ARG0]], %[[BB1_ARG0]]) :
// CHECK-NEXT: %[[BB1_ARG1:.*]] = alloc() {temp = true} : memref<11x11xf32>
// CHECK-NEXT: "xla_lhlo.copy"(%[[ARG1]], %[[BB1_ARG1]])
// CHECK-NEXT: br ^bb1(%[[BB1_ARG0]], %[[BB1_ARG1]] : memref<10xf32>, memref<11x11xf32>)
// CHECK-NEXT: ^bb1(%[[ARG2:.*]]: memref<10xf32>, %[[ARG3:.*]]: memref<11x11xf32>): // pred: ^bb0
// CHECK-NEXT: %[[EXP_RESULT:.*]] = alloc() {temp = true} : memref<10xf32>
// CHECK-NEXT: dealloc %[[ARG3]] : memref<11x11xf32>
// CHECK-NEXT: "xla_lhlo.exp"(%[[ARG2]], %[[EXP_RESULT]]) : (memref<10xf32>, memref<10xf32>) -> ()
// CHECK-NEXT: dealloc %[[ARG2]] : memref<10xf32>
// CHECK-NEXT: %[[BB2_ARG0:.*]] = alloc() {temp = true} : memref<10xf32>
// CHECK-NEXT: "xla_lhlo.copy"(%[[EXP_RESULT]], %[[BB2_ARG0]]) : (memref<10xf32>, memref<10xf32>) -> ()
// CHECK-NEXT: dealloc %[[EXP_RESULT]] : memref<10xf32>
// CHECK-NEXT: br ^bb2(%[[BB2_ARG0]] : memref<10xf32>)
// CHECK-NEXT: ^bb2(%[[ARG4:.*]]: memref<10xf32>): // pred: ^bb1
// CHECK-NEXT: %[[EXP2_RESULT:.*]] = alloc() {temp = true} : memref<10xf32>
// CHECK-NEXT: "xla_lhlo.exp"(%[[ARG4]], %[[EXP2_RESULT]]) : (memref<10xf32>, memref<10xf32>) -> ()
// CHECK-NEXT: dealloc %[[ARG4]] : memref<10xf32>
// CHECK-NEXT: "xla_lhlo.copy"(%[[EXP2_RESULT]], %[[RESULT]]) : (memref<10xf32>, memref<10xf32>) -> ()
// CHECK-NEXT: dealloc %[[EXP2_RESULT]] : memref<10xf32>
// CHECK-NEXT: "xla_lhlo.terminator"() : () -> ()


// CHECK-LABEL: func @remove_lhlo_copy_op_created_from_tensor_store
func @remove_lhlo_copy_op_created_from_tensor_store(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: memref<f32>) {
  %0 = "xla_hlo.max"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  tensor_store %0, %arg2 : memref<f32>
  return
}
// CHECK: (%[[NEW_ARG0:.*]]: memref<f32>, %[[NEW_ARG1:.*]]: memref<f32>, %[[RESULT:.*]]: memref<f32>)
// CHECK-NOT: %[[ALLOC_OPERAND:.*]] = alloc() {temp = true} : memref<f32>
// CHECK: "xla_lhlo.max"(%[[NEW_ARG0]], %[[NEW_ARG1]], %[[RESULT]]) : (memref<f32>, memref<f32>, memref<f32>) -> ()
// CHECK-NOT: "xla_lhlo.copy"(%[[ALLOC_OPERAND]], %[[RESULT]]) : (memref<f32>, memref<f32>) -> ()
// CHECK-NOT: dealloc %[[ALLOC_OPERAND]] : memref<f32>
// CHECK: "xla_lhlo.terminator"() : () -> ()

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
  // CHECK: xla_lhlo.terminator
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
