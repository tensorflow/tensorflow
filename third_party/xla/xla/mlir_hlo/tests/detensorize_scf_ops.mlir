// RUN: mlir-hlo-opt --split-input-file %s --detensorize-scf-ops | \
// RUN: FileCheck %s --dump-input=always

func.func @while_arg(%arg0: tensor<f32>) -> tensor<2xf32> {
  %false = arith.constant false
  %r = scf.while (%arg1 = %arg0) : (tensor<f32>) -> (tensor<2xf32>) {
    %scalar = tensor.extract %arg1[] : tensor<f32>
    %splat = tensor.splat %scalar : tensor<2xf32>
    scf.condition (%false) %splat : tensor<2xf32>
  } do {
  ^bb0(%arg2: tensor<2xf32>):
    %first = tensor.extract_slice %arg2[0] [1] [1] : tensor<2xf32> to tensor<f32>
    scf.yield %first : tensor<f32>
  }
  func.return %r : tensor<2xf32>
}

// CHECK-LABEL: @while_arg
// CHECK-SAME:    (%[[ARG:.*]]:
// CHECK:       %[[FALSE:.*]] = arith.constant false
// CHECK:       %[[SCALAR_ARG:.*]] = tensor.extract %[[ARG]]
// CHECK:       %[[RESULT:.*]] = scf.while (%[[WHILE_ARG:.*]] = %[[SCALAR_ARG]])
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[WHILE_ARG]]
// CHECK:         scf.condition(%[[FALSE]]) %[[SPLAT]]
// CHECK:       } do {
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice
// CHECK:         %[[SCALAR:.*]] = tensor.extract %[[SLICE]][]
// CHECK:         scf.yield %[[SCALAR]]
// CHECK:       }
// CHECK:       return %[[RESULT]]

// -----

func.func @while_return(%arg0: tensor<2xf32>) -> tensor<f32> {
  %false = arith.constant 0 : i1
  %r = scf.while (%arg1 = %arg0) : (tensor<2xf32>) -> (tensor<f32>) {
    %extract = tensor.extract_slice %arg1[0] [1] [1] : tensor<2xf32> to tensor<f32>
    scf.condition (%false) %extract : tensor<f32>
  } do {
  ^bb0(%arg2: tensor<f32>):
    %scalar = tensor.extract %arg2[] : tensor<f32>
    %splat = tensor.splat %scalar : tensor<2xf32>
    scf.yield %splat : tensor<2xf32>
  }
  func.return %r : tensor<f32>
}

// CHECK-LABEL: @while_return
// CHECK-SAME:    (%[[ARG:.*]]:
// CHECK:       %[[FALSE:.*]] = arith.constant false
// CHECK:       %[[SCALAR_RESULT:.*]] = scf.while (%[[WHILE_ARG:.*]] = %[[ARG]])
// CHECK:         %[[SLICE:.*]] = tensor.extract_slice
// CHECK:         %[[SCALAR:.*]] = tensor.extract %[[SLICE]][]
// CHECK:         scf.condition(%[[FALSE]]) %[[SCALAR]]
// CHECK:       } do {
// CHECK:       ^bb0(%[[BODY_ARG:.*]]: f32):
// CHECK:         %[[SPLAT:.*]] = tensor.splat %[[BODY_ARG]]
// CHECK:         scf.yield %[[SPLAT]]
// CHECK:       }
// CHECK:       %[[RESULT:.*]] = tensor.from_elements %[[SCALAR_RESULT]]
// CHECK:       return %[[RESULT]]

// -----

func.func @if_return(%cond: i1) -> tensor<f32> {
  %c0 = arith.constant dense<0.0> : tensor<f32>
  %c1 = arith.constant dense<1.0> : tensor<f32>
  %r = scf.if %cond -> tensor<f32> {
    scf.yield %c0 : tensor<f32>
  } else {
    scf.yield %c1 : tensor<f32>
  }
  return %r : tensor<f32>
}

// CHECK-LABEL: @if_return
// CHECK-SAME:    (%[[COND:.*]]:
// CHECK:       %[[C0:.*]] = arith.constant 0
// CHECK:       %[[C1:.*]] = arith.constant 1
// CHECK:       %[[RESULT_SCALAR:.*]] = scf.if %[[COND]] -> (f32) {
// CHECK:         scf.yield %[[C0]]
// CHECK:       } else {
// CHECK:         scf.yield %[[C1]]
// CHECK:       }
// CHECK:       %[[RESULT:.*]] = tensor.from_elements %[[RESULT_SCALAR]]
// CHECK:       return %[[RESULT]]

// -----

func.func @for(%arg: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant dense<0.0> : tensor<f32>
  %r = scf.for %i = %c0 to %c1 step %c1 iter_args(%farg = %arg) -> tensor<f32> {
    scf.yield %c2 : tensor<f32>
  }
  return %r : tensor<f32>
}

// CHECK-LABEL: @for
// CHECK-SAME:    (%[[ARG:.*]]:
// CHECK:       %[[CST:.*]] = arith.constant 0.0
// CHECK:       %[[C0:.*]] = arith.constant 0
// CHECK:       %[[C1:.*]] = arith.constant 1
// CHECK:       %[[ARG_SCALAR:.*]] = tensor.extract %[[ARG]]
// CHECK:       %[[RESULT_SCALAR:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[ARG_SCALAR]])
// CHECK:         scf.yield %[[CST]]
// CHECK:       }
// CHECK:       %[[RESULT:.*]] = tensor.from_elements %[[RESULT_SCALAR]]
// CHECK:       return %[[RESULT]]