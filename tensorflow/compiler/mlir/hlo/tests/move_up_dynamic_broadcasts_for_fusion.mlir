// RUN: mlir-hlo-opt --split-input-file --allow-unregistered-dialect --mhlo-move-up-dynamic-broadcasts-for-fusion --canonicalize --cse %s | FileCheck %s

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_unary
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x32xi16>)
func @shape_of_unary(%arg : tensor<?x32xi16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x32xi16> -> tensor<2xindex>
  // CHECK: %[[CASTED:.*]] = tensor.cast %[[SHAPE]] : tensor<2xindex> to tensor<?xindex>
  // CHECK: "use"(%[[CASTED]])
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%1) : (tensor<?xindex>) -> ()
  return
}

// -----

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_nary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>)
func @shape_of_nary(%arg0 : tensor<?x32xf16>, %arg1 : tensor<?x32xf16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG0]] : tensor<?x32xf16> -> tensor<2xindex>
  // CHECK: %[[CASTED:.*]] = tensor.cast %[[SHAPE]] : tensor<2xindex> to tensor<?xindex>
  // CHECK: "use"(%[[CASTED]])
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf16>
  %1 = mhlo.subtract %0, %arg1 : tensor<?x32xf16>
  %2 = shape.shape_of %1 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%2) : (tensor<?xindex>) -> ()
  return
}

// -----

// Broadcasts can be moved up over unary shape-preserving operations.
// CHECK-LABEL: @bcast_unary
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x32xi16>, %[[OUT_DIMS:.*]]: tensor<3xindex>)
func @bcast_unary(%arg : tensor<?x32xi16>, %out_dims : tensor<3xindex>)
    -> tensor<?x?x32xf16> {
  // CHECK:      %[[BCASTED_OPERAND:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG]], %[[OUT_DIMS]])
  // CHECK-SAME: broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x32xi16>, tensor<3xindex>) -> tensor<?x?x32xi16>
  // CHECK:      "mhlo.convert"(%[[BCASTED_OPERAND]]) : (tensor<?x?x32xi16>) -> tensor<?x?x32xf16>
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %out_dims) {
      broadcast_dimensions = dense<[0, 1]> : tensor<2xi64> } :
      (tensor<?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
  return %1 : tensor<?x?x32xf16>
}

// -----

// Broadcasts can be moved up over n-ary shape-preserving operations.
// CHECK-LABEL: @bcast_nary
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xf32>, %[[ARG1:.*]]: tensor<?x32xf32>, %[[OUT_DIMS:.*]]: tensor<3xindex>)
func @bcast_nary(%arg0 : tensor<?x32xf32>, %arg1 : tensor<?x32xf32>,
    %out_dims : tensor<3xindex>) -> tensor<?x?x32xf32> {
  // CHECK-NOT: subtract
  // CHECK:     %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[OUT_DIMS]])
  // CHECK:     %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[OUT_DIMS]])
  // CHECK:     %{{.*}} = mhlo.subtract %[[BCASTED_ARG0]], %[[BCASTED_ARG1]] : tensor<?x?x32xf32>
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf32>
  %1 = "mhlo.dynamic_broadcast_in_dim"(%0, %out_dims) {
      broadcast_dimensions = dense<[0, 1]> : tensor<2xi64> } :
      (tensor<?x32xf32>, tensor<3xindex>) -> tensor<?x?x32xf32>
  return %1 : tensor<?x?x32xf32>
}

// -----

// Exemplary IR as it appears in the lowering with `tf.Sub` and `tf.Cast`.
// CHECK-LABEL: @cast_sub
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x32xi16>, %[[ARG1:.*]]: tensor<?x?x32xf16>) -> tensor<?x?x32xf16>
func @cast_sub(%arg0: tensor<?x32xi16>, %arg1: tensor<?x?x32xf16>)
    -> tensor<?x?x32xf16> {
  // CHECK-NOT: convert
  // CHECK:     %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %{{.*}})
  // CHECK:     %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %{{.*}})
  // CHECK:     %[[CONVERTED_BCASTED_ARG0:.*]] = "mhlo.convert"(%[[BCASTED_ARG0]]) : (tensor<?x?x32xi16>) -> tensor<?x?x32xf16>
  // CHECK:     %{{.*}} = mhlo.subtract %[[BCASTED_ARG1]], %[[CONVERTED_BCASTED_ARG0]] : tensor<?x?x32xf16>
  %0 = "mhlo.convert"(%arg0) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = shape.shape_of %arg1 : tensor<?x?x32xf16> -> tensor<?xindex>
  %2 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
  %3 = shape.cstr_broadcastable %1, %2 : tensor<?xindex>, tensor<?xindex>
  %4 = shape.assuming %3 -> (tensor<?x?x32xf16>) {
    %5 = shape.shape_of %arg1 : tensor<?x?x32xf16> -> tensor<?xindex>
    %6 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
    %7 = shape.broadcast %5, %6 : tensor<?xindex>, tensor<?xindex>
        -> tensor<?xindex>
    %8 = tensor.cast %7 : tensor<?xindex> to tensor<3xindex>
    %9 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %8) {
        broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} :
        (tensor<?x?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
    %10 = "mhlo.dynamic_broadcast_in_dim"(%0, %8) {
        broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} :
        (tensor<?x32xf16>, tensor<3xindex>) -> tensor<?x?x32xf16>
    %11 = mhlo.subtract %9, %10 : tensor<?x?x32xf16>
    shape.assuming_yield %11 : tensor<?x?x32xf16>
  }
  return %4 : tensor<?x?x32xf16>
}
