// RUN: mlir-hlo-opt --split-input-file --allow-unregistered-dialect --mhlo-move-up-dynamic-broadcasts-for-fusion --canonicalize --cse %s | FileCheck %s

// Shape computations shall be reified.
// CHECK-LABEL: @shape_of_unary
// CHECK-SAME: (%[[ARG:.*]]: tensor<?x32xi16>)
func @shape_of_unary(%arg : tensor<?x32xi16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]] : tensor<?x32xi16> -> tensor<2xindex>
  // CHECK: "use"(%[[SHAPE]])
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
  // CHECK: "use"(%[[SHAPE]])
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf16>
  %1 = mhlo.subtract %0, %arg1 : tensor<?x32xf16>
  %2 = shape.shape_of %1 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%2) : (tensor<?xindex>) -> ()
  return
}
