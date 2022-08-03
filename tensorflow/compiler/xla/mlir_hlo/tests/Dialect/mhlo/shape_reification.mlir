// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect \
// RUN:   --shape-reification | \
// RUN: FileCheck %s

// CHECK-LABEL: @shape_of_unary
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x32xi16>
func.func @shape_of_unary(%arg : tensor<?x32xi16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG]]
  // CHECK: "use"(%[[SHAPE]])
  %0 = "mhlo.convert"(%arg) : (tensor<?x32xi16>) -> tensor<?x32xf16>
  %1 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<2xindex>
  "use"(%1) : (tensor<2xindex>) -> ()
  func.return
}

// -----

// CHECK-LABEL: @shape_of_nary
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>
func.func @shape_of_nary(%arg0 : tensor<?x32xf16>, %arg1 : tensor<?x32xf16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG0]]
  // CHECK: "use"(%[[SHAPE]])
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf16>
  %1 = mhlo.subtract %0, %arg1 : tensor<?x32xf16>
  %2 = shape.shape_of %1 : tensor<?x32xf16> -> tensor<2xindex>
  "use"(%2) : (tensor<2xindex>) -> ()
  func.return
}

// -----

// CHECK-LABEL: @insert_cast_if_needed
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>
func.func @insert_cast_if_needed(%arg0 : tensor<?x32xf16>,
    %arg1 : tensor<?x32xf16>) {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG0]] : tensor<?x32xf16> -> tensor<2xindex>
  // CHECK: %[[CASTED:.*]] = tensor.cast %[[SHAPE]] : tensor<2xindex> to tensor<?xindex>
  // CHECK: "use"(%[[CASTED]]) : (tensor<?xindex>) -> ()
  %0 = mhlo.subtract %arg0, %arg1 : tensor<?x32xf16>
  %2 = shape.shape_of %0 : tensor<?x32xf16> -> tensor<?xindex>
  "use"(%2) : (tensor<?xindex>) -> ()
  func.return
}

// -----

// CHECK-LABEL: @move_shape_of_into_assuming
// CHECK-SAME:  %[[ARG0:.*]]: !shape.witness, %[[ARG1:.*]]: tensor<?x32xf32>
func.func @move_shape_of_into_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<?x32xf32>) -> tensor<2xindex> {
  // CHECK:     %[[ASSUMING_RESULTS:.*]]:3 = shape.assuming %[[ARG0]] -> (tensor<?x32xf32>, tensor<?x32xf32>, tensor<2xindex>) {
  // CHECK:       %[[DUMMY_TENSOR:.*]] = "some.tensor"() : () -> tensor<?x32xf32>
  // CHECK:       %[[SHAPE:.*]] = shape.shape_of %[[DUMMY_TENSOR]]
  // CHECK:       shape.assuming_yield %[[ARG1]], %[[DUMMY_TENSOR]], %[[SHAPE]]
  // CHECK:     }
  // CHECK-NOT: shape_of
  // CHECK:     return %[[ASSUMING_RESULTS]]#2
  %0:2 = shape.assuming %arg0 -> (tensor<?x32xf32>, tensor<?x32xf32>) {
    %1 = "some.tensor"() : () -> tensor<?x32xf32>
    shape.assuming_yield %arg1, %1 : tensor<?x32xf32>, tensor<?x32xf32>
  }
  %2 = shape.shape_of %0#1 : tensor<?x32xf32> -> tensor<2xindex>
  "use"(%0#0, %0#1) : (tensor<?x32xf32>, tensor<?x32xf32>) -> ()
  func.return %2 : tensor<2xindex>
}

// -----

// CHECK-LABEL: @tree_conjunction
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xi1>, %[[ARG1:.*]]: tensor<?xi1>, %[[ARG2:.*]]: tensor<?xi1>, %[[ARG3:.*]]: tensor<?xi1>
func.func @tree_conjunction(%arg0: tensor<?xi1>, %arg1: tensor<?xi1>,
    %arg2: tensor<?xi1>, %arg3: tensor<?xi1>) -> tensor<?xi1> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[W01:.*]] = shape.cstr_broadcastable %[[S0]], %[[S1]]
  // CHECK:     %[[CONJ01_S01:.*]]:2 = shape.assuming %[[W01]]
  // CHECK-DAG:   %[[S01:.*]] = shape.broadcast %[[S0]], %[[S1]]
  // CHECK-DAG:   %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[S01]])
  // CHECK-DAG:   %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[S01]])
  // CHECK-DAG:   %[[CONJ01:.*]] = mhlo.and %[[BCASTED_ARG0]], %[[BCASTED_ARG1]]
  // CHECK:       shape.assuming_yield %[[CONJ01]], %[[S01]]
  // CHECK-DAG: %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[S3:.*]] = shape.shape_of %[[ARG3]]
  // CHECK-DAG: %[[W23:.*]] = shape.cstr_broadcastable %[[S2]], %[[S3]]
  // CHECK:     %[[CONJ23_S23:.*]]:2 = shape.assuming %[[W23]]
  // CHECK-DAG:   %[[S23:.*]] = shape.broadcast %[[S2]], %[[S3]]
  // CHECK-DAG:   %[[BCASTED_ARG2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG2]], %[[S23]])
  // CHECK-DAG:   %[[BCASTED_ARG3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG3]], %[[S23]])
  // CHECK-DAG:   %[[CONJ23:.*]] = mhlo.and %[[BCASTED_ARG2]], %[[BCASTED_ARG3]]
  // CHECK:       shape.assuming_yield %[[CONJ23]], %[[S23]]
  // CHECK-DAG: %[[W0123:.*]] = shape.cstr_broadcastable %[[CONJ01_S01]]#1, %[[CONJ23_S23]]#1
  // CHECK:     %[[CONJ0123_:.*]] = shape.assuming %[[W0123]]
  // CHECK-DAG:   %[[S0123:.*]] = shape.broadcast %[[CONJ01_S01]]#1, %[[CONJ23_S23]]#1
  // CHECK-DAG:   %[[BCASTED_CONJ01:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CONJ01_S01]]#0, %[[S0123]])
  // CHECK-DAG:   %[[BCASTED_CONJ23:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CONJ23_S23]]#0, %[[S0123]])
  // CHECK-DAG:   %[[CONJ0123:.*]] = mhlo.and %[[BCASTED_CONJ01]], %[[BCASTED_CONJ23]]
  // CHECK:       shape.assuming_yield %[[CONJ0123]]
  // CHECK:     return %[[CONJ0123_]]
  %0 = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xi1> -> tensor<1xindex>
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<1xindex>
  %3 = shape.assuming %2 -> (tensor<?xi1>) {
    %12 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %15 = mhlo.and %13, %14 : tensor<?xi1>
    shape.assuming_yield %15 : tensor<?xi1>
  }
  %4 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %5 = shape.shape_of %arg3 : tensor<?xi1> -> tensor<1xindex>
  %6 = shape.cstr_broadcastable %4, %5 : tensor<1xindex>, tensor<1xindex>
  %7 = shape.assuming %6 -> (tensor<?xi1>) {
    %12 = shape.broadcast %4, %5 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = "mhlo.dynamic_broadcast_in_dim"(%arg3, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %15 = mhlo.and %13, %14 : tensor<?xi1>
    shape.assuming_yield %15 : tensor<?xi1>
  }
  %8 = shape.shape_of %3 : tensor<?xi1> -> tensor<1xindex>
  %9 = shape.shape_of %7 : tensor<?xi1> -> tensor<1xindex>
  %10 = shape.cstr_broadcastable %8, %9 : tensor<1xindex>, tensor<1xindex>
  %11 = shape.assuming %10 -> (tensor<?xi1>) {
    %12 = shape.broadcast %8, %9 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%3, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = "mhlo.dynamic_broadcast_in_dim"(%7, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %15 = mhlo.and %13, %14 : tensor<?xi1>
    shape.assuming_yield %15 : tensor<?xi1>
  }
  func.return %11 : tensor<?xi1>
}

// -----

// CHECK-LABEL: @logical_and_chain
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xi1>, %[[ARG1:.*]]: tensor<?xi1>, %[[ARG2:.*]]: tensor<?xi1>
func.func @logical_and_chain(%arg0: tensor<?xi1>, %arg1: tensor<?xi1>,
    %arg2: tensor<?xi1>) -> tensor<?xi1> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %{{.*}} = "mhlo.dynamic_broadcast_in_dim"(%{{.*}}, %[[S0]])
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[S0_:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[W:.*]] = shape.cstr_broadcastable %[[S1]], %[[S0_]]
  // CHECK:     %[[S10_:.*]]:2 = shape.assuming %[[W]]
  // CHECK-DAG:   %[[S10:.*]] = shape.broadcast %[[S1]], %[[S0_]]
  // CHECK-DAG:   %{{.*}} = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[S10]])
  // CHECK-DAG:   %{{.*}} = "mhlo.dynamic_broadcast_in_dim"(%{{.*}}, %[[S10]])
  // CHECK:       shape.assuming_yield %{{.*}}, %[[S10]]
  // CHECK-DAG: %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[W:.*]] = shape.cstr_broadcastable %[[S2]], %[[S10_]]#1
  // CHECK:     %{{.*}} = shape.assuming %[[W]]
  // CHECK-DAG:   %[[S210:.*]] = shape.broadcast %[[S2]], %[[S10_]]#1
  // CHECK-DAG:   %{{.*}} = "mhlo.dynamic_broadcast_in_dim"(%[[ARG2]], %[[S210]])
  // CHECK-DAG:   %{{.*}} = "mhlo.dynamic_broadcast_in_dim"(%{{.*}}, %[[S210]])
  // CHECK:       shape.assuming_yield %{{.*}}
  // CHECK:     return %{{.*}}
  %0 = mhlo.constant dense<true> : tensor<i1>
  %1 = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1)
      {broadcast_dimensions = dense<> : tensor<0xi64>}
      : (tensor<i1>, tensor<1xindex>) -> tensor<?xi1>
  %3 = mhlo.and %arg0, %2 : tensor<?xi1>
  %4 = shape.shape_of %arg1 : tensor<?xi1> -> tensor<1xindex>
  %5 = shape.shape_of %3 : tensor<?xi1> -> tensor<1xindex>
  %6 = shape.cstr_broadcastable %4, %5 : tensor<1xindex>, tensor<1xindex>
  %7 = shape.assuming %6 -> (tensor<?xi1>) {
    %12 = shape.broadcast %4, %5 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = "mhlo.dynamic_broadcast_in_dim"(%3, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %15 = mhlo.and %13, %14 : tensor<?xi1>
    shape.assuming_yield %15 : tensor<?xi1>
  }
  %8 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %9 = shape.shape_of %7 : tensor<?xi1> -> tensor<1xindex>
  %10 = shape.cstr_broadcastable %8, %9 : tensor<1xindex>, tensor<1xindex>
  %11 = shape.assuming %10 -> (tensor<?xi1>) {
    %12 = shape.broadcast %8, %9 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = "mhlo.dynamic_broadcast_in_dim"(%7, %12)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %15 = mhlo.and %13, %14 : tensor<?xi1>
    shape.assuming_yield %15 : tensor<?xi1>
  }
  func.return %11 : tensor<?xi1>
}
