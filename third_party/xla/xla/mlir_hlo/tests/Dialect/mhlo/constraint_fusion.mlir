// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect \
// RUN:   --constraint-fusion | \
// RUN: FileCheck %s

// CHECK-LABEL: @tree_conjunction
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xi1>, %[[ARG1:.*]]: tensor<?xi1>, %[[ARG2:.*]]: tensor<?xi1>, %[[ARG3:.*]]: tensor<?xi1>
func.func @tree_conjunction(%arg0: tensor<?xi1>, %arg1: tensor<?xi1>,
    %arg2: tensor<?xi1>, %arg3: tensor<?xi1>) -> tensor<?xi1> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[S3:.*]] = shape.shape_of %[[ARG3]]
  // CHECK-DAG: %[[COMBINED_W:.*]] = shape.cstr_broadcastable %[[S0]], %[[S1]], %[[S2]], %[[S3]]
  // CHECK:     %[[RES:.*]] = shape.assuming %[[COMBINED_W]]
  // CHECK-DAG:   %[[S0_:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG:   %[[S1_:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG:   %[[S01:.*]] = shape.broadcast %[[S0_]], %[[S1_]]
  // CHECK-DAG:   %[[BCASTED_ARG0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[S01]])
  // CHECK-DAG:   %[[BCASTED_ARG1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[S01]])
  // CHECK-DAG:   %[[CONJ01:.*]] = mhlo.and %[[BCASTED_ARG0]], %[[BCASTED_ARG1]]
  // CHECK-DAG:   %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG:   %[[S3:.*]] = shape.shape_of %[[ARG3]]
  // CHECK-DAG:   %[[S23:.*]] = shape.broadcast %[[S2]], %[[S3]]
  // CHECK-DAG:   %[[BCASTED_ARG2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG2]], %[[S23]])
  // CHECK-DAG:   %[[BCASTED_ARG3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG3]], %[[S23]])
  // CHECK-DAG:   %[[CONJ23:.*]] = mhlo.and %[[BCASTED_ARG2]], %[[BCASTED_ARG3]]
  // CHECK-DAG:   %[[S0123:.*]] = shape.broadcast %[[S01]], %[[S23]]
  // CHECK-DAG:   %[[BCASTED_CONJ01:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CONJ01]], %[[S0123]])
  // CHECK-DAG:   %[[BCASTED_CONJ23:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[CONJ23]], %[[S0123]])
  // CHECK-DAG:   %[[CONJ0123:.*]] = mhlo.and %[[BCASTED_CONJ01]], %[[BCASTED_CONJ23]]
  // CHECK:       shape.assuming_yield %[[CONJ0123]]
  // CHECK:     return %[[RES]]
  %0 = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xi1> -> tensor<1xindex>
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<1xindex>
  %3:2 = shape.assuming %2 -> (tensor<?xi1>, tensor<1xindex>) {
    %10 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %11 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %10)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %10)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %13 = mhlo.and %11, %12 : tensor<?xi1>
    shape.assuming_yield %13, %10 : tensor<?xi1>, tensor<1xindex>
  }
  %4 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %5 = shape.shape_of %arg3 : tensor<?xi1> -> tensor<1xindex>
  %6 = shape.cstr_broadcastable %4, %5 : tensor<1xindex>, tensor<1xindex>
  %7:2 = shape.assuming %6 -> (tensor<?xi1>, tensor<1xindex>) {
    %10 = shape.broadcast %4, %5 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %11 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %10)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%arg3, %10)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %13 = mhlo.and %11, %12 : tensor<?xi1>
    shape.assuming_yield %13, %10 : tensor<?xi1>, tensor<1xindex>
  }
  %8 = shape.cstr_broadcastable %3#1, %7#1 : tensor<1xindex>, tensor<1xindex>
  %9 = shape.assuming %8 -> (tensor<?xi1>) {
    %10 = shape.broadcast %3#1, %7#1 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %11 = "mhlo.dynamic_broadcast_in_dim"(%3#0, %10)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%7#0, %10)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %13 = mhlo.and %11, %12 : tensor<?xi1>
    shape.assuming_yield %13 : tensor<?xi1>
  }
  func.return %9 : tensor<?xi1>
}

// -----

// CHECK-LABEL: @eliminate_duplicates
// CHECK-SAME:  %[[W:.*]]: !shape.witness, %[[S:.*]]: tensor<?xindex>, %[[ARG:.*]]: tensor<?xi1>
func.func @eliminate_duplicates(%arg0: !shape.witness, %arg1: tensor<?xindex>,
    %arg2: tensor<?xi1>) -> (tensor<?xf32>, tensor<?xf32>) {
  // CHECK-DAG: %[[SARG:.*]] = shape.shape_of %[[ARG]]
  // CHECK-DAG: %[[BCAST_CSTR:.*]] = shape.cstr_broadcastable %[[S]], %[[SARG]]
  // CHECK-DAG: %[[COMBINED_W:.*]] = shape.assuming_all %[[W]], %[[BCAST_CSTR]]
  // CHECK:     %[[RES:.*]]:2 = shape.assuming %[[COMBINED_W]]
  // CHECK:       %[[RESA:.*]] = "some.a"
  // CHECK:       %[[RESB:.*]] = "some.b"
  // CHECK:       shape.assuming_yield %[[RESA]], %[[RESB]]
  // CHECK:     return %[[RES]]#0, %[[RES]]#1
  %0 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %1 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %2 = shape.cstr_broadcastable %arg1, %arg1, %0, %1
      : tensor<?xindex>, tensor<?xindex>, tensor<1xindex>, tensor<1xindex>
  %3 = shape.cstr_broadcastable %arg1, %0, %1
      : tensor<?xindex>, tensor<1xindex>, tensor<1xindex>
  %4 = shape.assuming_all %arg0, %arg0, %3, %2
  %5 = shape.assuming %4 -> (tensor<?xf32>) {
    %9 = "some.a"() : () -> tensor<?xf32>
    shape.assuming_yield %9 : tensor<?xf32>
  }
  %6 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %7 = shape.cstr_broadcastable %arg1, %arg1, %6
      : tensor<?xindex>, tensor<?xindex>, tensor<1xindex>
  %8 = shape.assuming %7 -> (tensor<?xf32>) {
    %9 = "some.b"() : () -> tensor<?xf32>
    shape.assuming_yield %9 : tensor<?xf32>
  }
  func.return %5, %8 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: @inline_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>, %[[ARG2:.*]]: tensor<?x?x32xf16>
func.func @inline_bcast(%arg0: tensor<?x32xf16>, %arg1: tensor<?x32xf16>,
    %arg2: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[W:.*]] = shape.cstr_broadcastable %[[S0]], %[[S1]], %[[S2]]
  // CHECK:     %[[RES:.*]] = shape.assuming %[[W]]
  // CHECK:       %[[INNER_RES:.*]] = "some.op"
  // CHECK:       shape.assuming_yield %[[INNER_RES]]
  // CHECK:     return %[[RES]]
  %0 = shape.shape_of %arg0 : tensor<?x32xf16> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?x32xf16> -> tensor<2xindex>
  %2 = shape.cstr_broadcastable %0, %1 : tensor<2xindex>, tensor<2xindex>
  %3 = shape.assuming %2 -> (tensor<2xindex>) {
    %7 = shape.broadcast %0, %1 : tensor<2xindex>, tensor<2xindex>
        -> tensor<2xindex>
    shape.assuming_yield %7 : tensor<2xindex>
  }
  %4 = shape.shape_of %arg2 : tensor<?x?x32xf16> -> tensor<3xindex>
  %5 = shape.cstr_broadcastable %3, %4 : tensor<2xindex>, tensor<3xindex>
  %6 = shape.assuming %5 -> (tensor<?x?x32xf16>) {
    %7 = "some.op"() : () -> tensor<?x?x32xf16>
    shape.assuming_yield %7 : tensor<?x?x32xf16>
  }
  func.return %6 : tensor<?x?x32xf16>
}

// -----

// CHECK-LABEL: @move_cstr_broadcastable_out_of_assuming
// CHECK-SAME:  %[[W:.*]]: !shape.witness, %[[ARG0:.*]]: tensor<2xindex>, %[[ARG1:.*]]: tensor<3xindex>
func.func @move_cstr_broadcastable_out_of_assuming(%arg0 : !shape.witness,
    %arg1 : tensor<2xindex>, %arg2 : tensor<3xindex>) -> tensor<f32> {
  // CHECK: %[[BCASTABLE:.*]] = shape.cstr_broadcastable %[[ARG0]], %[[ARG1]]
  // CHECK: %[[COMBINED_W:.*]] = shape.assuming_all %[[W]], %[[BCASTABLE]]
  // CHECK: %[[RES:.*]] = shape.assuming %[[COMBINED_W]]
  // CHECK:   %[[INNER_RES:.*]] = "some.op"
  // CHECK:   shape.assuming_yield %[[INNER_RES]]
  // CHECK: return %[[RES:.*]]
  %0 = shape.assuming %arg0 -> (!shape.witness) {
    %2 = shape.cstr_broadcastable %arg1, %arg2
        : tensor<2xindex>, tensor<3xindex>
    shape.assuming_yield %2 : !shape.witness
  }
  %1 = shape.assuming %0 -> (tensor<f32>) {
    %2 = "some.op"() : () -> tensor<f32>
    shape.assuming_yield %2 : tensor<f32>
  }
  func.return %1 : tensor<f32>
}

// -----

// CHECK-LABEL: @elementwise_op
// CHECK-SAME:  %[[W:.*]]: !shape.witness, %[[ARG:.*]]: tensor<?xf32>
func.func @elementwise_op(%arg0 : !shape.witness, %arg1 : tensor<?xf32>)
    -> tensor<?xf32> {
  // CHECK:     %[[RES:.*]] = shape.assuming %[[W]]
  // CHECK-DAG:   %[[INNER_RES:.*]] = mhlo.tanh %[[ARG]]
  // CHECK:       shape.assuming_yield %[[INNER_RES]]
  // CHECK:     return %[[RES]]
  %0 = shape.assuming %arg0 -> tensor<?xf32> {
    shape.assuming_yield %arg1 : tensor<?xf32>
  }
  %1 = mhlo.tanh %0 : tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @multiple_assuming_regions
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>, %[[ARG2:.*]]: tensor<?x?x32xf16>
func.func @multiple_assuming_regions(%arg0: tensor<?x32xf16>,
    %arg1 : tensor<?x32xf16>, %arg2: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> {
  // CHECK-DAG: %[[SHAPE0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[SHAPE1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[SHAPE2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK:     %[[WITNESS:.*]] = shape.cstr_broadcastable %[[SHAPE0]], %[[SHAPE1]], %[[SHAPE2]]
  // CHECK:     %[[RES:.*]] = shape.assuming %[[WITNESS]]
  // CHECK:       "some.op"
  // CHECK:       %[[SOME_PROD:.*]] = "some.producer"
  // CHECK:       "another.op"
  // CHECK:       %[[INNER_RES:.*]] = "another.producer"
  // CHECK:       "use"(%[[SOME_PROD]], %[[INNER_RES]])
  // CHECK:       shape.assuming_yield %[[INNER_RES]]
  // CHECK:     return %[[RES]]
  %0 = shape.shape_of %arg0 : tensor<?x32xf16> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?x32xf16> -> tensor<2xindex>
  %2 = shape.shape_of %arg2 : tensor<?x?x32xf16> -> tensor<3xindex>
  %3 = shape.cstr_broadcastable %0, %1 : tensor<2xindex>, tensor<2xindex>
  %4 = shape.cstr_broadcastable %0, %1, %2 : tensor<2xindex>, tensor<2xindex>,
      tensor<3xindex>
  %5 = shape.assuming %3 -> (tensor<?x32xf16>) {
    "some.op"() : () -> ()
    %6 = "some.producer"() : () -> tensor<?x32xf16>
    shape.assuming_yield %6 : tensor<?x32xf16>
  }
  %7 = shape.assuming %4 -> (tensor<?x?x32xf16>) {
    "another.op"() : () -> ()
    %8 = "another.producer"() : () -> tensor<?x?x32xf16>
    shape.assuming_yield %8 : tensor<?x?x32xf16>
  }
  "use"(%5, %7) : (tensor<?x32xf16>, tensor<?x?x32xf16>) -> ()
  func.return %7 : tensor<?x?x32xf16>
}

// -----

// CHECK-LABEL: @redundant_cstrs
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?x32xf16>, %[[ARG1:.*]]: tensor<?x32xf16>, %[[ARG2:.*]]: tensor<?x?x32xf16>, %[[SARG0:.*]]: tensor<?xindex>, %[[SARG1:.*]]: tensor<?xindex>, %[[SARG2:.*]]: tensor<?xindex>
func.func @redundant_cstrs(%arg0: tensor<?x32xf16>, %arg1: tensor<?x32xf16>,
    %arg2: tensor<?x?x32xf16>, %arg3: tensor<?xindex>, %arg4: tensor<?xindex>,
    %arg5: tensor<?xindex>) {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[CSTR0:.*]] = shape.cstr_broadcastable %[[SARG0]], %[[SARG1]], %[[SARG2]]
  // CHECK-DAG: %[[CSTR1:.*]] = shape.cstr_broadcastable %[[SARG0]], %[[S1]], %[[S2]]
  // CHECK-DAG: %[[CSTR2:.*]] = shape.cstr_broadcastable %[[S0]], %[[S1]]
  // CHECK-DAG: %[[W:.*]] = shape.assuming_all %[[CSTR1]], %[[CSTR0]], %[[CSTR2]]
  // CHECK:     shape.assuming %[[W]]
  // CHECK:       "some.op"
  // CHECK:     return
  %s0 = shape.shape_of %arg0 : tensor<?x32xf16> -> tensor<2xindex>
  %s1 = shape.shape_of %arg1 : tensor<?x32xf16> -> tensor<2xindex>
  %s2 = shape.shape_of %arg2 : tensor<?x?x32xf16> -> tensor<3xindex>
  %cstr0 = shape.cstr_broadcastable %arg4, %arg3
      : tensor<?xindex>, tensor<?xindex>
  %cstr1 = shape.cstr_broadcastable %arg3, %arg4
      : tensor<?xindex>, tensor<?xindex>
  %cstr2 = shape.cstr_broadcastable %arg5, %arg3, %arg4
      : tensor<?xindex>, tensor<?xindex>, tensor<?xindex>
  %cstr3 = shape.cstr_broadcastable %s1, %s0 : tensor<2xindex>, tensor<2xindex>
  %cstr4 = shape.cstr_broadcastable %s1, %s2 : tensor<2xindex>, tensor<3xindex>
  %cstr5 = shape.cstr_broadcastable %s2, %s1, %arg3
      : tensor<3xindex>, tensor<2xindex>, tensor<?xindex>
  %combined_cstr = shape.assuming_all %cstr0, %cstr1, %cstr2, %cstr3, %cstr4,
      %cstr5
  shape.assuming %combined_cstr {
    "some.op"() : () -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: @duplicate_cstr_args
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x32xf16>, %[[SHAPE:.*]]: tensor<?xindex>
func.func @duplicate_cstr_args(%arg: tensor<?x32xf16>, %shape: tensor<?xindex>) {
  // CHECK-DAG: %[[S:.*]] = shape.shape_of %[[ARG]]
  // CHECK-DAG: %[[CSTR:.*]] = shape.cstr_broadcastable %[[SHAPE]], %[[S]]
  // CHECK:     shape.assuming %[[CSTR]]
  // CHECK:       "some.op"
  // CHECK:     return
  %s = shape.shape_of %arg : tensor<?x32xf16> -> tensor<2xindex>
  %cstr = shape.cstr_broadcastable %s, %shape, %shape, %s
      : tensor<2xindex>, tensor<?xindex>, tensor<?xindex>, tensor<2xindex>
  shape.assuming %cstr {
    "some.op"() : () -> ()
  }
  func.return
}

// -----

// CHECK-LABEL: @logical_and_chain
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xi1>, %[[ARG1:.*]]: tensor<?xi1>, %[[ARG2:.*]]: tensor<?xi1>
func.func @logical_and_chain(%arg0: tensor<?xi1>, %arg1: tensor<?xi1>,
    %arg2: tensor<?xi1>) -> tensor<?xi1> {
  // CHECK-DAG: %[[S0:.*]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[S1:.*]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[S2:.*]] = shape.shape_of %[[ARG2]]
  // CHECK-DAG: %[[W:.*]] = shape.cstr_broadcastable %[[S0]], %[[S1]], %[[S2]]
  // CHECK:     %[[RES:.*]] = shape.assuming %[[W]]
  // CHECK:       shape.assuming_yield
  // CHECK-NOT: shape.assuming
  // CHECK:     return %4
  %0 = mhlo.constant dense<true> : tensor<i1>
  %1 = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1)
      {broadcast_dimensions = dense<> : tensor<0xi64>}
      : (tensor<i1>, tensor<1xindex>) -> tensor<?xi1>
  %3 = mhlo.and %arg0, %2 : tensor<?xi1>
  %4 = shape.shape_of %arg1 : tensor<?xi1> -> tensor<1xindex>
  %5 = shape.shape_of %arg0 : tensor<?xi1> -> tensor<1xindex>
  %6 = shape.cstr_broadcastable %4, %5 : tensor<1xindex>, tensor<1xindex>
  %7:2 = shape.assuming %6 -> (tensor<?xi1>, tensor<1xindex>) {
    %11 = shape.broadcast %4, %5 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %11)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%3, %11)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = mhlo.and %12, %13 : tensor<?xi1>
    shape.assuming_yield %14, %11 : tensor<?xi1>, tensor<1xindex>
  }
  %8 = shape.shape_of %arg2 : tensor<?xi1> -> tensor<1xindex>
  %9 = shape.cstr_broadcastable %8, %7#1 : tensor<1xindex>, tensor<1xindex>
  %10 = shape.assuming %9 -> (tensor<?xi1>) {
    %11 = shape.broadcast %8, %7#1 : tensor<1xindex>, tensor<1xindex>
        -> tensor<1xindex>
    %12 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %11)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %13 = "mhlo.dynamic_broadcast_in_dim"(%7#0, %11)
        {broadcast_dimensions = dense<0> : tensor<1xi64>}
        : (tensor<?xi1>, tensor<1xindex>) -> tensor<?xi1>
    %14 = mhlo.and %12, %13 : tensor<?xi1>
    shape.assuming_yield %14 : tensor<?xi1>
  }
  func.return %10 : tensor<?xi1>
}
