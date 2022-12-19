// RUN: tf-tfrt-opt %s --split-input-file \
// RUN:     --tf-jitrt-symbolic-shape-optimization | \
// RUN: FileCheck %s

// CHECK-LABEL: @optimize_1dx1d_bcast
// CHECK-SAME:  %[[ARG0:.*]]: tensor<?xf32> {rt.symbolic_shape = dense<-2> : tensor<1xi64>}, %[[ARG1:.*]]: tensor<?xf32> {rt.symbolic_shape = dense<-2> : tensor<1xi64>}
func.func @optimize_1dx1d_bcast(
  %arg0: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg1: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> tensor<?xf32> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[ARG0]]
  // CHECK: %[[DYNAMIC:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[SHAPE]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK: return %[[DYNAMIC]]
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
      -> tensor<1xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
       : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %3: tensor<?xf32>
}

// -----

// CHECK-LABEL: @optimize_1dx2d_bcast_const_shape
// CHECK-SAME:  %[[ARG0_0:.*]]: tensor<512xf32>, %[[ARG1_0:.*]]: tensor<?x512xf32> {rt.symbolic_shape = dense<[-2, 512]> : tensor<2xi64>}
func.func @optimize_1dx2d_bcast_const_shape(
  %arg0: tensor<512xf32>,
  %arg1: tensor<?x512xf32>
    {rt.symbolic_shape = dense<[-2, 512]> : tensor<2xi64>}
) -> tensor<?x512xf32> {
  // CHECK: %[[SHAPE_0:.*]] = shape.shape_of %[[ARG1_0]]
  // CHECK: %[[DYNAMIC_0:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0_0]], %[[SHAPE_0]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK: return %[[DYNAMIC_0]]
  %0 = shape.const_shape [512] : tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?x512xf32> -> tensor<2xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<2xindex>
                             -> tensor<2xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
       : (tensor<512xf32>, tensor<2xindex>) -> tensor<?x512xf32>
  func.return %3: tensor<?x512xf32>
}

// -----

// CHECK-LABEL: @optimize_1dx1dx1d_bcast
// CHECK:       %[[ARG0_1:.*]]: tensor<?xf32> {rt.symbolic_shape = dense<-2> : tensor<1xi64>}, %[[ARG1_1:.*]]: tensor<?xf32> {rt.symbolic_shape = dense<-2> : tensor<1xi64>}, %[[ARG2:.*]]: tensor<?xf32> {rt.symbolic_shape = dense<-2> : tensor<1xi64>}
func.func @optimize_1dx1dx1d_bcast(
  %arg0: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg1: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg2: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> tensor<?xf32> {
  // CHECK: %[[SHAPE_1:.*]] = shape.shape_of %[[ARG0_1]]
  // CHECK: %[[DYNAMIC_1:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0_1]], %[[SHAPE_1]]) {broadcast_dimensions = dense<0> : tensor<1xi64>}
  // CHECK: return %[[DYNAMIC_1]]
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.shape_of %arg2 : tensor<?xf32> -> tensor<1xindex>
  %3 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
                             -> tensor<1xindex>
  %4 = shape.broadcast %3, %2 : tensor<1xindex>, tensor<1xindex>
                             -> tensor<1xindex>
  %5 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %4)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
       : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %5: tensor<?xf32>
}

// -----

// CHECK-LABEL: @optimize_2dx1d_bcast
// CHECK-SAME:  %[[ARG0_2:.*]]: tensor<10x?xf32> {rt.symbolic_shape = dense<[10, -2]> : tensor<2xi64>}, %[[ARG1_2:.*]]: tensor<?xf32> {rt.symbolic_shape = dense<-2> : tensor<1xi64>}
func.func @optimize_2dx1d_bcast(
  %arg0: tensor<10x?xf32>
    {rt.symbolic_shape = dense<[10, -2]> : tensor<2xi64>},
  %arg1: tensor<?xf32>
    {rt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> (tensor<10x?xf32>, tensor<10x?xf32>) {
  // CHECK: %[[SHAPE_2:.*]] = shape.shape_of %[[ARG0_2]]
  // CHECK: %[[DYNAMIC_2:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0_2]], %[[SHAPE_2]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK: %[[DYNAMIC_3:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1_2]], %[[SHAPE_2]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK: return %[[DYNAMIC_2]], %[[DYNAMIC_3]]
  %0 = shape.shape_of %arg0 : tensor<10x?xf32> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<2xindex>, tensor<1xindex>
                             -> tensor<2xindex>
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
       : (tensor<10x?xf32>, tensor<2xindex>) -> tensor<10x?xf32>
  %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2)
         {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
       : (tensor<?xf32>, tensor<2xindex>) -> tensor<10x?xf32>
  func.return %3, %4: tensor<10x?xf32>, tensor<10x?xf32>
}
