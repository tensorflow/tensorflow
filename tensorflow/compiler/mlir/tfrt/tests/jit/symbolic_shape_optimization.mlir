// RUN: tf-tfrt-opt %s -split-input-file -tf-cpurt-symbolic-shape-optimization \
// RUN: | FileCheck %s

// CHECK-LABEL: @optimize_1dx1d_constraint
func @optimize_1dx1d_constraint(
  %arg0: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg1: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> !shape.witness {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  // CHECK: shape.const_witness true
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<1xindex>
  return %2: !shape.witness
}

// -----

// CHECK-LABEL: @optimize_1dx1d_constraint_with_static_shape
func @optimize_1dx1d_constraint_with_static_shape(
  %arg0: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[10]> : tensor<1xi64>},
  %arg1: tensor<10xf32>
) -> !shape.witness {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<10xf32> -> tensor<1xindex>
  // CHECK: shape.const_witness true
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<1xindex>
  return %2: !shape.witness
}

// -----

// CHECK-LABEL: @optimize_1dx1d_constraint_with_const_shape
func @optimize_1dx1d_constraint_with_const_shape(
  %arg0: tensor<512xf32>,
  %arg1: tensor<?x512xf32>
    {cpurt.symbolic_shape = dense<[-2,512]> : tensor<2xi64>}
) -> !shape.witness {
  %0 = shape.const_shape [512] : tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?x512xf32> -> tensor<2xindex>
  // CHECK: shape.const_witness true
  %2 = shape.cstr_broadcastable %0, %1 : tensor<1xindex>, tensor<2xindex>
  return %2: !shape.witness
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>

// CHECK:       @optimize_1dx1d_bcast(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: tensor<?xf32>
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: tensor<?xf32>
func @optimize_1dx1d_bcast(
  %arg0: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg1: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> tensor<?xf32> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
                             -> tensor<1xindex>

  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[OUT:.*]] = linalg.init_tensor [%[[D0]]]
  // CHECK:      %[[RET:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK-SAME: ins(%[[ARG0]] : tensor<?xf32>)
  // CHECK-SAME: outs(%[[OUT]] : tensor<?xf32>)
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
       : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>

  return %3: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:       @optimize_1dx2d_bcast_const_shape(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: tensor<512xf32>
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: tensor<?x512xf32>
func @optimize_1dx2d_bcast_const_shape(
  %arg0: tensor<512xf32>,
  %arg1: tensor<?x512xf32>
    {cpurt.symbolic_shape = dense<[-2, 512]> : tensor<2xi64>}
) -> tensor<?x512xf32> {
  %0 = shape.const_shape [512] : tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?x512xf32> -> tensor<2xindex>
  %2 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<2xindex>
                             -> tensor<2xindex>

  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[D0:.*]] = tensor.dim %[[ARG1]], %[[C0]]
  // CHECK:      %[[OUT:.*]] = linalg.init_tensor [%[[D0]], 512]
  // CHECK:      %[[RET:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[ARG0]] : tensor<512xf32>)
  // CHECK-SAME: outs(%[[OUT]] : tensor<?x512xf32>)
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
       : (tensor<512xf32>, tensor<2xindex>) -> tensor<?x512xf32>

  return %3: tensor<?x512xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>

// CHECK:       @optimize_1dx1dx1d_bcast(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: tensor<?xf32>
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: tensor<?xf32>
// CHECK-SAME:    %[[ARG2:[a-z0-9]+]]: tensor<?xf32>
func @optimize_1dx1dx1d_bcast(
  %arg0: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg1: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>},
  %arg2: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> tensor<?xf32> {
  %0 = shape.shape_of %arg0 : tensor<?xf32> -> tensor<1xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.shape_of %arg2 : tensor<?xf32> -> tensor<1xindex>
  %3 = shape.broadcast %0, %1 : tensor<1xindex>, tensor<1xindex>
                             -> tensor<1xindex>
  %4 = shape.broadcast %3, %2 : tensor<1xindex>, tensor<1xindex>
                             -> tensor<1xindex>

  // CHECK:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK:      %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[OUT:.*]] = linalg.init_tensor [%[[D0]]]
  // CHECK:      %[[RET:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]]]
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK-SAME: ins(%[[ARG0]] : tensor<?xf32>)
  // CHECK-SAME: outs(%[[OUT]] : tensor<?xf32>)
  %5 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %4)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
       : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>

  return %5: tensor<?xf32>
}

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK:       @optimize_2dx1d_bcast(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: tensor<10x?xf32>
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: tensor<?xf32>
func @optimize_2dx1d_bcast(
  %arg0: tensor<10x?xf32>
    {cpurt.symbolic_shape = dense<[10, -2]> : tensor<2xi64>},
  %arg1: tensor<?xf32>
    {cpurt.symbolic_shape = dense<[-2]> : tensor<1xi64>}
) -> (tensor<10x?xf32>, tensor<10x?xf32>) {
  %0 = shape.shape_of %arg0 : tensor<10x?xf32> -> tensor<2xindex>
  %1 = shape.shape_of %arg1 : tensor<?xf32> -> tensor<1xindex>
  %2 = shape.broadcast %0, %1 : tensor<2xindex>, tensor<1xindex>
                             -> tensor<2xindex>

  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK:      %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]

  // CHECK:      %[[OUT0:.*]] = linalg.init_tensor [10, %[[D1]]]
  // CHECK:      %[[RET0:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP0]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[ARG0]] : tensor<10x?xf32>)
  // CHECK-SAME: outs(%[[OUT0]] : tensor<10x?xf32>)
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
       : (tensor<10x?xf32>, tensor<2xindex>) -> tensor<10x?xf32>

  // CHECK-DAG:  %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK:      %[[OUT1:.*]] = linalg.init_tensor [10, %[[D0]]]
  // CHECK:      %[[RET1:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP1]], #[[MAP0]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  // CHECK-SAME: ins(%[[ARG1]] : tensor<?xf32>)
  // CHECK-SAME: outs(%[[OUT1]] : tensor<10x?xf32>)
  %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2)
         {broadcast_dimensions = dense<[1]> : tensor<1xi64>}
       : (tensor<?xf32>, tensor<2xindex>) -> tensor<10x?xf32>

  // CHECK: return %[[RET0]], %[[RET1]]
  return %3, %4: tensor<10x?xf32>, tensor<10x?xf32>
}

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (0, d1, 0)>

// CHECK:       @optimize_3dx3d_bcast(
// CHECK-SAME:    %[[ARG0:[a-z0-9]+]]: tensor<?x1x?xf32>
// CHECK-SAME:    %[[ARG1:[a-z0-9]+]]: tensor<1x?x1xf32>
func @optimize_3dx3d_bcast(
  %arg0: tensor<?x1x?xf32>
    {cpurt.symbolic_shape = dense<[-2, 1, -3]> : tensor<3xi64>},
  %arg1: tensor<1x?x1xf32>
    {cpurt.symbolic_shape = dense<[1, -4, 1]> : tensor<3xi64>}
) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
  %0 = shape.shape_of %arg0 : tensor<?x1x?xf32> -> tensor<3xindex>
  %1 = shape.shape_of %arg1 : tensor<1x?x1xf32> -> tensor<3xindex>
  %2 = shape.broadcast %0, %1 : tensor<3xindex>, tensor<3xindex>
                             -> tensor<3xindex>

  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : index

  // CHECK:      %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK:      %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
  // CHECK:      %[[D2:.*]] = tensor.dim %[[ARG0]], %[[C2]]
  // CHECK:      %[[SHAPE:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]] : tensor<3xindex>
  // CHECK:      %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK:      %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK:      %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK:      %[[OUT0:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
  // CHECK:      %[[RET0:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
  // CHECK-SAME: ins(%[[ARG0]] : tensor<?x1x?xf32>)
  // CHECK-SAME: outs(%[[OUT0]] : tensor<?x?x?xf32>)
  %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2)
         {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
       : (tensor<?x1x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>

  // CHECK:      %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK:      %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK:      %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK:      %[[OUT1:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
  // CHECK:      %[[RET1:.*]] = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP2]], #[[MAP1]]]
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
  // CHECK-SAME: ins(%[[ARG1]] : tensor<1x?x1xf32>)
  // CHECK-SAME: outs(%[[OUT1]] : tensor<?x?x?xf32>)
  %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2)
         {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
       : (tensor<1x?x1xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>

  // CHECK: return %[[RET0]], %[[RET1]]
  return %3, %4: tensor<?x?x?xf32>, tensor<?x?x?xf32>
}
