// RUN: mlir-hlo-opt %s --hlo-one-shot-bufferize \
// RUN:  --cse --canonicalize --split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: @tensor.extract
// CHECK-SAME: (%[[ARG:.*]]: memref<?xf32>) -> f32
func.func @tensor.extract(%arg : tensor<?xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[ARG]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %c0 = arith.constant 0 : index
  %result = tensor.extract %arg[%c0] : tensor<?xf32>
  func.return %result : f32
}

// -----

// CHECK-LABEL: @tensor.from_elements
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
func.func @tensor.from_elements(%a : f32) -> f32 {
  // CHECK-DAG: %[[B:.*]] = arith.constant 1.2
  // CHECK-DAG: %[[C:.*]] = arith.constant 2.3
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[MEM:.*]] = memref.alloc
  // CHECK: store %[[A]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: store %[[B]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: store %[[C]], %[[MEM]][%[[C2]]] : memref<3xf32>
  %b = arith.constant 1.2 : f32
  %c = arith.constant 2.3 : f32
  %tfe = tensor.from_elements %a, %b, %c : tensor<3xf32>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<3xf32>
  func.return %result : f32
}

// -----

// CHECK-LABEL: @tensor.generate
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>) -> index
func.func @tensor.generate(%arg : tensor<*xf32>) -> index {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[SIZE:.*]] = memref.rank %[[ARG]] : memref<*xf32>
  // CHECK: %[[MEM:.*]] = memref.alloc
  // CHECK: scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[SIZE]]) step (%[[C1]]) {
  // CHECK:   %[[ELEM:.*]] = memref.dim %[[ARG]], %[[I]] : memref<*xf32>
  // CHECK:   memref.store %[[ELEM]], %[[MEM]][%[[I]]] : memref<?xindex>
  // CHECK:   scf.yield
  // CHECK: }
  %size = tensor.rank %arg : tensor<*xf32>
  %tfe = tensor.generate %size {
  ^bb0(%i : index):
    %elem = tensor.dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<?xindex>
  func.return %result : index
}

// -----

// CHECK: memref.global "private" constant @[[BUFFER:.*]] : memref<3xf32> = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]>
// CHECK-SAME: alignment = 64
// CHECK: @const
// CHECK-SAME: -> memref<3xf32>
func.func @const() -> tensor<3xf32> {
  // CHECK:  %[[RESULT:.*]] = memref.get_global @[[BUFFER]] : memref<3xf32>
  // CHECK:  return %[[RESULT]] : memref<3xf32>
  %result = arith.constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  func.return %result : tensor<3xf32>
}

// -----

// CHECK: memref.global "private" constant @[[BUFFER:.*]] : memref<3xf32> = dense<4.000000e+00>
// CHECK-SAME: alignment = 64
// CHECK: @const_splat
// CHECK-SAME: -> memref<3xf32>
func.func @const_splat() -> tensor<3xf32> {
  // CHECK:  %[[RESULT:.*]] = memref.get_global @[[BUFFER]] : memref<3xf32>
  // CHECK:  return %[[RESULT]] : memref<3xf32>
  %result = arith.constant dense<4.0> : tensor<3xf32>
  func.return %result : tensor<3xf32>
}

// -----

// CHECK-LABEL: @tensor_reshape
// CHECK-SAME: (%[[T:.*]]: memref<1x2x2xf32>)
func.func @tensor_reshape(%t : tensor<1x2x2xf32>) -> tensor<4xf32> {
  // CHECK: memref.collapse_shape %[[T]] {{.*}} : memref<1x2x2xf32> into memref<4xf32>
  %result = tensor.collapse_shape %t [[0, 1, 2]] : tensor<1x2x2xf32> into tensor<4xf32>
  func.return %result : tensor<4xf32>
}

// -----

// CHECK-LABEL: @slice
// CHECK-SAME: (%[[T:.*]]: memref<3xi32>)
func.func @slice(%t : tensor<3xi32>) -> tensor<1xi32> {
  // CHECK: memref.subview %[[T]][0] [1] [1] : memref<3xi32> to memref<1xi32>
  %result = tensor.extract_slice %t[0] [1] [1] : tensor<3xi32> to tensor<1xi32>
  func.return %result : tensor<1xi32>
}

// -----

func.func @dynamic_broadcast_return(%t : tensor<?x?xf32>, %shape : tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK: memref.copy
  %bcast = "mhlo.dynamic_broadcast_in_dim"(%t, %shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %bcast : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @arith_select
// CHECK-SAME: %[[C:.*]]: memref<i1>,
// CHECK-SAME: %[[LHS:.*]]: memref<1xf32>,
// CHECK-SAME: %[[RHS:.*]]: memref<1xf32>
func.func @arith_select(%c : tensor<i1>, %lhs: tensor<1xf32>, %rhs: tensor<1xf32>)
                  -> tensor<1xf32> {
  // CHECK: %[[COND:.*]] = memref.load %[[C]][]
  // CHECK: %[[RESULT:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]]
  // CHECK-SAME:             : memref<1xf32>
  %cond = tensor.extract %c[] : tensor<i1>
  %result = arith.select %cond, %lhs, %rhs : tensor<1xf32>
  func.return %result : tensor<1xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>
func.func @init_tensor_multiple_users(%lhs: tensor<10xf32>,
    %rhs: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %init = bufferization.alloc_tensor() : tensor<10xf32>
  %add = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%init : tensor<10xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %a = arith.addf %l, %r : f32
    linalg.yield %a : f32
  } -> tensor<10xf32>
  %sub = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%init : tensor<10xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %s = arith.subf %l, %r : f32
    linalg.yield %s : f32
  } -> tensor<10xf32>
  func.return %add, %sub : tensor<10xf32>, tensor<10xf32>
}
// CHECK-LABEL: func @init_tensor_multiple_users

// -----

// CHECK-LABEL:  func @tiled_dot
func.func @tiled_dot(%A: tensor<10xf32>, %B: tensor<10xf32>,
                %C: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  %dot = gml_st.loop (%i) = (%c0) to (%c10) step (%c2)
       ins (%A_ = %A: tensor<10xf32>, %B_ = %B: tensor<10xf32>)
       outs (%C_ = %C: tensor<f32>)
       iterators["reduction"] {
    %A_sub = tensor.extract_slice %A_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %B_sub = tensor.extract_slice %B_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %dot_sub = linalg.dot ins(%A_sub, %B_sub : tensor<?xf32>, tensor<?xf32>)
                          outs(%C_ : tensor<f32>) -> tensor<f32>
    gml_st.yield %dot_sub : tensor<f32>
  }
  // CHECK: gml_st.loop
  // CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>,
  // CHECK-SAME:      %[[B:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>
  // CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: memref<f32>)

  // CHECK-NEXT: %[[SV_A:.*]] = memref.subview %[[A]]
  // CHECK-NEXT: %[[SV_B:.*]] = memref.subview %[[B]]
  // CHECK-NEXT: linalg.dot ins(%[[SV_A]], %[[SV_B]]
  // CHECK-SAME:            outs(%[[C]] : memref<f32>)
  // CHECK-NEXT: memref.copy
  // CHECK-NEXT: gml_st.yield
  func.return %dot : tensor<f32>
}

// -----

#map0 = affine_map<(d0) -> (d0)>

func.func @tiled_add(%A: tensor<10xf32>, %B: tensor<10xf32>,
                  %C: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  %sum = gml_st.loop (%i) = (%c0) to (%c10) step (%c2)
       ins (%A_ = %A: tensor<10xf32>, %B_ = %B: tensor<10xf32>)
       outs (%C_ = %C: tensor<10xf32>) {
    %A_sub = tensor.extract_slice %A_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %B_sub = tensor.extract_slice %B_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %C_sub = tensor.extract_slice %C_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %sum_sub = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    } ins(%A_sub, %B_sub : tensor<?xf32>, tensor<?xf32>)
      outs(%C_sub : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
    } -> tensor<?xf32>
    %update = tensor.insert_slice %sum_sub into %C_[%i] [%c2] [1]
      : tensor<?xf32> into tensor<10xf32>
    gml_st.yield %update : tensor<10xf32>
  }
  // CHECK: gml_st.loop
  // CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>,
  // CHECK-SAME:      %[[B:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>
  // CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>)

  // CHECK-NEXT:  %[[SV_A:.*]] = memref.subview %[[A]]
  // CHECK-NEXT:  %[[SV_B:.*]] = memref.subview %[[B]]
  // CHECK-NEXT:  %[[SV_C:.*]] = memref.subview %[[C]]
  // CHECK-NEXT:  linalg.generic
  // CHECK-SAME:    ins(%[[SV_A]], %[[SV_B]]
  // CHECK-SAME:    outs(%[[SV_C]] : memref<2xf32, #map{{[0-9]}}>)
  // CHECK:         linalg.yield %{{[0-9]}} : f32
  // CHECK:       gml_st.yield
  func.return %sum : tensor<10xf32>
}

// -----

func.func @tiled_add_broadcast(%A: tensor<1x?x12xf32>, %B: tensor<?x?x12xf32>,
                          %shape: tensor<3xi32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %AA = "mhlo.dynamic_broadcast_in_dim"(%A, %shape)
    {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
    : (tensor<1x?x12xf32>, tensor<3xi32>) -> tensor<?x?x12xf32>

  %d0 = tensor.dim %AA, %c0 : tensor<?x?x12xf32>
  %d1 = tensor.dim %AA, %c1 : tensor<?x?x12xf32>
  %sum = gml_st.loop (%i0, %i1, %i2) = (%c0, %c0, %c0) to (%d0, %d1, %c8)
       step (%c1, %c1, %c8)
       ins (%A_ = %AA: tensor<?x?x12xf32>)
       outs (%B_ = %B: tensor<?x?x12xf32>) {
    %v_in = vector.transfer_read %A_[%i0, %i1, %i2], %cst
          {in_bounds = [true, true, true]}
          : tensor<?x?x12xf32>, vector<1x1x8xf32>
    %v_add = arith.addf %v_in, %v_in : vector<1x1x8xf32>
    %v_out = vector.transfer_write %v_add, %B_[%i0, %i1, %i2]
           {in_bounds = [true, true, true]}
           : vector<1x1x8xf32>, tensor<?x?x12xf32>
    gml_st.yield %v_out : tensor<?x?x12xf32>
  }
  // CHECK: gml_st.loop
  // CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{[0-9]+}}: memref<?x?x12xf32
  // CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: memref<?x?x12xf32>)
  // CHECK: memref.copy
  func.return %sum : tensor<?x?x12xf32>
}

// -----

#map0 = affine_map<()[s0] -> ((s0 floordiv 8) * 8)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
func.func @init_tensor_multiple_users(%arg0: tensor<1x?xf32>)
    -> (tensor<1x?xf32>, tensor<1x?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<1.000000e+00> : vector<1x8xf32>
  %cst_1 = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c1 : tensor<1x?xf32>
  %init = bufferization.alloc_tensor(%0) : tensor<1x?xf32>
  %2 = affine.apply #map0()[%0]
  %3 = gml_st.loop (%i, %j) = (%c0, %c0) to (%c1, %2) step (%c1, %c8)
      ins (%arg3 = %arg0: tensor<1x?xf32>)
      outs (%arg4 = %init: tensor<1x?xf32>) {
    %7 = vector.transfer_read %arg3[%i, %j], %cst {in_bounds = [true, true]}
      : tensor<1x?xf32>, vector<1x8xf32>
    %8 = arith.subf %cst_0, %7 : vector<1x8xf32>
    %9 = vector.transfer_write %8, %arg4[%i, %j] {in_bounds = [true, true]}
      : vector<1x8xf32>, tensor<1x?xf32>
    gml_st.yield %9 : tensor<1x?xf32>
  }
  %4 = gml_st.loop (%i, %j) = (%c0, %2) to (%c1, %0) step (%c1, %c8)
      ins (%arg3 = %arg0: tensor<1x?xf32>)
      outs (%arg4 = %3: tensor<1x?xf32>) {
    %7 = affine.apply #map1(%j)[%0]
    %8 = tensor.extract_slice %arg3[%i, %j] [1, %7] [1, 1]
      : tensor<1x?xf32> to tensor<1x?xf32>
    %9 = tensor.extract_slice %arg4[%i, %j] [1, %7] [1, 1]
      : tensor<1x?xf32> to tensor<1x?xf32>
    %10 = linalg.generic {
      indexing_maps = [#map2, #map2],
      iterator_types = ["parallel", "parallel"]}
      ins(%8 : tensor<1x?xf32>) outs(%9 : tensor<1x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):
      %12 = arith.subf %cst_1, %arg5 : f32
      linalg.yield %12 : f32
    } -> tensor<1x?xf32>
    %11 = tensor.insert_slice %10 into %arg4[%i, %j] [1, %7] [1, 1]
      : tensor<1x?xf32> into tensor<1x?xf32>
    gml_st.yield %11 : tensor<1x?xf32>
  }
  %5 = gml_st.loop (%i, %j) = (%c0, %c0) to (%c1, %2) step (%c1, %c8)
      ins (%arg3 = %arg0: tensor<1x?xf32>)
      outs (%arg4 = %init: tensor<1x?xf32>) {
    %7 = vector.transfer_read %arg3[%i, %j], %cst
      {in_bounds = [true, true]} : tensor<1x?xf32>, vector<1x8xf32>
    %8 = arith.subf %cst_0, %7 : vector<1x8xf32>
    %9 = arith.subf %cst_0, %8 : vector<1x8xf32>
    %10 = vector.transfer_write %9, %arg4[%i, %j]
      {in_bounds = [true, true]} : vector<1x8xf32>, tensor<1x?xf32>
    gml_st.yield %10 : tensor<1x?xf32>
  }
  %6 = gml_st.loop (%i, %j) = (%c0, %2) to (%c1, %0) step (%c1, %c8)
      ins (%arg3 = %arg0: tensor<1x?xf32>)
      outs (%arg4 = %5: tensor<1x?xf32>) {
    %7 = affine.apply #map1(%j)[%0]
    %8 = tensor.extract_slice %arg3[%i, %j] [1, %7] [1, 1]
      : tensor<1x?xf32> to tensor<1x?xf32>
    %9 = tensor.extract_slice %arg4[%i, %j] [1, %7] [1, 1]
      : tensor<1x?xf32> to tensor<1x?xf32>
    %10 = linalg.generic {
      indexing_maps = [#map2, #map2],
      iterator_types = ["parallel", "parallel"]}
      ins(%8 : tensor<1x?xf32>) outs(%9 : tensor<1x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):
      %12 = arith.subf %cst_1, %arg5 : f32
      %13 = arith.subf %cst_1, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<1x?xf32>
    %11 = tensor.insert_slice %10 into %arg4[%i, %j] [1, %7] [1, 1]
      : tensor<1x?xf32> into tensor<1x?xf32>
    gml_st.yield %11 : tensor<1x?xf32>
  }
  func.return %4, %6 : tensor<1x?xf32>, tensor<1x?xf32>
}
// CHECK-LABEL: init_tensor_multiple_users
// CHECK: %[[BUF2:.*]] = memref.alloc
// CHECK: %[[BUF1:.*]] = memref.alloc
// CHECK: gml_st.loop
// CHECK:   %[[BUF1]]
// CHECK: gml_st.loop
// CHECK:   %[[BUF1]]
// CHECK: gml_st.loop
// CHECK:   %[[BUF2]]
// CHECK: gml_st.loop
// CHECK:   %[[BUF2]]
// CHECK: return %[[BUF1]], %[[BUF2]]

// -----

// Test that scf ops are bufferized
// CHECK-LABEL:   func @if(
// CHECK-SAME:             %[[PRED:.*]]: i1,
// CHECK-SAME:             %[[TRUE_TENSOR:.*]]: memref<?xf32>,
// CHECK-SAME:             %[[FALSE_TENSOR:.*]]: memref<?xf32>) -> memref<?xf32> {
// CHECK:             %[[IF_RES:.*]] = scf.if %[[PRED]] -> (memref<?xf32, #map>) {
func.func @if(%pred: i1, %true_val: tensor<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
  %0 = scf.if %pred -> (tensor<?xf32>) {
    scf.yield %true_val : tensor<?xf32>
  } else {
    scf.yield %false_val : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}
