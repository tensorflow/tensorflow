// RUN: mlir-hlo-opt %s --computeop-and-func-bufferize \
// RUN: --gml-tiled-loop-bufferize --cse --canonicalize --final-bufferize \
// RUN: --split-input-file | FileCheck %s

//      CHECK:  func @tiled_dot
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
  // CHECK: memref.copy
  // CHECK: gml_st.loop
  // CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{[0-9]+}}: memref<?x?x12xf32>)
  // CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: memref<?x?x12xf32>)
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
  %init = linalg.init_tensor [1, %0] : tensor<1x?xf32>
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
// CHECK: %[[BUF1:.*]] = memref.alloc
// CHECK: gml_st.loop
// CHECK:   %[[BUF1]]
// CHECK: gml_st.loop
// CHECK:   %[[BUF1]]
// CHECK: %[[BUF2:.*]] = memref.alloc
// CHECK: gml_st.loop
// CHECK:   %[[BUF2]]
// CHECK: gml_st.loop
// CHECK:   %[[BUF2]]
// CHECK: return %[[BUF1]], %[[BUF2]]
