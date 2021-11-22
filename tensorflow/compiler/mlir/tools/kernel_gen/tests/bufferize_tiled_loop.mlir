// RUN: kernel-gen-opt %s --computeop-and-func-bufferize \
// RUN: --tiled-loop-bufferize --cse --canonicalize --final-bufferize \
// RUN: --split-input-file | FileCheck %s

//      CHECK:  func @tiled_dot
func @tiled_dot(%A: tensor<10xf32>, %B: tensor<10xf32>,
                %C: tensor<f32>) -> tensor<f32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  %dot = linalg.tiled_loop (%i) = (%c0) to (%c10) step (%c2)
       ins (%A_ = %A: tensor<10xf32>, %B_ = %B: tensor<10xf32>)
       outs (%C_ = %C: tensor<f32>)
       iterators["reduction"] {
    %A_sub = tensor.extract_slice %A_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %B_sub = tensor.extract_slice %B_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %dot_sub = linalg.dot ins(%A_sub, %B_sub : tensor<?xf32>, tensor<?xf32>)
                          outs(%C_ : tensor<f32>) -> tensor<f32>
    linalg.yield %dot_sub : tensor<f32>
  }
  // CHECK: linalg.tiled_loop
  // CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>,
  // CHECK-SAME:      %[[B:arg[0-9]]] = %{{arg[0-9]}}: memref<10xf32>
  // CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: memref<f32>)

  // CHECK-NEXT: %[[SV_A:.*]] = memref.subview %[[A]]
  // CHECK-NEXT: %[[SV_B:.*]] = memref.subview %[[B]]
  // CHECK-NEXT: linalg.dot ins(%[[SV_A]], %[[SV_B]]
  // CHECK-SAME:            outs(%[[C]] : memref<f32>)
  // CHECK-NEXT: linalg.yield
  return %dot : tensor<f32>
}

// -----

#map0 = affine_map<(d0) -> (d0)>

func @tiled_add(%A: tensor<10xf32>, %B: tensor<10xf32>,
                  %C: tensor<10xf32>) -> tensor<10xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  %sum = linalg.tiled_loop (%i) = (%c0) to (%c10) step (%c2)
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
    linalg.yield %update : tensor<10xf32>
  }
  // CHECK: linalg.tiled_loop
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
  // CHECK:       linalg.yield
  return %sum : tensor<10xf32>
}

// -----

func @tiled_add_broadcast(%A: tensor<1x?x12xf32>, %B: tensor<?x?x12xf32>,
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
  %sum = linalg.tiled_loop (%i0, %i1, %i2) = (%c0, %c0, %c0) to (%d0, %d1, %c8)
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
    linalg.yield %v_out : tensor<?x?x12xf32>
  }
  // CHECK: memref.copy
  // CHECK: linalg.tiled_loop
  // CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{[0-9]+}}: memref<?x?x12xf32>)
  // CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: memref<?x?x12xf32>)
  return %sum : tensor<?x?x12xf32>
}
