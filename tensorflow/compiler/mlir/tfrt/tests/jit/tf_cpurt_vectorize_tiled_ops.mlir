// RUN: tf-tfrt-opt %s --tf-cpurt-vectorize-tiled-ops --split-input-file |\
// RUN: FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
func @tiled_add(%A: tensor<8xf32>, %B: tensor<8xf32>,
                  %C: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %sum = linalg.tiled_loop (%i) = (%c0) to (%c8) step (%c2)
       ins (%A_ = %A: tensor<8xf32>, %B_ = %B: tensor<8xf32>)
       outs (%C_ = %C: tensor<8xf32>) {
    %A_sub = tensor.extract_slice %A_[%i] [2] [1]
      : tensor<8xf32> to tensor<2xf32>
    %B_sub = tensor.extract_slice %B_[%i] [2] [1]
      : tensor<8xf32> to tensor<2xf32>
    %C_sub = tensor.extract_slice %C_[%i] [2] [1]
      : tensor<8xf32> to tensor<2xf32>
    %sum_sub = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    } ins(%A_sub, %B_sub : tensor<2xf32>, tensor<2xf32>)
      outs(%C_sub : tensor<2xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
    } -> tensor<2xf32>
    %update = tensor.insert_slice %sum_sub into %C_[%i] [2] [1]
      : tensor<2xf32> into tensor<8xf32>
    linalg.yield %update : tensor<8xf32>
  }
  return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @tiled_add

// CHECK-DAG:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index

// CHECK: linalg.tiled_loop (%[[IV:arg[0-9]]]) =
// CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>,
// CHECK-SAME:      %[[B:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>
// CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>)

// CHECK-NEXT: %[[LHS:.*]] = vector.transfer_read %[[A]][%[[IV]]], %[[CST]]
// CHECK-SAME:   {in_bounds = [true]} : tensor<8xf32>, vector<2xf32>
// CHECK-NEXT: %[[RHS:.*]] = vector.transfer_read %[[B]][%[[IV]]], %[[CST]]
// CHECK-SAME:   {in_bounds = [true]} : tensor<8xf32>, vector<2xf32>

// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : vector<2xf32>

// CHECK-NEXT: %{{.*}} = vector.transfer_write %[[SUM]], %[[C]][%[[IV]]]
// CHECK-SAME:   {in_bounds = [true]} : vector<2xf32>, tensor<8xf32>

// -----

func @tiled_reduction_2d(%in: tensor<80x60xf32>) -> tensor<80xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c60 = arith.constant 60 : index
  %c80 = arith.constant 80 : index
  %cst = arith.constant 0.000000e+00 : f32

  %init = linalg.init_tensor [80] : tensor<80xf32>
  %out = linalg.fill(%cst, %init) : f32, tensor<80xf32> -> tensor<80xf32>

  %sum = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%c80, %c60) step (%c4, %c4)
          ins (%in_ = %in: tensor<80x60xf32>, %cst_ = %cst: f32)
          outs (%out_ = %out: tensor<80xf32>)
          iterators["parallel", "reduction"] {
    %in_sub = tensor.extract_slice %in_[%i, %j] [4, 4] [1, 1]
        : tensor<80x60xf32> to tensor<4x4xf32>
    %out_sub = tensor.extract_slice %out_[%i] [4] [1]
        : tensor<80xf32> to tensor<4xf32>
    %local_fill = linalg.fill(%cst_, %out_sub)
        : f32, tensor<4xf32> -> tensor<4xf32>
    %reduced_tile = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%in_sub : tensor<4x4xf32>)
        outs(%local_fill : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32):
        %0 = arith.addf %a, %b : f32
        linalg.yield %0 : f32
    } -> tensor<4xf32>

    %acc = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%reduced_tile : tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32):
        %1 = arith.addf %a, %b : f32
        linalg.yield %1 : f32
    } -> tensor<4xf32>
    %update = tensor.insert_slice %acc into %out_[%i] [4] [1]
        : tensor<4xf32> into tensor<80xf32>
    linalg.yield %update : tensor<80xf32>
  }
  return %sum : tensor<80xf32>
}

// CHECK-LABEL: func @tiled_reduction_2d

// CHECK: linalg.tiled_loop
// CHECK-SAME: ins (%{{arg[0-9]}} = %{{arg[0-9]}}: tensor<80x60xf32>,
// CHECK-SAME:      %[[CST:arg[0-9]]] = %{{.*}}: f32

// CHECK: %[[BCAST:.*]] = vector.broadcast %[[CST]] : f32 to vector<4xf32>
// CHECK-NOT: vector.transfer_write %[[BCAST]]
// CHECK: addf %{{.*}}, %[[BCAST]] : vector<4xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0) -> ()>
func @reduction_1d(%arg0: tensor<16xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.fill(%cst, %0) : f32, tensor<f32> -> tensor<f32>
  %2 = linalg.init_tensor [8] : tensor<8xf32>
  %3 = linalg.fill(%cst, %2) : f32, tensor<8xf32> -> tensor<8xf32>
  %4 = linalg.tiled_loop (%arg1) = (%c0) to (%c16) step (%c8)
      ins (%arg2 = %arg0: tensor<16xf32>)
      outs (%arg3 = %3: tensor<8xf32>)
      iterators["reduction"] {
    %6 = tensor.extract_slice %arg2[%arg1] [8] [1]
      : tensor<16xf32> to tensor<8xf32>
    %7 = tensor.expand_shape %6 [[0, 1]]
      : tensor<8xf32> into tensor<1x8xf32>
    %8 = linalg.generic {indexing_maps = [#map0, #map1],
                         iterator_types = ["reduction", "parallel"]}
                         ins(%7 : tensor<1x8xf32>)
                         outs(%arg3 : tensor<8xf32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      %9 = arith.addf %arg4, %arg5 : f32
      linalg.yield %9 : f32
    } -> tensor<8xf32>
    linalg.yield %8 : tensor<8xf32>
  }
  %5 = linalg.generic {indexing_maps = [#map2, #map3],
                       iterator_types = ["reduction"]}
                       ins(%4 : tensor<8xf32>)
                       outs(%1 : tensor<f32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %6 = arith.addf %arg1, %arg2 : f32
    linalg.yield %6 : f32
  } -> tensor<f32>
  return %5 : tensor<f32>
}
// CHECK-LABEL: func @reduction_1d

// CHECK: linalg.tiled_loop
// CHECK-SAME: ins (%[[IN:arg[0-9]]] = %{{arg[0-9]}}: tensor<16xf32>)

// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[IN]]
// CHECK: %[[VECTOR:.*]] = vector.transfer_read %[[SLICE]]
// CHECK: vector.shape_cast %[[VECTOR]] : vector<8xf32> to vector<1x8xf32>
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
