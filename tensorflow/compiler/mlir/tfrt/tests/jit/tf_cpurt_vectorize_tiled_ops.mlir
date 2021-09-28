// RUN: tf-tfrt-opt %s --tf-cpurt-vectorize-tiled-ops | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
func @tiled_add(%A: tensor<8xf32>, %B: tensor<8xf32>,
                  %C: tensor<8xf32>) -> tensor<8xf32> {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c8 = constant 8 : index
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
        %0 = std.addf %a, %b : f32
        linalg.yield %0 : f32
    } -> tensor<2xf32>
    %update = tensor.insert_slice %sum_sub into %C_[%i] [%c2] [1]
      : tensor<2xf32> into tensor<8xf32>
    linalg.yield %update : tensor<8xf32>
  }
  return %sum : tensor<8xf32>
}
// CHECK-LABEL: func @tiled_add

// CHECK-DAG:  %[[CST:.*]] = constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C0:.*]] = constant 0 : index

// CHECK: linalg.tiled_loop
// CHECK-SAME: ins (%[[A:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>,
// CHECK-SAME:      %[[B:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>
// CHECK-SAME: outs (%[[C:arg[0-9]]] = %{{arg[0-9]}}: tensor<8xf32>)

// CHECK-NEXT: %[[SUB_A:.*]] = tensor.extract_slice %[[A]]
// CHECK-NEXT: %[[SUB_B:.*]] = tensor.extract_slice %[[B]]
// CHECK-NEXT: %[[SUB_C:.*]] = tensor.extract_slice %[[C]]

// CHECK-NEXT: %[[LHS:.*]] = vector.transfer_read %1[%[[C0]]], %[[CST]]
// CHECK-SAME:   {in_bounds = [true]} : tensor<2xf32>, vector<2xf32>
// CHECK-NEXT: %[[RHS:.*]] = vector.transfer_read %2[%[[C0]]], %[[CST]]
// CHECK-SAME:   {in_bounds = [true]} : tensor<2xf32>, vector<2xf32>

// CHECK-NEXT: %[[SUM:.*]] = addf %[[LHS]], %[[RHS]] : vector<2xf32>

// CHECK-NEXT: %{{.*}} = vector.transfer_write %[[SUM]], %[[SUB_C]][%[[C0]]]
// CHECK-SAME:   {in_bounds = [true]} : vector<2xf32>, tensor<2xf32>

func @tiled_reduction(%in: tensor<80x60xf32>) -> tensor<80xf32> {
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c60 = constant 60 : index
  %c80 = constant 80 : index
  %cst = constant 0.000000e+00 : f32

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
        %0 = addf %a, %b : f32
        linalg.yield %0 : f32
    } -> tensor<4xf32>

    %acc = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%reduced_tile : tensor<4xf32>)
        outs(%out_sub : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32):
        %1 = addf %a, %b : f32
        linalg.yield %1 : f32
    } -> tensor<4xf32>
    %update = tensor.insert_slice %acc into %out_[%i] [4] [1]
        : tensor<4xf32> into tensor<80xf32>
    linalg.yield %update : tensor<80xf32>
  }
  return %sum : tensor<80xf32>
}

// CHECK-LABEL: func @tiled_reduction

// CHECK: linalg.tiled_loop
// CHECK-SAME: ins (%{{arg[0-9]}} = %{{arg[0-9]}}: tensor<80x60xf32>,
// CHECK-SAME:      %[[CST:arg[0-9]]] = %{{.*}}: f32

// CHECK: vector.broadcast %[[CST]] : f32 to vector<4xf32>
