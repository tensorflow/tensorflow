// RUN: tf-tfrt-opt %s -allow-unregistered-dialect -split-input-file \
// RUN: -tf-cpurt-peel-tiled-loops -cse -canonicalize | FileCheck %s

#map0 = affine_map<(d0) -> (8, -d0 + 102401)>
#map1 = affine_map<(d0)[s0] -> (d0 + s0)>

func @tanh_1d(%arg0: memref<102401xf32>) -> memref<102401xf32> {
  %c102401 = arith.constant 102401 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<102401xf32>
  linalg.tiled_loop (%arg1) = (%c0) to (%c102401) step (%c8)
      ins (%arg2 = %arg0: memref<102401xf32>)
      outs (%arg3 = %0: memref<102401xf32>) {
    %1 = affine.min #map0(%arg1)
    %2 = memref.subview %arg2[%arg1] [%1] [1]
        : memref<102401xf32> to memref<?xf32, #map1>
    %3 = memref.subview %arg3[%arg1] [%1] [1]
        : memref<102401xf32> to memref<?xf32, #map1>
    %4 = vector.transfer_read %2[%c0], %cst
        : memref<?xf32, #map1>, vector<8xf32>
    %5 = math.tanh %4 : vector<8xf32>
    vector.transfer_write %5, %3[%c0] : vector<8xf32>, memref<?xf32, #map1>
    linalg.copy(%3, %3) : memref<?xf32, #map1>, memref<?xf32, #map1>
    linalg.yield
  }
  return %0 : memref<102401xf32>
}

// CHECK-DAG:  #[[$MAP:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// CHECK-LABEL: func @tanh_1d

// CHECK:       linalg.tiled_loop
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<8xf32, #[[$MAP]]>
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<8xf32, #[[$MAP]]>

// CHECK:       linalg.tiled_loop
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<?xf32, #[[$MAP]]>
// CHECK:           memref.subview
// CHECK-SAME:        memref<102401xf32> to memref<?xf32, #[[$MAP]]>

// -----

func @tanh_3d(%d0: index, %d1: index, %d2: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  linalg.tiled_loop (%arg1 ,%arg2, %arg3) = (%c0, %c0, %c0)
    to (%d0, %d1, %d2) step (%c8, %c1, %c8)
    ins () outs () {
    "prevent.dce"() : () -> ()
    linalg.yield
  }
  return
}

// CHECK-LABEL: func @tanh_3d(
// CHECK-SAME:    %[[D0:[a-z0-9]+]]: index, %[[D1:[a-z0-9]+]]: index,
// CHECK-SAME:    %[[D2:[a-z0-9]+]]: index) {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index

// CHECK-DAG:     %[[SPLIT0:.*]] = affine.apply{{.*}}%[[D0]]
// CHECK-DAG:     %[[SPLIT2:.*]] = affine.apply{{.*}}%[[D2]]

// CHECK:     linalg.tiled_loop{{.*}}(%[[C0]], %[[C0]], %[[C0]])
// CHECK-SAME:  to (%[[SPLIT0]], %arg1, %[[SPLIT2]])
// CHECK-SAME:  step  (%[[C8]], %[[C1]], %[[C8]])

// CHECK:     linalg.tiled_loop{{.*}}(%[[SPLIT0]], %[[C0]], %[[C0]])
// CHECK-SAME:  to (%arg0, %arg1, %[[SPLIT2]])
// CHECK-SAME:  step  (%[[C8]], %[[C1]], %[[C8]])

// CHECK:     linalg.tiled_loop{{.*}}(%[[C0]], %[[C0]], %[[SPLIT2]])
// CHECK-SAME:  to (%arg0, %arg1, %arg2)
// CHECK-SAME:  step  (%[[C8]], %[[C1]], %[[C8]])

// -----

func @reduce_column_sum_2d_dynamic(%in: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index

  %dim_X = tensor.dim %in, %c0 : tensor<?x?xf32>
  %dim_Y = tensor.dim %in, %c1 : tensor<?x?xf32>

  %1 = linalg.init_tensor [%dim_Y] : tensor<?xf32>
  %2 = linalg.fill(%cst, %1) : f32, tensor<?xf32> -> tensor<?xf32>
  %5 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%dim_Y, %dim_X)
         step (%c4, %c4)
         ins (%in_ = %in: tensor<?x?xf32>, %cst_ = %cst: f32)
         outs (%out_ = %2: tensor<?xf32>)
         iterators["parallel", "reduction"] {
    %6 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%j)[%dim_X]
    %9 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%i)[%dim_Y]

    %8 = tensor.extract_slice %in_[%j, %i] [%6, %9] [1, 1]
           : tensor<?x?xf32> to tensor<?x?xf32>
    %11 = tensor.extract_slice %out_[%i] [%9] [1]
           : tensor<?xf32> to tensor<?xf32>

    %12 = linalg.fill(%cst_, %11) : f32, tensor<?xf32> -> tensor<?xf32>
    %13 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                             affine_map<(d0, d1) -> (d0)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%8 : tensor<?x?xf32>)
            outs(%12 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %14 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>,
                             affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%13 : tensor<?xf32>)
            outs(%11 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %15 = tensor.insert_slice %14 into %out_[%i] [%9] [1]
            : tensor<?xf32> into tensor<?xf32>
    linalg.yield %15 : tensor<?xf32>
  }
  return %5 : tensor<?xf32>
}

// CHECK-LABEL: func @reduce_column_sum_2d_dynamic

// CHECK:       linalg.fill
// CHECK:       linalg.tiled_loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<4x4xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<4xf32>

// CHECK:       linalg.tiled_loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<4x?xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<?xf32>

// CHECK:       linalg.tiled_loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<?xf32>

// -----


func @reduce_row_sum_2d_dynamic(%in: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  %dim_X = tensor.dim %in, %c0 : tensor<?x?xf32>
  %dim_Y = tensor.dim %in, %c1 : tensor<?x?xf32>

  %1 = linalg.init_tensor [%dim_X] : tensor<?xf32>
  %2 = linalg.fill(%cst, %1) : f32, tensor<?xf32> -> tensor<?xf32>
  %5 = linalg.tiled_loop (%i, %j) = (%c0, %c0) to (%dim_X, %dim_Y)
    step (%c4, %c4)
    ins (%in_ = %in: tensor<?x?xf32>, %cst_ = %cst: f32)
    outs (%out_ = %2: tensor<?xf32>)
    iterators["parallel", "reduction"] {
    %6 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%i)[%dim_X]
    %7 = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%j)[%dim_Y]

    %8 = tensor.extract_slice %in_[%i, %j] [%6, %7] [1, 1]
           : tensor<?x?xf32> to tensor<?x?xf32>
    %11 = tensor.extract_slice %out_[%i] [%6] [1]
           : tensor<?xf32> to tensor<?xf32>
    %12 = linalg.fill(%cst_, %11) : f32, tensor<?xf32> -> tensor<?xf32>
    %13 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0)>],
            iterator_types = ["parallel", "reduction"]}
            ins(%8 : tensor<?x?xf32>)
            outs(%12 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %14 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>,
                             affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%13 : tensor<?xf32>)
            outs(%11 : tensor<?xf32>) {
          ^bb0(%arg6: f32, %arg7: f32):
            %16 = arith.addf %arg6, %arg7 : f32
            linalg.yield %16 : f32
          } -> tensor<?xf32>
    %15 = tensor.insert_slice %14 into %out_[%i] [%6] [1]
            : tensor<?xf32> into tensor<?xf32>
    linalg.yield %15 : tensor<?xf32>
  }
  return %5 : tensor<?xf32>
}

// CHECK-LABEL: func @reduce_row_sum_2d_dynamic

// CHECK:       linalg.fill
// CHECK:       linalg.tiled_loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<4x4xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<4xf32>

// CHECK:       linalg.tiled_loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<?x4xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<?xf32>

// CHECK:       linalg.tiled_loop
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:           tensor.extract_slice
// CHECK-SAME:        tensor<?xf32> to tensor<?xf32>
