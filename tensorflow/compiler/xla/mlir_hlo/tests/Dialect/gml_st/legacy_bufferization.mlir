// RUN: mlir-hlo-opt %s -test-gml-st-bufferization --canonicalize -cse \
// RUN:   -split-input-file | FileCheck %s

func.func private @some_use(memref<?xf32>)

#TILE_MAP = affine_map<(d0)[s0] -> (3, -d0 + s0)>

//  CHECK-DAG: #[[$TILE_MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>

//      CHECK:  func @tiled_dot(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32>
// CHECK-SAME:    %[[c:[a-zA-Z0-9]*]]: memref<f32>
func.func @tiled_dot(%A: tensor<?xf32> {bufferization.writeable = false},
                %B: tensor<?xf32> {bufferization.writeable = false},
                %c: tensor<f32> {bufferization.writeable = true},
                %effecting: memref<?xf32>) -> tensor<f32> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: gml_st.loop {{.*}} to (%[[M]]) {{.*}} %[[A]]{{.*}}%[[B]]{{.*}}outs{{.*}}%[[c]]
  // CHECK-NOT: copy
  %1 = gml_st.loop (%arg3) = (%c0) to (%0) step (%c3)
       ins (%arg4 = %A: tensor<?xf32>, %use = %effecting : memref<?xf32>,
            %arg5 = %B: tensor<?xf32>)
       outs (%arg6 = %c: tensor<f32>)
       iterators["reduction"] {
    // CHECK-NOT:   alloc

    %2 = tensor.dim %arg4, %c0 : tensor<?xf32>
    %3 = affine.min #TILE_MAP(%arg3)[%2]

    //     CHECK:   %[[SV_A:.*]] = memref.subview {{.*}}
    %4 = tensor.extract_slice %arg4[%arg3] [%3] [1]
      : tensor<?xf32> to tensor<?xf32>
    %5 = tensor.dim %arg5, %c0 : tensor<?xf32>
    %6 = affine.min #TILE_MAP(%arg3)[%5]

    //     CHECK:   %[[SV_B:.*]] = memref.subview {{.*}}
    %7 = tensor.extract_slice %arg5[%arg3] [%6] [1] : tensor<?xf32> to tensor<?xf32>

    //     CHECK:   linalg.dot ins(%[[SV_A]], %[[SV_B]] : memref<?xf32, #map{{[0-9]}}>, memref<?xf32, #map{{[0-9]}}>) outs(%{{.*}} : memref<f32>)
    %8 = linalg.dot ins(%4, %7 : tensor<?xf32>, tensor<?xf32>)
                    outs(%arg6 : tensor<f32>) -> tensor<f32>

    //     CHECK:   call @some_use(%{{.*}}) : (memref<?xf32>) -> ()
    func.call @some_use(%use) : (memref<?xf32>) -> ()

    gml_st.yield %8 : tensor<f32>
    //     CHECK:   gml_st.yield
    // CHECK-NOT:   tensor
  }

  //     CHECK: return
  // CHECK-NOT: tensor
  func.return %1 : tensor<f32>
}

// -----

#TILE_MAP = affine_map<(d0)[s0] -> (3, -d0 + s0)>

//      CHECK:  func @tiled_fill(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32>
func.func @tiled_fill(%A: tensor<?xf32> {bufferization.writeable = true}) -> tensor<?xf32> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: gml_st.loop {{.*}} to (%[[M]]) {{.*}} outs{{.*}}%[[A]]
  %1 = gml_st.loop (%arg3) = (%c0) to (%0) step (%c3)
      outs (%arg1 = %A: tensor<?xf32>) iterators["parallel"] {
    // CHECK-NOT:   alloc

    %2 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %3 = affine.min #TILE_MAP(%arg3)[%2]

    //     CHECK:   %[[SV_A:.*]] = memref.subview {{.*}}
    %4 = tensor.extract_slice %arg1[%arg3] [%3] [1] : tensor<?xf32> to tensor<?xf32>

    //     CHECK:   linalg.fill ins(%{{.*}}: f32) outs(%[[SV_A]] : memref<?xf32, #map{{[0-9]}}>)
    %5 = linalg.fill ins(%f0: f32) outs(%4: tensor<?xf32>)
      -> tensor<?xf32>
    %6 = tensor.insert_slice %5 into %arg1[%arg3] [%3] [1] : tensor<?xf32> into tensor<?xf32>

    gml_st.yield %6 : tensor<?xf32>
    //     CHECK:   gml_st.yield
    // CHECK-NOT:   tensor
  }

  //     CHECK: return
  // CHECK-NOT: tensor
  func.return %1 : tensor<?xf32>
}

// -----

//      CHECK:  func @tiled_loop_yield_out_of_place(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]*]]: memref<?xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]*]]: memref<?xf32>
func.func @tiled_loop_yield_out_of_place(
    %A: tensor<?xf32> {bufferization.writeable = true},
    %B: tensor<?xf32> {bufferization.writeable = true}) -> tensor<?xf32> {
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  //     CHECK: %[[M:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32>
  %0 = tensor.dim %A, %c0 : tensor<?xf32>

  //     CHECK: gml_st.loop {{.*}} to (%[[M]]) {{.*}} outs{{.*}}%[[A]]
  %1 = gml_st.loop (%arg3) = (%c0) to (%0) step (%c3)
      outs (%arg1 = %A: tensor<?xf32>)
      iterators["parallel"]
  {
    // CHECK-NOT:   alloc
    //     CHECK:   memref.copy %[[B]], %[[A]]
    gml_st.yield %B : tensor<?xf32>
    //     CHECK:   gml_st.yield
    // CHECK-NOT:   tensor
  }

  //     CHECK: return
  // CHECK-NOT: tensor
  func.return %1 : tensor<?xf32>
}
