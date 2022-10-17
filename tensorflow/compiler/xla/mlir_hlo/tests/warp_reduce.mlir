// RUN: mlir-hlo-opt --gml-st-to-gpu %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @generic_reduce
func.func @generic_reduce(%arg0 : memref<1xf32>) {

  %cst = arith.constant 1.0 : f32
  %bcast = vector.broadcast %cst : f32 to vector<1x32xf32>

  // CHECK: %[[LANE_ID:.*]] = gpu.lane_id
  // CHECK: %[[X0:.*]] = vector.extractelement {{.*}}%[[LANE_ID]]
  // CHECK: %[[Y0:.*]], %{{.*}} = gpu.shuffle xor %[[X0]], %c1
  // CHECK: %[[X1:.*]] = arith.addf %[[X0]], %[[Y0]]
  // CHECK: %[[Y1:.*]], %{{.*}} = gpu.shuffle xor %[[X1]], %c2
  // CHECK: %[[X2:.*]] = arith.addf %[[X1]], %[[Y1]]
  // CHECK: %[[Y2:.*]], %{{.*}} = gpu.shuffle xor %[[X2]], %c4
  // CHECK: %[[X3:.*]] = arith.addf %[[X2]], %[[Y2]]
  // CHECK: %[[Y3:.*]], %{{.*}} = gpu.shuffle xor %[[X3]], %c8
  // CHECK: %[[X4:.*]] = arith.addf %[[X3]], %[[Y3]]
  // CHECK: %[[Y4:.*]], %{{.*}} = gpu.shuffle xor %[[X4]], %c16
  // CHECK: %[[X5:.*]] = arith.addf %[[X4]], %[[Y4]]
  // CHECK: memref.store %[[X5]], %arg0[%c0] : memref<1xf32>
  linalg.generic {
    indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%bcast : vector<1x32xf32>) outs(%arg0: memref<1xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %add = arith.addf %lhs, %rhs : f32
    linalg.yield %add : f32
  }
  func.return
}

// CHECK-LABEL: func @vector_reduce
func.func @vector_reduce(%arg0 : memref<1xf32>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %init = vector.broadcast %cst : f32 to vector<1xf32>
  %bcast = vector.broadcast %cst : f32 to vector<1x32xf32>

  // CHECK: %[[LANE_ID:.*]] = gpu.lane_id
  // CHECK: %[[X0:.*]] = vector.extractelement {{.*}}%[[LANE_ID]]
  // CHECK: %[[Y0:.*]], %{{.*}} = gpu.shuffle xor %[[X0]], %c1
  // CHECK: %[[X1:.*]] = arith.addf %[[X0]], %[[Y0]]
  // CHECK: %[[Y1:.*]], %{{.*}} = gpu.shuffle xor %[[X1]], %c2
  // CHECK: %[[X2:.*]] = arith.addf %[[X1]], %[[Y1]]
  // CHECK: %[[Y2:.*]], %{{.*}} = gpu.shuffle xor %[[X2]], %c4
  // CHECK: %[[X3:.*]] = arith.addf %[[X2]], %[[Y2]]
  // CHECK: %[[Y3:.*]], %{{.*}} = gpu.shuffle xor %[[X3]], %c8
  // CHECK: %[[X4:.*]] = arith.addf %[[X3]], %[[Y3]]
  // CHECK: %[[Y4:.*]], %{{.*}} = gpu.shuffle xor %[[X4]], %c16
  // CHECK: %[[X5:.*]] = arith.addf %[[X4]], %[[Y4]]
  // CHECK: %[[ACC:.*]] = vector.extract {{.*}}[0]
  // CHECK: %[[Y5:.*]] = arith.addf %[[X5]], %[[ACC]]
  // CHECK: %[[SUM:.*]] = vector.broadcast %[[Y5]]
  %sum = vector.multi_reduction <add>, %bcast, %init [1] : vector<1x32xf32> to vector<1xf32>
  // CHECK: vector.transfer_write %[[SUM]], %arg0[%c0]
  vector.transfer_write %sum, %arg0[%c0] : vector<1xf32>, memref<1xf32>

  func.return
}

#stride1 = strided<[1], offset: ?>

// CHECK-LABEL: func @gpu_launch
func.func @gpu_launch() -> memref<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64xf32>
  // CHECK: gpu.launch
  gml_st.parallel (%arg1) = (%c0) to (%c64) step (%c4) {
    %1 = memref.subview %0[%arg1] [4] [1] : memref<64xf32> to memref<4xf32, #stride1>
    gml_st.parallel (%arg2) = (%c0) to (%c4) step (%c1) {
      %2 = memref.subview %1[%arg2] [1] [1] : memref<4xf32, #stride1> to memref<1xf32, #stride1>

      gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
        gml_st.set_yield
      }

      %3 = arith.constant dense<1.0> : vector<1x32xf32>
      // CHECK-NOT linalg.generic
      linalg.generic {
        indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]
      } ins(%3 : vector<1x32xf32>) outs(%2: memref<1xf32, #stride1>) {
      ^bb0(%lhs: f32, %rhs: f32):
        %add = arith.addf %lhs, %rhs : f32
        linalg.yield %add : f32
      }

      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return %0 : memref<64xf32>
}



