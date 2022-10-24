// RUN: mlir-hlo-opt -split-input-file -gml-st-to-gpu %s | FileCheck %s

// CHECK-LABEL: func @vector_reduce
func.func @vector_reduce(%arg0 : memref<1xf32>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %init = vector.broadcast %cst : f32 to vector<1xf32>
  %lane = gpu.lane_id
  %tile = gml_st.tile [%lane] [1] [1] : !gml_st.tile<1>
  %bcast = vector.broadcast %cst : f32 to vector<1xf32>

  // CHECK: %[[CST:.*]] = arith.constant 1.0
  // CHECK: %[[Y0:.*]], %{{.*}} = gpu.shuffle xor %[[CST]], %c1
  // CHECK: %[[X1:.*]] = arith.addf %[[Y0]], %[[CST]]
  // CHECK: %[[Y1:.*]], %{{.*}} = gpu.shuffle xor %[[X1]], %c2
  // CHECK: %[[X2:.*]] = arith.addf %[[X1]], %[[Y1]]
  // CHECK: %[[Y2:.*]], %{{.*}} = gpu.shuffle xor %[[X2]], %c4
  // CHECK: %[[X3:.*]] = arith.addf %[[X2]], %[[Y2]]
  // CHECK: %[[Y3:.*]], %{{.*}} = gpu.shuffle xor %[[X3]], %c8
  // CHECK: %[[X4:.*]] = arith.addf %[[X3]], %[[Y3]]
  // CHECK: %[[Y4:.*]], %{{.*}} = gpu.shuffle xor %[[X4]], %c16
  // CHECK: %[[X5:.*]] = arith.addf %[[X4]], %[[Y4]]
  // CHECK: %[[Y5:.*]] = arith.addf %[[X5]], %[[CST]]
  // CHECK: %[[SUM:.*]] = vector.broadcast %[[Y5]]
  %dist = gml_st.distribute %bcast into[%tile]
    : vector<1xf32> into vector<1x32xf32>[!gml_st.tile<1>]
  %sum = vector.multi_reduction <add>, %dist, %init [1]
    : vector<1x32xf32> to vector<1xf32>
  // CHECK: vector.transfer_write %[[SUM]], %arg0[%c0]
  vector.transfer_write %sum, %arg0[%c0] : vector<1xf32>, memref<1xf32>

  func.return
}

// -----

#stride1 = strided<[1], offset: ?>

// CHECK-LABEL: func @gpu_launch
func.func @gpu_launch() -> memref<64xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant dense<0.0> : vector<1xf32>
  %0 = memref.alloc() : memref<64xf32>
  // CHECK: gpu.launch
  gml_st.parallel (%arg1) = (%c0) to (%c64) step (%c4) {
    %1 = memref.subview %0[%arg1] [4] [1]
      : memref<64xf32> to memref<4xf32, #stride1>
    gml_st.parallel (%arg2) = (%c0) to (%c4) step (%c1) {
      %2 = memref.subview %1[%arg2] [1] [1]
        : memref<4xf32, #stride1> to memref<1xf32, #stride1>

      %init = vector.broadcast %cst : vector<1xf32> to vector<1x32xf32>
      %3 = gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
        %tile = gml_st.tile [0, %arg3] [1, 1] [1, 1] : !gml_st.tile<1x1>
        %elem = arith.constant dense<1.0> : vector<1x1xf32>
        gml_st.set_yield %elem into %init[%tile]
          : vector<1x1xf32> into vector<1x32xf32>[!gml_st.tile<1x1>]
      } : vector<1x32xf32>

      // CHECK-NOT: vector.multi_reduction
      %sum = vector.multi_reduction <add>, %3, %cst [1]
        : vector<1x32xf32> to vector<1xf32>
      vector.transfer_write %sum, %2[%c0] {in_bounds = [true]}
        : vector<1xf32>, memref<1xf32, #stride1>
      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return %0 : memref<64xf32>
}
