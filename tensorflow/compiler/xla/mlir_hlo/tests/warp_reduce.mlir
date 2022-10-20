// RUN: mlir-hlo-opt --gml-st-to-gpu %s | FileCheck %s

// CHECK-LABEL: func @vector_reduce
func.func @vector_reduce(%arg0 : memref<1xf32>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %init = vector.broadcast %cst : f32 to vector<1xf32>
  %bcast = vector.broadcast %cst : f32 to vector<1x32xf32>

  // CHECK: %[[CST:.*]] = arith.constant 1.0
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
  // CHECK: %[[Y5:.*]] = arith.addf %[[X5]], %[[CST]]
  // CHECK: %[[SUM:.*]] = vector.broadcast %[[Y5]]
  %sum = vector.multi_reduction <add>, %bcast, %init [1] : vector<1x32xf32> to vector<1xf32>
  // CHECK: vector.transfer_write %[[SUM]], %arg0[%c0]
  vector.transfer_write %sum, %arg0[%c0] : vector<1xf32>, memref<1xf32>

  func.return
}
