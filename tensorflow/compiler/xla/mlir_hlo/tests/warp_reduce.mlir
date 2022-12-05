// RUN: mlir-hlo-opt -split-input-file %s \
// RUN:   -gml-st-simtfy="block-distribution-label=block" \
// RUN:   -gml-st-to-gpu="warp-distribution-label=warp" \
// RUN: | FileCheck %s

// CHECK-LABEL: func @vector_reduce
func.func @vector_reduce(
  %arg0: vector<1xf32>,
  %arg1: vector<1xf32>
) -> vector<1xf32> {

  %lane = gpu.lane_id
  %tile = gml_st.tile [%lane] [1] [1] : !gml_st.tile<1>
  %dist = gml_st.distribute %arg1 into[%tile]
    : vector<1xf32> into vector<1x32xf32>[!gml_st.tile<1>]

  // CHECK: %[[X0:.*]] = vector.extract %arg1[0]
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
  // CHECK: %[[Y5:.*]] = vector.extract %arg0[0]
  // CHECK: %[[X6:.*]] = arith.addf %[[Y5]], %[[X5]]
  // CHECK: %[[RESULT:.*]] = vector.broadcast %[[X6]]
  %result = vector.multi_reduction <add>, %dist, %arg0
    {"gml-st-distribution-label" = "warp"} [1]
    : vector<1x32xf32> to vector<1xf32>

  // CHECK: return %[[RESULT]]
  func.return %result : vector<1xf32>
}

// -----

// CHECK-LABEL: func @vector_reduce_small
func.func @vector_reduce_small(
  %arg0: vector<1xf32>,
  %arg1: vector<1xf32>
) -> vector<1xf32> {

  %lane = gpu.lane_id
  %tile = gml_st.tile [%lane] [1] [1] : !gml_st.tile<1>
  %dist = gml_st.distribute %arg1 into[%tile]
    : vector<1xf32> into vector<1x4xf32>[!gml_st.tile<1>]

  // CHECK: %[[X0:.*]] = vector.extract %arg1[0]
  // CHECK: %[[Y0:.*]], %{{.*}} = gpu.shuffle xor %[[X0]], %c1
  // CHECK: %[[X1:.*]] = arith.addf %[[X0]], %[[Y0]]
  // CHECK: %[[Y1:.*]], %{{.*}} = gpu.shuffle xor %[[X1]], %c2
  // CHECK: %[[X2:.*]] = arith.addf %[[X1]], %[[Y1]]
  // CHECK: %[[Y2:.*]] = vector.extract %arg0[0]
  // CHECK: %[[X3:.*]] = arith.addf %[[Y2]], %[[X2]]
  // CHECK: %[[RESULT:.*]] = vector.broadcast %[[X3]]
  %result = vector.multi_reduction <add>, %dist, %arg0
    {"gml-st-distribution-label" = "warp"} [1]
    : vector<1x4xf32> to vector<1xf32>

  // CHECK: return %[[RESULT]]
  func.return %result : vector<1xf32>
}

// -----

// CHECK-LABEL: func @vector_reduce_fp16
func.func @vector_reduce_fp16(
  %arg0: vector<1xf16>,
  %arg1: vector<1xf16>
) -> vector<1xf16> {

  %lane = gpu.lane_id
  %tile = gml_st.tile [%lane] [1] [1] : !gml_st.tile<1>
  %dist = gml_st.distribute %arg1 into[%tile]
    : vector<1xf16> into vector<1x2xf16>[!gml_st.tile<1>]

  // CHECK: %[[X0:.*]] = vector.extract %arg1[0]
  // CHECK: %[[A0:.*]] = arith.bitcast %[[X0]]
  // CHECK: %[[B0:.*]] = arith.extui %[[A0]]
  // CHECK: %[[C0:.*]], %{{.*}} = gpu.shuffle xor %[[B0]], %c1
  // CHECK: %[[D0:.*]] = arith.trunci %[[C0]]
  // CHECK: %[[Y0:.*]] = arith.bitcast %[[D0]]
  // CHECK: %[[X1:.*]] = arith.maxf %[[X0]], %[[Y0]]
  // CHECK: %[[Y1:.*]] = vector.extract %arg0[0]
  // CHECK: %[[X2:.*]] = arith.maxf %[[Y1]], %[[X1]]
  // CHECK: %[[RESULT:.*]] = vector.broadcast %[[X2]]
  %result = vector.multi_reduction <maxf>, %dist, %arg0
    {"gml-st-distribution-label" = "warp"} [1]
    : vector<1x2xf16> to vector<1xf16>

  // CHECK: return %[[RESULT]]
  func.return %result : vector<1xf16>
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
  gml_st.parallel (%arg1) = (%c0) to (%c64) step (%c4) distribution ("block") {
    %1 = memref.subview %0[%arg1] [4] [1]
      : memref<64xf32> to memref<4xf32, #stride1>
    gml_st.parallel (%arg2) = (%c0) to (%c4) step (%c1) distribution ("warp") {
      %2 = memref.subview %1[%arg2] [1] [1]
        : memref<4xf32, #stride1> to memref<1xf32, #stride1>

      %init = vector.broadcast %cst : vector<1xf32> to vector<1x32xf32>
      %3 = gml_st.parallel (%arg3) = (%c0) to (%c32) step (%c1)
          distribution ("thread") {
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

// -----

func.func @transform_only_warp_level_multi_reduction(%in: vector<4x10xi32>)
    -> i32 {
  %acc = arith.constant 0 : i32
  %result = vector.multi_reduction <add>, %in, %acc
    {"gml-st-distribution-level" = "not-warp"} [0, 1] : vector<4x10xi32> to i32
  func.return %result : i32
}

// CHECK-LABEL: @transform_only_warp_level_multi_reduction
// CHECK: vector.multi_reduction <add>, %[[IN:.*]], %[[ACC:.*]]
// CHECK-SAME {"gml-st-distribution-level" = "not-warp"} [0, 1]
