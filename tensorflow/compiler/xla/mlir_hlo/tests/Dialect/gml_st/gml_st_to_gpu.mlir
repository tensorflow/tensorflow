// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics \
// RUN:   --gml-st-to-gpu="warp-distribution-label=warp" -cse \
// RUN: | FileCheck %s
// We run CSE above to deduplicate constant definitions, which would confuse
// FileCheck.

#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

func.func @vectorized_tiling(%arg0: memref<2048xf32>) -> memref<2048xf32> {
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c0f = arith.constant 0.0 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
  %c1 = arith.constant 1 : index
  %map0 = affine.apply #map(%c2048)[%c0, %c1024]
  %map1 = affine.apply #map(%c1024)[%c0, %c128]
  %map2 = affine.apply #map(%c128)[%c0, %c4]
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %map0, %grid_y = %c1, %grid_z = %c1) threads(%tx, %ty, %tz) in (%block_x = %map2, %block_y = %map1, %block_z = %c1) {
    %apply_bx = affine.apply #map1(%bx)[%c0, %c1024]
    %block_arg = memref.subview %arg0[%apply_bx] [1024] [1] {"gml-st-distribution-label" = "block"} : memref<2048xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %block_out = memref.subview %alloc[%apply_bx] [1024] [1] {"gml-st-distribution-label" = "block"} : memref<2048xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %apply_ty = affine.apply #map1(%ty)[%c0, %c128]
    %warp_arg = memref.subview %block_arg[%apply_ty] [128] [1] {"gml-st-distribution-label" = "warp"} : memref<1024xf32, strided<[1], offset: ?>> to memref<128xf32, strided<[1], offset: ?>>
    %transfer_read = vector.transfer_read %warp_arg[%c0], %c0f {"gml-st-distribution-label" = "warp", in_bounds = [true]} : memref<128xf32, strided<[1], offset: ?>>, vector<128xf32>
    %warp_out = memref.subview %block_out[%apply_ty] [128] [1] {"gml-st-distribution-label" = "warp"} : memref<1024xf32, strided<[1], offset: ?>> to memref<128xf32, strided<[1], offset: ?>>
    %apply_tx = affine.apply #map1(%tx)[%c0, %c4]
    %materialized_tile = gml_st.materialize %transfer_read[%apply_tx] [4] [1]
      : vector<128xf32> to vector<4xf32>
    %result = math.absf %materialized_tile : vector<4xf32>
    %tile = gml_st.tile [%apply_tx] [4] [1] : !gml_st.tile<4>
    %distribute = gml_st.distribute %result into[%tile] : vector<4xf32> into vector<128xf32>[!gml_st.tile<4>]
    vector.transfer_write %distribute, %warp_out[%c0] {"gml-st-distribution-label" = "warp", in_bounds = [true]} : vector<128xf32>, memref<128xf32, strided<[1], offset: ?>>
    gpu.terminator
  }
  return %alloc : memref<2048xf32>
}

// CHECK-LABEL: @vectorized_tiling
// CHECK-SAME: %[[ARG:.*]]: memref
// CHECK:      %[[OUT:.*]] = memref.alloc
// CHECK:      gpu.launch {{.*}} threads(%[[TID:.*]], %{{.*}}, %{{.*}}) in
// CHECK-NOT:    gml_st.parallel
// CHECK-DAG:    %[[BARG:.*]] = memref.subview %[[ARG]]
// CHECK-DAG:    %[[BOUT:.*]] = memref.subview %[[OUT]]
// CHECK-NOT:    gml_st.parallel
// CHECK-DAG:    %[[WARG:.*]] = memref.subview %[[BARG]]
// CHECK-DAG:    %[[WOUT:.*]] = memref.subview %[[BOUT]]
// CHECK-NOT:    gml_st.parallel
// CHECK-DAG:    %[[OFS:.*]] = affine.apply {{.*}}(%[[TID]])
// CHECK-DAG:    %[[TARG:.*]] = memref.subview %[[WARG]][%[[OFS]]] [4] [1]
// CHECK-DAG:    %[[TVARG:.*]] = vector.transfer_read %[[TARG]][%c0]
// CHECK-SAME:      vector<4xf32>
// CHECK-DAG:    %[[TVOUT:.*]] = math.absf %[[TVARG]]
// CHECK-DAG:    %[[TOUT:.*]] = memref.subview %[[WOUT]][%[[OFS]]] [4] [1]
// CHECK-DAG:    vector.transfer_write %[[TVOUT]], %[[TOUT]][%c0]

// -----

func.func @materialize_scalar_of_transfer_read(
      %in: memref<32xf32>, %idx: index) -> f32 {
  %pad = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %vector = vector.transfer_read %in[%c0], %pad {in_bounds = [true]}
    : memref<32xf32>, vector<32xf32>
  %value = gml_st.materialize %vector[%idx] [1] [1]
    : vector<32xf32> to f32
  return %value : f32
}
// CHECK-LABEL: @materialize_scalar_of_transfer_read(
// CHECK-SAME: %[[IN:.*]]: memref<32xf32>, %[[IDX:.*]]: index
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[SUBVIEW:.*]] = memref.subview %[[IN]][%[[IDX]]]
// CHECK:      %[[VALUE:.*]] = memref.load %[[SUBVIEW]][%[[C0]]]
// CHECK:      return %[[VALUE]] : f32
