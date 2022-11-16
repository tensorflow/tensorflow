// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics \
// RUN:   --gml-st-simtfy="block-distribution-label=block" --gml-st-to-gpu="warp-distribution-label=warp" -cse \
// RUN: | FileCheck %s
// We run CSE above to deduplicate constant definitions, which would confuse
// FileCheck.

#map = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @simple(%arg2: memref<2048xf32>) -> memref<2048xf32> {
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %2 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
  gml_st.parallel (%arg3) = (%c0) to (%c2048) step (%c128) distribution ("block") {
    %3 = memref.subview %2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    %4 = memref.subview %arg2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    gml_st.parallel (%arg4) = (%c0) to (%c128) step (%c32) distribution ("warp") {
      %5 = memref.subview %3[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      %6 = memref.subview %4[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      gml_st.parallel (%arg5) = (%c0) to (%c32) step (%c1) distribution ("thread") {
        %7 = memref.load %6[%arg5] : memref<32xf32, #map>
        %8 = math.log %7 : f32
        memref.store %8, %5[%arg5] : memref<32xf32, #map>
        gml_st.set_yield
      }
      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return %2 : memref<2048xf32>
}

// CHECK-LABEL: @simple
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
// CHECK:       gpu.launch blocks
// CHECK-SAME:  ({{.*}}) in ({{.*}} = %[[C16]], {{.*}} = %[[C1]], {{.*}} = %[[C1]]) threads
// CHECK-SAME:  ({{.*}}) in ({{.*}} = %[[C32]], {{.*}} = %[[C4]], {{.*}} = %[[C1]])
// CHECK:       affine.apply {{.*}}[%[[C0]], %[[C128]]]
// CHECK-NEXT:  memref.subview
// CHECK-SAME:  "gml-st-distribution-label" = "block"
// CHECK:       affine.apply {{.*}}[%[[C0]], %[[C32]]]
// CHECK-NEXT:  memref.subview
// CHECK-SAME:  "gml-st-distribution-label" = "warp"
// CHECK:       affine.apply {{.*}}[%[[C0]], %[[C1]]]
// CHECK-NOT:   scf.if
// CHECK:       memref.load
// CHECK-NOT:   "gml-st-distribution-label"
// CHECK:       math.log
// CHECK:       memref.store

// -----

#map = affine_map<(d0)[s0] -> (d0 + s0)>

func.func @sibling_parallels(%arg2: memref<2048xf32>) -> memref<2048xf32> {
  %c0 = arith.constant 0 : index
  %c2048 = arith.constant 2048 : index
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %2 = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
  gml_st.parallel (%arg3) = (%c0) to (%c2048) step (%c128) distribution ("block") {
    %3 = memref.subview %2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    %4 = memref.subview %arg2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    gml_st.parallel (%arg4) = (%c0) to (%c128) step (%c32) distribution ("warp") {
      %5 = memref.subview %3[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      %6 = memref.subview %4[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      %7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
      gml_st.parallel (%arg5) = (%c0) to (%c32) step (%c1) distribution ("thread") {
        %8 = memref.load %6[%arg5] : memref<32xf32, #map>
        %9 = math.log %8 : f32
        memref.store %9, %7[%arg5] : memref<32xf32>
        gml_st.set_yield
      }
      gml_st.parallel (%arg6) = (%c0) to (%c32) step (%c1) distribution ("thread") {
        %10 = memref.load %7[%arg6] : memref<32xf32>
        %11 = math.absf %10 : f32
        memref.store %11, %5[%arg6] : memref<32xf32, #map>
        gml_st.set_yield
      }
      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return %2 : memref<2048xf32>
}

// CHECK-LABEL: @sibling_parallels
// CHECK:       gpu.launch blocks
// CHECK:       affine.apply
// CHECK:       affine.apply
// CHECK:       affine.apply
// CHECK:       memref.load
// CHECK:       math.log
// CHECK:       memref.store
// CHECK-NOT:   affine.apply
// CHECK:       memref.load
// CHECK:       math.absf
// CHECK:       memref.store

// -----

func.func @too_deep_nesting() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<index>
  // expected-error@+1 {{failed to simtfy}}
  gml_st.parallel (%arg3) = (%c0) to (%c1) step (%c1) distribution ("block") {
    gml_st.parallel (%arg4) = (%c0) to (%c1) step (%c1) distribution ("warp") {
      gml_st.parallel (%arg5) = (%c0) to (%c1) step (%c1) distribution ("thread") {
        gml_st.parallel (%arg6) = (%c0) to (%c1) step (%c1) {
          memref.store %c0, %alloc[] : memref<index>
          gml_st.set_yield
        }
        gml_st.set_yield
      }
      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return
}


// -----

func.func @mismatched_bounds() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloc1 = memref.alloc() : memref<index>
  %alloc2 = memref.alloc() : memref<index>
  // expected-error@+1 {{failed to simtfy}}
  gml_st.parallel (%arg3) = (%c0) to (%c1) step (%c1) distribution ("block") {
    gml_st.parallel (%arg4) = (%c0) to (%c1) step (%c1) distribution ("warp") {
      gml_st.parallel (%arg5) = (%c0) to (%c1) step (%c1) distribution ("thread") {
        memref.store %c0, %alloc1[] : memref<index>
        gml_st.set_yield
      }
      gml_st.parallel (%arg6) = (%c0) to (%c2) step (%c1) distribution ("thread") {
        memref.store %c0, %alloc2[] : memref<index>
        gml_st.set_yield
      }
      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return
}

// -----

func.func @mmultple_induction_vars() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %alloc = memref.alloc() : memref<index>
  // expected-error@+1 {{failed to simtfy}}
  gml_st.parallel (%arg1, %arg2) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) distribution ("block") {
    memref.store %c0, %alloc[] : memref<index>
    gml_st.set_yield
  }
  return
}

// -----

#layout = strided<[1], offset: ?>

func.func @imperfect_tiling(%arg0: memref<2051xf32>) -> memref<2051xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c2051 = arith.constant 2051 : index
  %0 = memref.alloc() {alignment = 64 : i64} : memref<2051xf32>
  gml_st.parallel (%arg1) = (%c0) to (%c2051) step (%c128) distribution ("block") {
    %1 = affine.min affine_map<(d0) -> (-d0 + 2051, 128)>(%arg1)
    %2 = memref.subview %arg0[%arg1] [%1] [1] : memref<2051xf32> to memref<?xf32, #layout>
    %3 = memref.subview %0[%arg1] [%1] [1] : memref<2051xf32> to memref<?xf32, #layout>
    gml_st.parallel (%arg2) = (%c0) to (%1) step (%c32) distribution ("warp") {
      %4 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg2)[%1]
      %5 = memref.subview %2[%arg2] [%4] [1] : memref<?xf32, #layout> to memref<?xf32, #layout>
      %6 = memref.subview %3[%arg2] [%4] [1] : memref<?xf32, #layout> to memref<?xf32, #layout>
      gml_st.parallel (%arg3) = (%c0) to (%4) step (%c1) distribution ("thread") {
        %7 = memref.load %5[%arg3] : memref<?xf32, #layout>
        %8 = math.log %7 : f32
        memref.store %8, %6[%arg3] : memref<?xf32, #layout>
        gml_st.set_yield
      }
      gml_st.set_yield
    }
    gml_st.set_yield
  }
  return %0 : memref<2051xf32>
}

// CHECK-LABEL: @imperfect_tiling
// CHECK:       gpu.launch blocks(%[[BLOCKID:.*]], %{{.*}}, %{{.*}}) in {{.*}} threads
// CHECK-SAME:             (%[[THREADID:.*]], %[[WARPID:.*]], %{{.*}}) in
// CHECK-DAG:   %[[ARG1:.*]] = affine.apply {{.*}}(%[[BLOCKID]])
// CHECK-DAG:   %[[BTILESIZE:.*]] = affine.min {{.*}}(%[[ARG1]])
// CHECK-DAG:   %[[ARG2:.*]] = affine.apply {{.*}}(%[[WARPID]])
// CHECK-DAG:   %[[WCOND:.*]] = arith.cmpi slt, %[[ARG2:.*]], %[[BTILESIZE]]
// CHECK-DAG:   scf.if %[[WCOND]]
// CHECK-DAG:   %[[WTILESIZE:.*]] = affine.min {{.*}}(%[[ARG2]])[%[[BTILESIZE]]]
// CHECK-DAG:   %[[ARG3:.*]] = affine.apply {{.*}}(%[[THREADID]])
// CHECK-DAG:   %[[TCOND:.*]] = arith.cmpi slt, %[[ARG3:.*]], %[[WTILESIZE]]
// CHECK-DAG:   scf.if %[[TCOND]]
// CHECK:       memref.load
// CHECK:       math.log
// CHECK:       memref.store

// -----

#layout = strided<[1], offset: ?>

func.func @vectorized_tiling(%arg0: memref<2048xf32>) -> memref<2048xf32> {
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048xf32>
  gml_st.parallel (%arg1) = (%c0) to (%c2048) step (%c1024) distribution ("block") {
    %subview = memref.subview %arg0[%arg1] [1024] [1]
      : memref<2048xf32> to memref<1024xf32, #layout>
    %subview_0 = memref.subview %alloc[%arg1] [1024] [1]
      : memref<2048xf32> to memref<1024xf32, #layout>
    gml_st.parallel (%arg2) = (%c0) to (%c1024) step (%c128) distribution ("warp") {
      %subview_1 = memref.subview %subview[%arg2] [128] [1]
        : memref<1024xf32, #layout> to memref<128xf32, #layout>
      %0 = vector.transfer_read %subview_1[%c0], %cst {in_bounds = [true]}
        : memref<128xf32, #layout>, vector<128xf32>
      %subview_2 = memref.subview %subview_0[%arg2] [128] [1]
        : memref<1024xf32, #layout> to memref<128xf32, #layout>
      %1 = vector.transfer_read %subview_2[%c0], %cst {in_bounds = [true]}
        : memref<128xf32, #layout>, vector<128xf32>
      %2 = gml_st.parallel (%arg3) = (%c0) to (%c128) step (%c4) distribution ("thread") {
        %4 = gml_st.tile [%arg3] [4] [1] : !gml_st.tile<4>
        %5 = gml_st.materialize %0[%4]
          : vector<128xf32>[!gml_st.tile<4>] to vector<4xf32>
        %6 = math.absf %5 : vector<4xf32>
        gml_st.set_yield %6 into %1[%4]
          : vector<4xf32> into vector<128xf32>[!gml_st.tile<4>]
      } : vector<128xf32>
      vector.transfer_write %2, %subview_2[%c0] {in_bounds = [true]}
        : vector<128xf32>, memref<128xf32, #layout>
      gml_st.set_yield
    }
    gml_st.set_yield
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
  %tile = gml_st.tile [%idx] [1] [1] : !gml_st.tile<1>
  %value = gml_st.materialize %vector[%tile]
    : vector<32xf32>[!gml_st.tile<1>] to f32
  return %value : f32
}
// CHECK-LABEL: @materialize_scalar_of_transfer_read(
// CHECK-SAME: %[[IN:.*]]: memref<32xf32>, %[[IDX:.*]]: index
// CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[SUBVIEW:.*]] = memref.subview %[[IN]][%[[IDX]]]
// CHECK:      %[[VALUE:.*]] = memref.load %[[SUBVIEW]][%[[C0]]]
// CHECK:      return %[[VALUE]] : f32
