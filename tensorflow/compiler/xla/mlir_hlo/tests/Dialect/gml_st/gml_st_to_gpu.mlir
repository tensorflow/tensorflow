// RUN: mlir-hlo-opt %s -split-input-file -verify-diagnostics \
// RUN:   -gml-st-to-gpu -cse \
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
  gml_st.parallel (%arg3) = (%c0) to (%c2048) step (%c128) {
    %3 = memref.subview %2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    %4 = memref.subview %arg2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    gml_st.parallel (%arg4) = (%c0) to (%c128) step (%c32) {
      %5 = memref.subview %3[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      %6 = memref.subview %4[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      gml_st.parallel (%arg5) = (%c0) to (%c32) step (%c1) {
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
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG:   %[[C2048:.*]] = arith.constant 2048 : index
// CHECK-DAG:   %[[GRID:.*]] = affine.apply {{.*}}(%[[C2048]])[%[[C0]], %[[C128]]]
// CHECK-DAG:   %[[BLOCK:.*]] = affine.apply {{.*}}(%[[C128]])[%[[C0]], %[[C32]]]
// CHECK-DAG:   %[[WARP:.*]] = affine.apply {{.*}}(%[[C32]])[%[[C0]], %[[C1]]]
// CHECK:       gpu.launch blocks
// CHECK-SAME:  ({{.*}}) in ({{.*}} = %[[GRID]], {{.*}} = %[[C1]], {{.*}} = %[[C1]]) threads
// CHECK-SAME:  ({{.*}}) in ({{.*}} = %[[WARP]], {{.*}} = %[[BLOCK]], {{.*}} = %[[C1]])
// CHECK:       affine.apply {{.*}}[%[[C0]], %[[C128]]]
// CHECK:       affine.apply {{.*}}[%[[C0]], %[[C32]]]
// CHECK:       affine.apply {{.*}}[%[[C0]], %[[C1]]]
// CHECK-NOT:   scf.if
// CHECK:       memref.load
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
  gml_st.parallel (%arg3) = (%c0) to (%c2048) step (%c128) {
    %3 = memref.subview %2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    %4 = memref.subview %arg2[%arg3] [128] [1] : memref<2048xf32> to memref<128xf32, #map>
    gml_st.parallel (%arg4) = (%c0) to (%c128) step (%c32) {
      %5 = memref.subview %3[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      %6 = memref.subview %4[%arg4] [32] [1] : memref<128xf32, #map> to memref<32xf32, #map>
      %7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
      gml_st.parallel (%arg5) = (%c0) to (%c32) step (%c1) {
        %8 = memref.load %6[%arg5] : memref<32xf32, #map>
        %9 = math.log %8 : f32
        memref.store %9, %7[%arg5] : memref<32xf32>
        gml_st.set_yield
      }
      gml_st.parallel (%arg6) = (%c0) to (%c32) step (%c1) {
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
  // expected-error@+1 {{failed to legalize}}
  gml_st.parallel (%arg3) = (%c0) to (%c1) step (%c1) {
    gml_st.parallel (%arg4) = (%c0) to (%c1) step (%c1) {
      gml_st.parallel (%arg5) = (%c0) to (%c1) step (%c1) {
        gml_st.parallel (%arg6) = (%c0) to (%c1) step (%c1) {
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
  // expected-error@+1 {{failed to legalize}}
  gml_st.parallel (%arg3) = (%c0) to (%c1) step (%c1) {
    gml_st.parallel (%arg4) = (%c0) to (%c1) step (%c1) {
      gml_st.parallel (%arg5) = (%c0) to (%c1) step (%c1) {
        gml_st.set_yield
      }
      gml_st.parallel (%arg6) = (%c0) to (%c2) step (%c1) {
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
  // expected-error@+1 {{failed to legalize}}
  gml_st.parallel (%arg1, %arg2) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
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
  gml_st.parallel (%arg1) = (%c0) to (%c2051) step (%c128) {
    %1 = affine.min affine_map<(d0) -> (-d0 + 2051, 128)>(%arg1)
    %2 = memref.subview %arg0[%arg1] [%1] [1] : memref<2051xf32> to memref<?xf32, #layout>
    %3 = memref.subview %0[%arg1] [%1] [1] : memref<2051xf32> to memref<?xf32, #layout>
    gml_st.parallel (%arg2) = (%c0) to (%1) step (%c32) {
      %4 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 32)>(%arg2)[%1]
      %5 = memref.subview %2[%arg2] [%4] [1] : memref<?xf32, #layout> to memref<?xf32, #layout>
      %6 = memref.subview %3[%arg2] [%4] [1] : memref<?xf32, #layout> to memref<?xf32, #layout>
      gml_st.parallel (%arg3) = (%c0) to (%4) step (%c1) {
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
