// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @types
func @types() {
  // CHECK: %{{.*}} = gml_st.point [42] : !gml_st.point
  %0 = gml_st.point [42] : !gml_st.point
  // CHECK: %{{.*}} = gml_st.tile [0] [42] [1] : !gml_st.tile<42>
  %1 = gml_st.tile [0] [42] [1] : !gml_st.tile<42>
  return
}

// -----

// CHECK-LABEL: @materialize
// CHECK-SAME: %[[MEMREF:.*]]: memref<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<42>, %[[POINT:.*]]: !gml_st.point
func @materialize(%memref: memref<?x?xf32>, %tile: !gml_st.tile<42>, %point: !gml_st.point) {
  // CHECK: %{{.*}} = gml_st.materialize %[[MEMREF]] at %[[TILE]] : memref<?x?xf32> at !gml_st.tile<42>
  %0 = gml_st.materialize %memref at %tile : memref<?x?xf32> at !gml_st.tile<42>
  // CHECK: %{{.*}} = gml_st.materialize %[[MEMREF]] at %[[POINT]] : memref<?x?xf32> at !gml_st.point
  %1 = gml_st.materialize %memref at %point : memref<?x?xf32> at !gml_st.point
  return
}

// -----

// CHECK-LABEL: @materialize
// CHECK-SAME: %[[TENSOR:.*]]: tensor<?x?xf32>, %[[TILE:.*]]: !gml_st.tile<42>, %[[POINT:.*]]: !gml_st.point
func @materialize(%tensor: tensor<?x?xf32>, %tile: !gml_st.tile<42>, %point: !gml_st.point) {
  // CHECK: %{{.*}} = gml_st.materialize %[[TENSOR]] at %[[TILE]] : tensor<?x?xf32> at !gml_st.tile<42>
  %0 = gml_st.materialize %tensor at %tile : tensor<?x?xf32> at !gml_st.tile<42>
  // CHECK: %{{.*}} = gml_st.materialize %[[TENSOR]] at %[[POINT]] : tensor<?x?xf32> at !gml_st.point
  %1 = gml_st.materialize %tensor at %point : tensor<?x?xf32> at !gml_st.point
  return
}
