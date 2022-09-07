// RUN: xla-gpu-opt %s -xla-memref-get-global-to-arg=min-num-elements=2 \
// RUN:   | FileCheck %s

#map = affine_map<(d0, d1) -> (d0 + 2 * d1)>

memref.global "private" constant @cst0 : memref<2x3xf32> =
  dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
         [4.000000e+00, 5.000000e+00, 6.000000e+00]]>

memref.global "private" constant @cst1 : memref<f32> =
  dense<1.000000e+00>

memref.global "private" constant @cst2 : memref<2x3xf32, #map> =
  dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
         [4.000000e+00, 5.000000e+00, 6.000000e+00]]>

// CHECK: func.func @get_global(
// CHECK-SAME:   %[[ARG0:.*]]: memref<24xi8> {lmhlo.constant_name = "cst0"},
// CHECK-SAME:   %[[ARG1:.*]]: memref<4xi8> {lmhlo.constant_name = "cst1"},
// CHECK-SAME:   %[[ARG2:.*]]: memref<24xi8> {lmhlo.constant_name = "cst2"}
// CHECK-SAME: )
func.func @get_global(%arg0: memref<24xi8> {lmhlo.constant_name = "cst0"},
                      %arg1: memref<4xi8> {lmhlo.constant_name = "cst1"},
                      %arg2: memref<24xi8> {lmhlo.constant_name = "cst2"})
    -> (memref<2x3xf32>, memref<f32>, memref<2x3xf32, #map>) {

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = memref.view %[[ARG0]][%[[C0]]][] {{.*}} memref<2x3xf32>
  %0 = memref.get_global @cst0 : memref<2x3xf32>

  // CHECK: %[[V1:.*]] = memref.get_global {{.*}} : memref<f32>
  %1 = memref.get_global @cst1 : memref<f32>

  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[F:.*]] = memref.view %[[ARG2]][%[[C0_1]]][] {{.*}} memref<6xf32>
  // CHECK: %[[V2:.*]] = memref.reinterpret_cast %[[F]]
  // CHECK-SAME: to offset: [0], sizes: [2, 3], strides: [1, 2]
  %2 = memref.get_global @cst2 : memref<2x3xf32, #map>

  // CHECK: return %[[V0]], %[[V1]], %[[V2]]
  // CHECK-SAME: : memref<2x3xf32>, memref<f32>, memref<2x3xf32, #map>
  return %0, %1, %2 : memref<2x3xf32>, memref<f32>, memref<2x3xf32, #map>
}
