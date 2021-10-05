// RUN: lhlo-tfrt-opt %s       \
// RUN:   -lmhlo-to-gpu-binary \
// RUN: | FileCheck %s

// Note: the gpu-to-tfrt-gpu pass does not convert gpu.launch_func yet.

// CHECK: module attributes {gpu.container_module} {
// CHECK: gpu.module @[[gpu_module:.*]] attributes
// CHECK-SAME: {nvvm.cubin = "{{.*}}"} {
// CHECK:   gpu.func @[[kernel:.*]](%arg0: memref<4096xf32>) kernel {
// CHECK:     gpu.return
// CHECK:   }
// CHECK: }
// CHECK: func @fusion(%arg0: memref<4096xf32>) {
func @fusion(%arg0: memref<4096xf32>) {

    // CHECK: %[[bx:.*]] = constant 4 : index
    // CHECK: %[[by:.*]] = constant 1 : index
    // CHECK: %[[bz:.*]] = constant 1 : index
    // CHECK: %[[tx:.*]] = constant 256 : index
    // CHECK: %[[ty:.*]] = constant 1 : index
    // CHECK: %[[tz:.*]] = constant 1 : index
    // CHECK: gpu.launch_func @[[gpu_module]]::@[[kernel]]
    // CHECK-SAME: blocks in (%[[bx]], %[[by]], %[[bz]])
    // CHECK-SAME: threads in (%[[tx]], %[[ty]], %[[tz]])
    // CHECK-SAME: args(%arg0 : memref<4096xf32>)
    "lmhlo.fusion"() ( {
      %tensor = memref.tensor_load %arg0 : memref<4096xf32>
      %result = mhlo.add %tensor, %tensor : tensor<4096xf32>
      memref.tensor_store %result, %arg0 : memref<4096xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()

  "lmhlo.terminator"() : () -> ()
}
