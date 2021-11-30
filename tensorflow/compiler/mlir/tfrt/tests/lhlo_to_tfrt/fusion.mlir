// RUN: lhlo-tfrt-opt %s --split-input-file \
// RUN:   -lmhlo-to-gpu-binary \
// RUN: | FileCheck %s

// Note: the gpu-to-tfrt-gpu pass does not convert gpu.launch_func yet.

// CHECK: module attributes {gpu.container_module} {
// CHECK: gpu.module @[[gpu_module:.*]] attributes {
// CHECK-SAME: binary = "
// CHECK-SAME:   .visible .entry _fusion(
// CHECK-SAME:     .param .u64 _fusion_param_0
// CHECK-SAME:   )
// CHECK-SAME: "} {
// CHECK: gpu.func @[[kernel:.*]](%arg0: memref<4096xf32>) kernel {
// CHECK:  gpu.return
// CHECK: }
// CHECK: func @fusion(%arg0: memref<4096xf32>) {
func @fusion(%arg0: memref<4096xf32>) {

    // CHECK: %[[bx:.*]] = arith.constant 4 : index
    // CHECK: %[[by:.*]] = arith.constant 1 : index
    // CHECK: %[[bz:.*]] = arith.constant 1 : index
    // CHECK: %[[tx:.*]] = arith.constant 256 : index
    // CHECK: %[[ty:.*]] = arith.constant 1 : index
    // CHECK: %[[tz:.*]] = arith.constant 1 : index
    // CHECK: gpu.launch_func @[[gpu_module]]::@[[kernel]]
    // CHECK-SAME: blocks in (%[[bx]], %[[by]], %[[bz]])
    // CHECK-SAME: threads in (%[[tx]], %[[ty]], %[[tz]])
    // CHECK-SAME: args(%arg0 : memref<4096xf32>)
    "lmhlo.fusion"() ( {
      %tensor = bufferization.to_tensor %arg0 : memref<4096xf32>
      %result = mhlo.add %tensor, %tensor : tensor<4096xf32>
      memref.tensor_store %result, %arg0 : memref<4096xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()

  "lmhlo.terminator"() : () -> ()
}

// -----

memref.global "private" constant @zero : memref<f32> = dense<0x80000000>
memref.global "private" constant @ones : memref<8xf32> = dense<
  "0x3F8000003F8000003F8000003F8000003F8000003F8000003F8000003F800000"
>


// CHECK: module attributes {gpu.container_module} {
// CHECK: gpu.module @[[gpu_module:.*]] attributes {
// CHECK-SAME: binary = "
// CHECK-SAME:   .visible .global .align 64 .b8 zero[4] = {0, 0, 0, 128};
// CHECK-SAME:   .visible .global .align 64 .b8 ones[32];
// CHECK-SAME:   .visible .entry _fusion(
// CHECK-SAME:     .param .u64 _fusion_param_0,
// CHECK-SAME:     .param .u64 _fusion_param_1
// CHECK-SAME:   )
// CHECK-SAME: ", constants = [@ones]} {
// CHECK: gpu.func @[[kernel:.*]](
// CHECK-SAME:   %arg0: memref<8x128xf32>, %arg1: memref<8xf32>
// CHECK-SAME: ) kernel {
// CHECK:   gpu.return
// CHECK: }
// CHECK: func @fusion(%arg0: memref<8x128xf32>, %arg1: memref<8xf32>) {
func @fusion(%arg0: memref<8x128xf32>, %arg1: memref<8xf32>) {

    %zero = memref.get_global @zero : memref<f32>
    %ones = memref.get_global @ones : memref<8xf32>

    // CHECK: %[[bx:.*]] = arith.constant 1 : index
    // CHECK: %[[by:.*]] = arith.constant 1 : index
    // CHECK: %[[bz:.*]] = arith.constant 1 : index
    // CHECK: %[[tx:.*]] = arith.constant 2 : index
    // CHECK: %[[ty:.*]] = arith.constant 1 : index
    // CHECK: %[[tz:.*]] = arith.constant 1 : index
    // CHECK: gpu.launch_func @[[gpu_module]]::@[[kernel]]
    // CHECK-SAME: blocks in (%[[bx]], %[[by]], %[[bz]])
    // CHECK-SAME: threads in (%[[tx]], %[[ty]], %[[tz]])
    // CHECK-SAME: args(%arg0 : memref<8x128xf32>, %arg1 : memref<8xf32>)
    "lmhlo.fusion"() ({
      %clamp = bufferization.to_tensor %zero : memref<f32>
      %bias = bufferization.to_tensor %ones : memref<8xf32>
      %tensor = bufferization.to_tensor %arg0 : memref<8x128xf32>
      %0 = "mhlo.reduce"(%tensor, %clamp) ({
      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %max = mhlo.maximum %arg2, %arg3 : tensor<f32>
        "mhlo.return"(%max) : (tensor<f32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>}
        : (tensor<8x128xf32>, tensor<f32>) -> tensor<8xf32>
      %1 = mhlo.add %0, %bias : tensor<8xf32>
      memref.tensor_store %1, %arg1 : memref<8xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()

  "lmhlo.terminator"() : () -> ()
}
