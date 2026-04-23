// RUN: xla-opt %s --triton-xla-atomics | FileCheck %s

// Test lowering of AtomicWriteOp to tt.extern_elementwise

// CHECK-LABEL: tt.func @nomask_kernel
// CHECK-SAME: (%[[ARG0:.+]]: !tt.ptr<i32>, %[[ARG1:.+]]: i32)
tt.func @nomask_kernel(%ptr : !tt.ptr<i32>, %value : i32) {
  // CHECK-NEXT: %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: {libname = "", libpath = "", pure = false, symbol = "xla_atomicwrite_relaxed_gpu_nomask"}
  // CHECK-SAME: : (!tt.ptr<i32>, i32) -> i32
  triton_xla.atomic_write gpu, relaxed, %ptr,  %value: (!tt.ptr<i32>, i32) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: tt.func @nomask_vector_kernel
// CHECK-SAME: (%[[ARG0:.+]]: tensor<8x!tt.ptr<i32>>, %[[ARG1:.+]]: i32)
tt.func @nomask_vector_kernel(%ptr : tensor<8x!tt.ptr<i32>>, %value : i32) {
  // CHECK-NEXT: %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: {libname = "", libpath = "", pure = false, symbol = "xla_atomicwrite_relaxed_gpu_nomask"}
  // CHECK-SAME: : (tensor<8x!tt.ptr<i32>>, i32) -> tensor<8xi32>
  triton_xla.atomic_write gpu, relaxed, %ptr,  %value:
    (tensor<8x!tt.ptr<i32>>, i32) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: tt.func @mask_kernel
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x!tt.ptr<i32>>, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: tensor<4xi1>)
tt.func @mask_kernel(%ptr : tensor<4x!tt.ptr<i32>>, %value : i32, %mask : tensor<4xi1>) {
  // CHECK-NEXT: %[[RES:.+]] = tt.extern_elementwise %[[ARG0]], %[[ARG1]], %[[ARG2]]
  // CHECK-SAME: {libname = "", libpath = "", pure = false, symbol = "xla_atomicwrite_release_system_mask"}
  // CHECK-SAME: : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> tensor<4xi32>
  triton_xla.atomic_write sys, release, %ptr,  %value, %mask:
    (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: tt.func @test_different_scopes
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x!tt.ptr<i32>>, %[[ARG1:.+]]: i32)
tt.func @test_different_scopes(%ptr: tensor<4x!tt.ptr<i32>>, %value: i32) {
  // CHECK: tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_atomicwrite_release_gpu_nomask"
  triton_xla.atomic_write gpu, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  
  // CHECK: tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_atomicwrite_release_cta_nomask"
  triton_xla.atomic_write cta, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: tt.func @test_scalar_with_mask
// CHECK-SAME: (%[[ARG0:.+]]: !tt.ptr<i32>, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: i1)
tt.func @test_scalar_with_mask(%ptr: !tt.ptr<i32>, %value: i32, %mask: i1) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.extern_elementwise %[[ARG0]], %[[ARG1]], %[[ARG2]]
  // CHECK-SAME: symbol = "xla_atomicwrite_release_system_mask"
  triton_xla.atomic_write sys, release, %ptr, %value, %mask : (!tt.ptr<i32>, i32, i1) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}
