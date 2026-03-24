// RUN: xla-opt %s --triton-xla-atomics | FileCheck %s

// CHECK-LABEL: tt.func @nomask_kernel
// CHECK-SAME: (%[[ARG0:.+]]: !tt.ptr<i32>, %[[ARG1:.+]]: i32)
tt.func @nomask_kernel(%ptr : !tt.ptr<i32>, %value : i32) {
  // CHECK-NEXT: %[[RES:.+]] = tt.elementwise_inline_asm
  // CHECK-SAME: st.global.gpu.relaxed.u32 [$1], $2;
  // CHECK-SAME: {constraints = "=r,l,r", packed_element = 1 : i32, pure = false}
  // CHECK-SAME: %[[ARG0]], %[[ARG1]] : !tt.ptr<i32>, i32 -> i32
  triton_xla.atomic_write gpu, relaxed, %ptr,  %value: (!tt.ptr<i32>, i32) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: tt.func @nomask_vector_kernel
// CHECK-SAME: (%[[ARG0:.+]]: tensor<8x!tt.ptr<i32>>, %[[ARG1:.+]]: i32)
tt.func @nomask_vector_kernel(%ptr : tensor<8x!tt.ptr<i32>>, %value : i32) {
  // CHECK-NEXT: %[[RES:.+]] = tt.elementwise_inline_asm
  // CHECK-SAME: st.global.gpu.relaxed.u32 [$1], $2;
  // CHECK-SAME: {constraints = "=r,l,r", packed_element = 1 : i32, pure = false}
  // CHECK-SAME: %[[ARG0]], %[[ARG1]] : tensor<8x!tt.ptr<i32>>, i32 -> tensor<8xi32>
  triton_xla.atomic_write gpu, relaxed, %ptr,  %value:
    (tensor<8x!tt.ptr<i32>>, i32) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: tt.func @mask_kernel
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x!tt.ptr<i32>>, %[[ARG1:.+]]: i32, %[[ARG2:.+]]: tensor<4xi1>)
tt.func @mask_kernel(%ptr : tensor<4x!tt.ptr<i32>>, %value : i32, %mask : tensor<4xi1>) {
  // CHECK-NEXT: %[[RES:.+]] = tt.elementwise_inline_asm
  // CHECK-SAME: {
  // CHECK-SAME: .reg .pred %p<>;
  // CHECK-SAME: setp.ne.u32 %p<>, $3, 0;
  // CHECK-SAME: @%p st.global.sys.release.u32 [$1], $2;
  // CHECK-SAME: }
  // CHECK-SAME: {constraints = "=r,l,r,r", packed_element = 1 : i32, pure = false}
  // CHECK-SAME: %[[ARG0]], %[[ARG1]], %[[ARG2]] : tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1> -> tensor<4xi32>
  triton_xla.atomic_write sys, release, %ptr,  %value, %mask:
    (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  // CHECK-NEXT: tt.return
  tt.return
}
