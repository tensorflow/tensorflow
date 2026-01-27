// RUN: xla-opt %s --triton-xla-atomics | FileCheck %s

// CHECK-LABEL: tt.func @nomask_kernel
// CHECK-SAME:    %[[ARG0:.*]]: !tt.ptr<i32>
// CHECK-SAME:    %[[ARG1:.*]]: i32
tt.func @nomask_kernel(%ptr : !tt.ptr<i32>, %expected : i32) {
// CHECK-NEXT:  %[[RES:.+]] = tt.elementwise_inline_asm
// CHECK-SAME:  {
// CHECK-SAME:  .reg .pred %p<1>;
// CHECK-SAME:  .reg .b32 %r<1>;
// CHECK-SAME:  wait:
// CHECK-SAME:  ld.global.gpu.relaxed.u32 %r0, [$1];
// CHECK-SAME:  setp.eq.u32 %p0, %r0, $2;
// CHECK-SAME:  @%p0 bra wait;
// CHECK-SAME:  }
// CHECK-SAME:  {constraints = "=r,l,r", packed_element = 1 : i32, pure = false}
// CHECK-SAME:  %[[ARG0]], %[[ARG1]]
// CHECK-SAME:  !tt.ptr<i32>, i32 -> i32
  triton_xla.atomic_spin_wait gpu, relaxed, %ptr, equal_to, %expected  : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: tt.func @masked_kernel
// CHECK-SAME:    %[[ARG0:.*]]: tensor<4x!tt.ptr<i32>>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<4xi1>
// CHECK-SAME:    %[[ARG2:.*]]: i32
tt.func @masked_kernel(
  %ptr: tensor<4x!tt.ptr<i32>>,
  %mask: tensor<4xi1>,
  %expected: i32
) {
// CHECK:         tt.elementwise_inline_asm
// CHECK-SAME:    {
// CHECK-SAME:    .reg .pred %p<2>;
// CHECK-SAME:    .reg .b32 %r<1>;
// CHECK-SAME:    setp.ne.u32 %p0, $3, 0;
// CHECK-SAME:    @%!p0 bra done;
// CHECK-SAME:    wait:
// CHECK-SAME:    ld.global.gpu.acquire.u32 %r0, [$1];
// CHECK-SAME:    setp.lt.u32 %p1, %r0, $2;
// CHECK-SAME:    @%p1 bra wait;
// CHECK-SAME:    done:
// CHECK-SAME:    }
// CHECK-SAME:    {constraints = "=r,l,r,r", packed_element = 1 : i32, pure = false}
// CHECK-SAME:    %[[ARG0]], %[[ARG2]], %[[ARG1]]
// CHECK-SAME:    tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1> -> tensor<4xi32>
  triton_xla.atomic_spin_wait gpu, acquire, %ptr, less_than, %expected, %mask
      : (tensor<4x!tt.ptr<i32>>, i32, tensor<4xi1>) -> ()
  tt.return
}
