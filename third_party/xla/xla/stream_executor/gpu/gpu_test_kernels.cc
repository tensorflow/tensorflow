/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/stream_executor/gpu/gpu_test_kernels.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/gpu_test_kernel_traits.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor::gpu {
absl::StatusOr<internal::AddI32Kernel::KernelType> LoadAddI32TestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::AddI32Kernel>(executor);
}

absl::StatusOr<internal::MulI32Kernel::KernelType> LoadMulI32TestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::MulI32Kernel>(executor);
}

absl::StatusOr<internal::IncAndCmpKernel::KernelType> LoadCmpAndIncTestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::IncAndCmpKernel>(executor);
}

absl::StatusOr<internal::AddI32Ptrs3Kernel::KernelType>
LoadAddI32Ptrs3TestKernel(StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::AddI32Ptrs3Kernel>(executor);
}

absl::StatusOr<internal::CopyKernel::KernelType> LoadCopyTestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::CopyKernel>(executor);
}

absl::StatusOr<KernelLoaderSpec> GetAddI32TestKernelSpec(
    Platform::Id platform_id) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .FindKernel<internal::AddI32Kernel>(platform_id);
}

absl::StatusOr<KernelLoaderSpec>
GetIncrementBy5I32TestKernelSpecWithCustomArgsPacking(
    Platform::Id platform_id) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .FindKernel<internal::IncrementBy5I32KernelWithCustomArgsPacking>(
          platform_id);
}

KernelLoaderSpec GetAddI32PtxKernelSpec() {
  // PTX kernel compiled from:
  //
  //  __global__ void add(int* a, int* b, int* c) {
  //    int index = threadIdx.x + blockIdx.x * blockDim.x;
  //    c[index] = a[index] + b[index];
  //  }
  //
  // Easiest way to get PTX from C++ is to use https://godbolt.org.
  static constexpr absl::string_view kAddI32KernelPtx = R"(
.version 4.0
.target sm_50
.address_size 64

.visible .entry AddI32(
        .param .u64 AddI32_param_0,
        .param .u64 AddI32_param_1,
        .param .u64 AddI32_param_2
)
{
        .reg .b32       %r<8>;
        .reg .b64       %rd<11>;
        .loc    1 1 0

        ld.param.u64    %rd1, [AddI32_param_0];
        ld.param.u64    %rd2, [AddI32_param_1];
        ld.param.u64    %rd3, [AddI32_param_2];
        .loc    1 3 3
        cvta.to.global.u64      %rd4, %rd3;
        cvta.to.global.u64      %rd5, %rd2;
        cvta.to.global.u64      %rd6, %rd1;
        mov.u32         %r1, %tid.x;
        mov.u32         %r2, %ctaid.x;
        mov.u32         %r3, %ntid.x;
        mad.lo.s32      %r4, %r2, %r3, %r1;
        .loc    1 4 3
        mul.wide.s32    %rd7, %r4, 4;
        add.s64         %rd8, %rd6, %rd7;
        ld.global.u32   %r5, [%rd8];
        add.s64         %rd9, %rd5, %rd7;
        ld.global.u32   %r6, [%rd9];
        add.s32         %r7, %r6, %r5;
        add.s64         %rd10, %rd4, %rd7;
        st.global.u32   [%rd10], %r7;
        .loc    1 5 1
        ret;

})";

  return KernelLoaderSpec::CreateCudaPtxInMemorySpec(kAddI32KernelPtx, "AddI32",
                                                     3);
}

KernelLoaderSpec GetTmaPtxKernelSpec() {
  // PTX kernel compiled from
  // https://github.com/jax-ml/jax/blob/739dbd3c52872e43098e28d3318b8f5f597b159d/tests/pallas/pallas_test.py#L547
  // test configuration:
  // m_1024_n_1024_k_512_dtype_float16_bm_128_bn_128_bk_32_gm_8
  // autotuner config:
  // --xla_gpu_override_gemm_autotuner='16 block_n: 16 block_k: 128 split_k: 1
  // num_stages: 4 num_warps: 4 num_ctas: 1 is_tma_allowed: true '
  static constexpr absl::string_view kTmaKernelPtx = R"(
.version 8.2
.target sm_90
.address_size 64

    // .globl    tma_dot_kernel
.extern .shared .align 16 .b8 global_smem[24600];

.visible .entry tma_dot_kernel(
    .param .align 64 .b8 tma_dot_kernel_param_0[128],
    .param .align 64 .b8 tma_dot_kernel_param_1[128],
    .param .align 64 .b8 tma_dot_kernel_param_2[128]
)
.reqntid 128, 1, 1
{
    .reg .pred     %p<46>;
    .reg .b32     %r<819>;
    .reg .b64     %rd<47>;

    mov.b64     %rd16, tma_dot_kernel_param_0;
    mov.b64     %rd17, tma_dot_kernel_param_1;
    mov.b64     %rd18, tma_dot_kernel_param_2;
    cvta.param.u64     %rd15, %rd18;
    cvta.param.u64     %rd4, %rd17;
    cvta.param.u64     %rd3, %rd16;
    mov.u32     %r769, %ctaid.x;
    shr.u32     %r770, %r769, 2;
    and.b32     %r7, %r770, 536870896;
    shl.b32     %r771, %r769, 4;
    and.b32     %r10, %r771, 1008;
    mov.u32     %r772, %tid.x;
    setp.eq.b32     %p1, %r772, 0;
    mov.b64     %rd19, global_smem;
    cvt.u32.u64     %r768, %rd19;
    add.s32     %r4, %r768, 24576;
    // begin inline asm
    @%p1 mbarrier.init.shared::cta.b64 [%r4], 1;
    // end inline asm
    bar.sync     0;
    add.s32     %r13, %r768, 24584;
    // begin inline asm
    @%p1 mbarrier.init.shared::cta.b64 [%r13], 1;
    // end inline asm
    bar.sync     0;
    add.s32     %r22, %r768, 24592;
    // begin inline asm
    @%p1 mbarrier.init.shared::cta.b64 [%r22], 1;
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r4], 8192;
    // end inline asm
    bar.sync     0;
    shr.u32     %r773, %r772, 5;
    shfl.sync.idx.b32     %r774, %r773, 0, 31, -1;
    elect.sync     %r775|%p29, -1;
    setp.lt.u32     %p30, %r772, 64;
    and.pred     %p5, %p30, %p29;
    and.b32     %r776, %r774, 1;
    shl.b32     %r777, %r776, 10;
    mul.wide.u32     %rd20, %r777, 2;
    add.s64     %rd21, %rd19, %rd20;
    shl.b32     %r6, %r776, 6;
    cvt.u32.u64     %r5, %rd21;
    // begin inline asm
    @%p5 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r5], [%rd3, {%r6, %r7}], [%r4];
    // end inline asm
    bar.sync     0;
    elect.sync     %r778|%p31, -1;
    setp.lt.u32     %p32, %r772, 32;
    and.pred     %p6, %p32, %p31;
    add.s64     %rd22, %rd19, 12288;
    cvt.u32.u64     %r9, %rd22;
    mov.b32     %r11, 0;
    // begin inline asm
    @%p6 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r9], [%rd4, {%r10, %r11}], [%r4];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r13], 8192;
    // end inline asm
    bar.sync     0;
    elect.sync     %r779|%p33, -1;
    and.pred     %p8, %p30, %p33;
    add.s64     %rd23, %rd19, 4096;
    add.s64     %rd24, %rd23, %rd20;
    or.b32     %r15, %r6, 128;
    cvt.u32.u64     %r14, %rd24;
    // begin inline asm
    @%p8 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r14], [%rd3, {%r15, %r7}], [%r13];
    // end inline asm
    bar.sync     0;
    elect.sync     %r780|%p34, -1;
    and.pred     %p9, %p32, %p34;
    add.s32     %r393, %r768, 16384;
    mov.b32     %r20, 128;
    // begin inline asm
    @%p9 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r393], [%rd4, {%r10, %r20}], [%r13];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r22], 8192;
    // end inline asm
    bar.sync     0;
    elect.sync     %r781|%p35, -1;
    and.pred     %p11, %p30, %p35;
    add.s64     %rd25, %rd19, 8192;
    add.s64     %rd26, %rd25, %rd20;
    or.b32     %r24, %r6, 256;
    cvt.u32.u64     %r23, %rd26;
    // begin inline asm
    @%p11 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r23], [%rd3, {%r24, %r7}], [%r22];
    // end inline asm
    bar.sync     0;
    elect.sync     %r782|%p36, -1;
    and.pred     %p12, %p32, %p36;
    add.s32     %r576, %r768, 20480;
    mov.b32     %r29, 256;
    // begin inline asm
    @%p12 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r576], [%rd4, {%r10, %r29}], [%r22];
    // end inline asm
    shl.b32     %r783, %r772, 7;
    and.b32     %r784, %r783, 1920;
    shl.b32     %r785, %r772, 4;
    and.b32     %r786, %r785, 112;
    or.b32     %r787, %r784, %r786;
    and.b32     %r788, %r772, 16;
    xor.b32     %r789, %r787, %r788;
    cvt.u64.u32     %rd27, %r789;
    xor.b32     %r790, %r789, 32;
    cvt.u64.u32     %rd28, %r790;
    xor.b32     %r791, %r789, 64;
    cvt.u64.u32     %rd29, %r791;
    xor.b32     %r792, %r789, 96;
    cvt.u64.u32     %rd30, %r792;
    shl.b32     %r793, %r772, 5;
    and.b32     %r794, %r793, 864;
    bfe.s32     %r795, %r772, 2, 1;
    and.b32     %r796, %r795, 144;
    or.b32     %r797, %r796, %r794;
    shr.u32     %r798, %r772, 1;
    and.b32     %r799, %r798, 16;
    xor.b32     %r800, %r797, %r799;
    cvt.u64.u32     %rd31, %r800;
    add.s64     %rd32, %rd22, %rd31;
    bar.sync     0;
    // begin inline asm
    
{
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared.b64 complete, [%r4], %r11;
    @!complete bra.uni waitLoop;
}

    // end inline asm
    add.s64     %rd33, %rd19, %rd27;
    cvt.u32.u64     %r37, %rd33;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r101, %r102, %r103, %r104}, [%r37];
    // end inline asm
    add.s32     %r591, %r37, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r157, %r158, %r159, %r160}, [%r591];
    // end inline asm
    add.s64     %rd34, %rd19, %rd28;
    cvt.u32.u64     %r47, %rd34;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r115, %r116, %r117, %r118}, [%r47];
    // end inline asm
    add.s32     %r601, %r47, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r171, %r172, %r173, %r174}, [%r601];
    // end inline asm
    add.s64     %rd35, %rd19, %rd29;
    cvt.u32.u64     %r57, %rd35;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r129, %r130, %r131, %r132}, [%r57];
    // end inline asm
    add.s32     %r611, %r57, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r185, %r186, %r187, %r188}, [%r611];
    // end inline asm
    add.s64     %rd36, %rd19, %rd30;
    cvt.u32.u64     %r67, %rd36;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r143, %r144, %r145, %r146}, [%r67];
    // end inline asm
    add.s32     %r621, %r67, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r199, %r200, %r201, %r202}, [%r621];
    // end inline asm
    cvt.u32.u64     %r77, %rd32;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r105, %r106, %r119, %r120}, [%r77];
    // end inline asm
    add.s32     %r631, %r77, 1024;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r133, %r134, %r147, %r148}, [%r631];
    // end inline asm
    add.s32     %r636, %r77, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r161, %r162, %r175, %r176}, [%r636];
    // end inline asm
    add.s32     %r641, %r77, 3072;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r189, %r190, %r203, %r204}, [%r641];
    // end inline asm
    mov.b32     %r107, %r11;
    mov.b32     %r108, %r11;
    mov.b32     %r109, %r11;
    mov.b32     %r110, %r11;
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r101, %r102, %r103, %r104 }, { %r105, %r106 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r115, %r116, %r117, %r118 }, { %r119, %r120 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r129, %r130, %r131, %r132 }, { %r133, %r134 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r143, %r144, %r145, %r146 }, { %r147, %r148 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r157, %r158, %r159, %r160 }, { %r161, %r162 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r171, %r172, %r173, %r174 }, { %r175, %r176 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r185, %r186, %r187, %r188 }, { %r189, %r190 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r199, %r200, %r201, %r202 }, { %r203, %r204 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.arrive.expect_tx.shared.b64 _, [%r4], 8192;
    // end inline asm
    // begin inline asm
    fence.proxy.async.shared::cta;
    // end inline asm
    bar.sync     0;
    elect.sync     %r801|%p37, -1;
    and.pred     %p14, %p30, %p37;
    or.b32     %r207, %r6, 384;
    // begin inline asm
    @%p14 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r5], [%rd3, {%r207, %r7}], [%r4];
    // end inline asm
    bar.sync     0;
    elect.sync     %r802|%p38, -1;
    and.pred     %p15, %p32, %p38;
    mov.b32     %r212, 384;
    // begin inline asm
    @%p15 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r9], [%rd4, {%r10, %r212}], [%r4];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    
{
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared.b64 complete, [%r13], %r11;
    @!complete bra.uni waitLoop;
}

    // end inline asm
    add.s64     %rd37, %rd23, %rd27;
    cvt.u32.u64     %r220, %rd37;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r284, %r285, %r286, %r287}, [%r220];
    // end inline asm
    add.s32     %r225, %r220, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r340, %r341, %r342, %r343}, [%r225];
    // end inline asm
    add.s64     %rd38, %rd23, %rd28;
    cvt.u32.u64     %r230, %rd38;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r298, %r299, %r300, %r301}, [%r230];
    // end inline asm
    add.s32     %r235, %r230, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r354, %r355, %r356, %r357}, [%r235];
    // end inline asm
    add.s64     %rd39, %rd23, %rd29;
    cvt.u32.u64     %r240, %rd39;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r312, %r313, %r314, %r315}, [%r240];
    // end inline asm
    add.s32     %r245, %r240, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r368, %r369, %r370, %r371}, [%r245];
    // end inline asm
    add.s64     %rd40, %rd23, %rd30;
    cvt.u32.u64     %r250, %rd40;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r326, %r327, %r328, %r329}, [%r250];
    // end inline asm
    add.s32     %r255, %r250, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r382, %r383, %r384, %r385}, [%r255];
    // end inline asm
    add.s32     %r260, %r77, 4096;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r288, %r289, %r302, %r303}, [%r260];
    // end inline asm
    add.s32     %r265, %r77, 5120;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r316, %r317, %r330, %r331}, [%r265];
    // end inline asm
    add.s32     %r270, %r77, 6144;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r344, %r345, %r358, %r359}, [%r270];
    // end inline asm
    add.s32     %r275, %r77, 7168;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r372, %r373, %r386, %r387}, [%r275];
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r284, %r285, %r286, %r287 }, { %r288, %r289 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r298, %r299, %r300, %r301 }, { %r302, %r303 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r312, %r313, %r314, %r315 }, { %r316, %r317 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r326, %r327, %r328, %r329 }, { %r330, %r331 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r340, %r341, %r342, %r343 }, { %r344, %r345 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r354, %r355, %r356, %r357 }, { %r358, %r359 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r368, %r369, %r370, %r371 }, { %r372, %r373 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r382, %r383, %r384, %r385 }, { %r386, %r387 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    bar.sync     0;
    mov.pred     %p16, 0;
    // begin inline asm
    @%p16 mbarrier.arrive.expect_tx.shared.b64 _, [%r13], 8192;
    // end inline asm
    // begin inline asm
    fence.proxy.async.shared::cta;
    // end inline asm
    bar.sync     0;
    elect.sync     %r803|%p39, -1;
    or.b32     %r390, %r6, 512;
    add.s32     %r389, %r5, 4096;
    // begin inline asm
    @%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r389], [%rd3, {%r390, %r7}], [%r13];
    // end inline asm
    bar.sync     0;
    elect.sync     %r804|%p40, -1;
    mov.b32     %r395, 512;
    // begin inline asm
    @%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r393], [%rd4, {%r10, %r395}], [%r13];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    
{
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared.b64 complete, [%r22], %r11;
    @!complete bra.uni waitLoop;
}

    // end inline asm
    add.s64     %rd41, %rd25, %rd27;
    cvt.u32.u64     %r403, %rd41;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r467, %r468, %r469, %r470}, [%r403];
    // end inline asm
    add.s32     %r408, %r403, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r523, %r524, %r525, %r526}, [%r408];
    // end inline asm
    add.s64     %rd42, %rd25, %rd28;
    cvt.u32.u64     %r413, %rd42;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r481, %r482, %r483, %r484}, [%r413];
    // end inline asm
    add.s32     %r418, %r413, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r537, %r538, %r539, %r540}, [%r418];
    // end inline asm
    add.s64     %rd43, %rd25, %rd29;
    cvt.u32.u64     %r423, %rd43;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r495, %r496, %r497, %r498}, [%r423];
    // end inline asm
    add.s32     %r428, %r423, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r551, %r552, %r553, %r554}, [%r428];
    // end inline asm
    add.s64     %rd44, %rd25, %rd30;
    cvt.u32.u64     %r433, %rd44;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r509, %r510, %r511, %r512}, [%r433];
    // end inline asm
    add.s32     %r438, %r433, 2048;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r565, %r566, %r567, %r568}, [%r438];
    // end inline asm
    add.s32     %r443, %r77, 8192;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r471, %r472, %r485, %r486}, [%r443];
    // end inline asm
    add.s32     %r448, %r77, 9216;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r499, %r500, %r513, %r514}, [%r448];
    // end inline asm
    add.s32     %r453, %r77, 10240;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r527, %r528, %r541, %r542}, [%r453];
    // end inline asm
    add.s32     %r458, %r77, 11264;
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r555, %r556, %r569, %r570}, [%r458];
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r467, %r468, %r469, %r470 }, { %r471, %r472 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r481, %r482, %r483, %r484 }, { %r485, %r486 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r495, %r496, %r497, %r498 }, { %r499, %r500 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r509, %r510, %r511, %r512 }, { %r513, %r514 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r523, %r524, %r525, %r526 }, { %r527, %r528 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r537, %r538, %r539, %r540 }, { %r541, %r542 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r551, %r552, %r553, %r554 }, { %r555, %r556 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r565, %r566, %r567, %r568 }, { %r569, %r570 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p16 mbarrier.arrive.expect_tx.shared.b64 _, [%r22], 8192;
    // end inline asm
    // begin inline asm
    fence.proxy.async.shared::cta;
    // end inline asm
    bar.sync     0;
    elect.sync     %r805|%p41, -1;
    or.b32     %r573, %r6, 640;
    add.s32     %r572, %r5, 8192;
    // begin inline asm
    @%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r572], [%rd3, {%r573, %r7}], [%r22];
    // end inline asm
    bar.sync     0;
    elect.sync     %r806|%p42, -1;
    mov.b32     %r578, 640;
    // begin inline asm
    @%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r576], [%rd4, {%r10, %r578}], [%r22];
    // end inline asm
    bar.sync     0;
    mov.b32     %r581, 1;
    // begin inline asm
    
{
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared.b64 complete, [%r4], %r581;
    @!complete bra.uni waitLoop;
}

    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r650, %r651, %r652, %r653}, [%r37];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r706, %r707, %r708, %r709}, [%r591];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r664, %r665, %r666, %r667}, [%r47];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r720, %r721, %r722, %r723}, [%r601];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r678, %r679, %r680, %r681}, [%r57];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r734, %r735, %r736, %r737}, [%r611];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r692, %r693, %r694, %r695}, [%r67];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r748, %r749, %r750, %r751}, [%r621];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r654, %r655, %r668, %r669}, [%r77];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r682, %r683, %r696, %r697}, [%r631];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r710, %r711, %r724, %r725}, [%r636];
    // end inline asm
    // begin inline asm
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r738, %r739, %r752, %r753}, [%r641];
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r650, %r651, %r652, %r653 }, { %r654, %r655 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r664, %r665, %r666, %r667 }, { %r668, %r669 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r678, %r679, %r680, %r681 }, { %r682, %r683 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r692, %r693, %r694, %r695 }, { %r696, %r697 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r706, %r707, %r708, %r709 }, { %r710, %r711 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r720, %r721, %r722, %r723 }, { %r724, %r725 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r734, %r735, %r736, %r737 }, { %r738, %r739 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    // begin inline asm
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %r107, %r108, %r109, %r110 }, { %r748, %r749, %r750, %r751 }, { %r752, %r753 }, { %r107, %r108, %r109, %r110 };
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p16 mbarrier.arrive.expect_tx.shared.b64 _, [%r4], 8192;
    // end inline asm
    // begin inline asm
    fence.proxy.async.shared::cta;
    // end inline asm
    bar.sync     0;
    elect.sync     %r807|%p43, -1;
    or.b32     %r756, %r6, 768;
    // begin inline asm
    @%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r5], [%rd3, {%r756, %r7}], [%r4];
    // end inline asm
    bar.sync     0;
    elect.sync     %r808|%p44, -1;
    mov.b32     %r761, 768;
    // begin inline asm
    @%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r9], [%rd4, {%r10, %r761}], [%r4];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.inval.shared::cta.b64 [%r4];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.inval.shared::cta.b64 [%r13];
    // end inline asm
    bar.sync     0;
    // begin inline asm
    @%p1 mbarrier.inval.shared::cta.b64 [%r22];
    // end inline asm
    and.b32     %r809, %r785, 448;
    shl.b32     %r810, %r772, 3;
    and.b32     %r811, %r810, 24;
    or.b32     %r812, %r809, %r811;
    shl.b32     %r813, %r772, 1;
    and.b32     %r814, %r813, 48;
    and.b32     %r815, %r772, 32;
    xor.b32     %r816, %r814, %r815;
    xor.b32     %r817, %r816, %r812;
    cvt.u64.u32     %rd45, %r817;
    add.s64     %rd46, %rd19, %rd45;
    st.shared.v2.b32     [%rd46], {%r107, %r108};
    st.shared.v2.b32     [%rd46+512], {%r109, %r110};
    // begin inline asm
    fence.proxy.async.shared::cta;
    // end inline asm
    bar.sync     0;
    elect.sync     %r818|%p45, -1;
    and.pred     %p28, %p32, %p45;
    // begin inline asm
    @%p28 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd15, {%r10, %r7}], [%r768];
    // end inline asm
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group.read     0;
    bar.sync     0;
    ret;

}
)";

  return KernelLoaderSpec::CreateCudaPtxInMemorySpec(kTmaKernelPtx,
                                                     "tma_dot_kernel", 3);
}  // NOLINT
}  // namespace stream_executor::gpu
