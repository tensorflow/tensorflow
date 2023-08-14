/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/stream_executor/kernel.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

static constexpr double kTolerance = 0.1f;

// Comparison kernel code: compare two buffers of
// fp8/bf16/fp16/fp32/fp64/int8_t/int32_t of length buffer_length where the
// relative error does not exceed the passed rel_error_threshold. Write the
// number of mismatches into out parameter mismatch_count.

// NaN's are considered equal, and for half's we clamp all numbers to largest
// and smallest numbers representable to avoid miscomparisons due to overflows.
//
// The PTX below is compiled from the CUDA code below. The following command was
// used with NVCC from CUDA 11.8
//
//   nvcc --gpu-architecture=compute_50 --ptx buffer_compare.cu
//
// The CUDA code follows:

// #include <cuda_bf16.h>
// #include <cuda_fp16.h>
// #include <cuda_fp8.h>
//
// namespace {
//
// __device__ __inline__ float __xla_buffer_comparator_canonicalize(float input)
// {
//   // All fp16 infinities are treated as 65505 or -65505, in order to avoid
//   // differences due to overflows.
//   return isnan(input) ? input : max(-65505.0f, min(input, 65505.0f));
// }
//
// } // end anonymous namespace
//
// extern "C" { // avoid name mangling
//
//
// __global__ void __xla_fp8_e4m3fn_comparison(__nv_fp8_storage_t *buffer_a,
//                                             __nv_fp8_storage_t *buffer_b,
//                                             float rel_error_threshold,
//                                             unsigned long long buffer_length,
//                                             int *mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length)
//     return;
//   // TODO(philipphack): Replace with direct conversion to float when this
//   // functionality becomes availabe.
//   float elem_a =
//       __half2float(__nv_cvt_fp8_to_halfraw(buffer_a[idx], __NV_E4M3));
//   float elem_b =
//       __half2float(__nv_cvt_fp8_to_halfraw(buffer_b[idx], __NV_E4M3));
//   elem_a = __xla_buffer_comparator_canonicalize(elem_a);
//   elem_b = __xla_buffer_comparator_canonicalize(elem_b);
//   if (isnan(elem_a) && isnan(elem_b))
//     return;
//
//   float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) +
//   1);
//
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_fp8_e5m2_comparison(__nv_fp8_storage_t *buffer_a,
//                                           __nv_fp8_storage_t *buffer_b,
//                                           float rel_error_threshold,
//                                           unsigned long long buffer_length,
//                                           int *mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length)
//     return;
//   // TODO(philipphack): Replace with direct conversion to float when this
//   // functionality becomes availabe.
//   float elem_a =
//       __half2float(__nv_cvt_fp8_to_halfraw(buffer_a[idx], __NV_E5M2));
//   float elem_b =
//       __half2float(__nv_cvt_fp8_to_halfraw(buffer_b[idx], __NV_E5M2));
//   elem_a = __xla_buffer_comparator_canonicalize(elem_a);
//   elem_b = __xla_buffer_comparator_canonicalize(elem_b);
//   if (isnan(elem_a) && isnan(elem_b))
//     return;
//
//   float rel_error = abs(elem_a - elem_b) / (max(abs(elem_a), abs(elem_b)) +
//   1);
//
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_fp16_comparison(__half* buffer_a, __half* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float elem_a = __half2float(buffer_a[idx]);
//   float elem_b = __half2float(buffer_b[idx]);
//   elem_a = __xla_buffer_comparator_canonicalize(elem_a);
//   elem_b = __xla_buffer_comparator_canonicalize(elem_b);
//   if (isnan(elem_a) && isnan(elem_b)) return;
//
//   float rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_fp32_comparison(float* buffer_a, float* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float elem_a = buffer_a[idx];
//   float elem_b = buffer_b[idx];
//   if (isnan(elem_a) && isnan(elem_b)) return;
//   if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b))
//     return;
//
//   float rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_fp64_comparison(double* buffer_a, double* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//
//   double elem_a = buffer_a[idx];
//   double elem_b = buffer_b[idx];
//   if (isnan(elem_a) && isnan(elem_b)) return;
//   if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b))
//     return;
//   double rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_bf16_comparison(__nv_bfloat16* buffer_a,
//                                       __nv_bfloat16* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float elem_a = __bfloat162float(buffer_a[idx]);
//   float elem_b = __bfloat162float(buffer_b[idx]);
//   elem_a = __xla_buffer_comparator_canonicalize(elem_a);
//   elem_b = __xla_buffer_comparator_canonicalize(elem_b);
//   if (isnan(elem_a) && isnan(elem_b)) return;
//
//   float rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// // TODO(b/191520348): The comparison below requires exact equality.
// __global__ void __xla_int8_comparison(int8_t* buffer_a, int8_t* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float a = buffer_a[idx];
//   float b = buffer_b[idx];
//   float rel_error = abs(a - b) / (max(abs(a), abs(b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//       atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_int32_comparison(int* buffer_a, int* buffer_b,
//                                        float rel_error_threshold,
//                                        unsigned long long buffer_length,
//                                        int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float elem_a = static_cast<float>(buffer_a[idx]);
//   float elem_b = static_cast<float>(buffer_b[idx]);
//   float rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
// } // end extern declaration

static const char* buffer_compare_ptx = R"(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-31833905
// Cuda compilation tools, release 11.8, V11.8.89
// Based on NVVM 7.0.1
//

.version 7.8
.target sm_50
.address_size 64

	// .globl	__xla_fp8_e4m3fn_comparison

.visible .entry __xla_fp8_e4m3fn_comparison(
	.param .u64 __xla_fp8_e4m3fn_comparison_param_0,
	.param .u64 __xla_fp8_e4m3fn_comparison_param_1,
	.param .f32 __xla_fp8_e4m3fn_comparison_param_2,
	.param .u64 __xla_fp8_e4m3fn_comparison_param_3,
	.param .u64 __xla_fp8_e4m3fn_comparison_param_4
)
{
	.reg .pred 	%p<19>;
	.reg .b16 	%rs<71>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd2, [__xla_fp8_e4m3fn_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_fp8_e4m3fn_comparison_param_1];
	ld.param.f32 	%f12, [__xla_fp8_e4m3fn_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_fp8_e4m3fn_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_fp8_e4m3fn_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB0_27;

	cvta.to.global.u64 	%rd6, %rd2;
	add.s64 	%rd7, %rd6, %rd1;
	ld.global.u8 	%rs1, [%rd7];
	shl.b16 	%rs38, %rs1, 8;
	and.b16  	%rs2, %rs38, -32768;
	and.b16  	%rs3, %rs38, 30720;
	shr.u16 	%rs39, %rs3, 1;
	add.s16 	%rs59, %rs39, 8192;
	and.b16  	%rs60, %rs38, 1792;
	shr.u16 	%rs62, %rs60, 1;
	and.b16  	%rs40, %rs1, 127;
	setp.eq.s16 	%p2, %rs40, 127;
	mov.u16 	%rs70, 32767;
	mov.u16 	%rs63, %rs70;
	@%p2 bra 	$L__BB0_10;

	setp.eq.s16 	%p3, %rs3, 0;
	@%p3 bra 	$L__BB0_4;

	or.b16  	%rs41, %rs62, %rs2;
	or.b16  	%rs63, %rs41, %rs59;
	bra.uni 	$L__BB0_10;

$L__BB0_4:
	setp.eq.s16 	%p4, %rs60, 0;
	mov.u16 	%rs61, 0;
	@%p4 bra 	$L__BB0_9;

	and.b16  	%rs43, %rs1, 4;
	setp.ne.s16 	%p5, %rs43, 0;
	@%p5 bra 	$L__BB0_8;

	mov.u16 	%rs57, %rs60;

$L__BB0_7:
	shl.b16 	%rs60, %rs57, 1;
	add.s16 	%rs59, %rs59, -1024;
	and.b16  	%rs44, %rs57, 512;
	setp.eq.s16 	%p6, %rs44, 0;
	mov.u16 	%rs57, %rs60;
	@%p6 bra 	$L__BB0_7;

$L__BB0_8:
	and.b16  	%rs62, %rs60, 1022;
	mov.u16 	%rs61, %rs59;

$L__BB0_9:
	or.b16  	%rs45, %rs61, %rs2;
	or.b16  	%rs63, %rs45, %rs62;

$L__BB0_10:
	// begin inline asm
	{  cvt.f32.f16 %f27, %rs63;}

	// end inline asm
	cvta.to.global.u64 	%rd8, %rd3;
	add.s64 	%rd9, %rd8, %rd1;
	ld.global.u8 	%rs19, [%rd9];
	shl.b16 	%rs48, %rs19, 8;
	and.b16  	%rs20, %rs48, -32768;
	and.b16  	%rs21, %rs48, 30720;
	shr.u16 	%rs49, %rs21, 1;
	add.s16 	%rs66, %rs49, 8192;
	and.b16  	%rs67, %rs48, 1792;
	shr.u16 	%rs69, %rs67, 1;
	and.b16  	%rs50, %rs19, 127;
	setp.eq.s16 	%p7, %rs50, 127;
	@%p7 bra 	$L__BB0_19;

	setp.eq.s16 	%p8, %rs21, 0;
	@%p8 bra 	$L__BB0_13;

	or.b16  	%rs51, %rs69, %rs20;
	or.b16  	%rs70, %rs51, %rs66;
	bra.uni 	$L__BB0_19;

$L__BB0_13:
	setp.eq.s16 	%p9, %rs67, 0;
	mov.u16 	%rs68, 0;
	@%p9 bra 	$L__BB0_18;

	and.b16  	%rs53, %rs19, 4;
	setp.ne.s16 	%p10, %rs53, 0;
	@%p10 bra 	$L__BB0_17;

	mov.u16 	%rs64, %rs67;

$L__BB0_16:
	shl.b16 	%rs67, %rs64, 1;
	add.s16 	%rs66, %rs66, -1024;
	and.b16  	%rs54, %rs64, 512;
	setp.eq.s16 	%p11, %rs54, 0;
	mov.u16 	%rs64, %rs67;
	@%p11 bra 	$L__BB0_16;

$L__BB0_17:
	and.b16  	%rs69, %rs67, 1022;
	mov.u16 	%rs68, %rs66;

$L__BB0_18:
	or.b16  	%rs55, %rs68, %rs20;
	or.b16  	%rs70, %rs55, %rs69;

$L__BB0_19:
	// begin inline asm
	{  cvt.f32.f16 %f29, %rs70;}

	// end inline asm
	abs.f32 	%f15, %f27;
	setp.gtu.f32 	%p12, %f15, 0f7F800000;
	@%p12 bra 	$L__BB0_21;

	mov.f32 	%f16, 0f477FE100;
	min.f32 	%f17, %f27, %f16;
	mov.f32 	%f18, 0fC77FE100;
	max.f32 	%f27, %f18, %f17;

$L__BB0_21:
	abs.f32 	%f28, %f29;
	setp.gtu.f32 	%p13, %f28, 0f7F800000;
	@%p13 bra 	$L__BB0_23;

	mov.f32 	%f19, 0f477FE100;
	min.f32 	%f20, %f29, %f19;
	mov.f32 	%f21, 0fC77FE100;
	max.f32 	%f29, %f21, %f20;
	abs.f32 	%f28, %f29;

$L__BB0_23:
	abs.f32 	%f10, %f27;
	setp.gtu.f32 	%p14, %f10, 0f7F800000;
	setp.gtu.f32 	%p15, %f28, 0f7F800000;
	and.pred  	%p16, %p14, %p15;
	@%p16 bra 	$L__BB0_27;

	sub.f32 	%f22, %f27, %f29;
	abs.f32 	%f23, %f22;
	max.f32 	%f24, %f10, %f28;
	add.f32 	%f25, %f24, 0f3F800000;
	div.rn.f32 	%f11, %f23, %f25;
	setp.gt.f32 	%p17, %f11, %f12;
	@%p17 bra 	$L__BB0_26;

	abs.f32 	%f26, %f11;
	setp.le.f32 	%p18, %f26, 0f7F800000;
	@%p18 bra 	$L__BB0_27;

$L__BB0_26:
	cvta.to.global.u64 	%rd10, %rd4;
	atom.global.add.u32 	%r5, [%rd10], 1;

$L__BB0_27:
	ret;

}
	// .globl	__xla_fp8_e5m2_comparison
.visible .entry __xla_fp8_e5m2_comparison(
	.param .u64 __xla_fp8_e5m2_comparison_param_0,
	.param .u64 __xla_fp8_e5m2_comparison_param_1,
	.param .f32 __xla_fp8_e5m2_comparison_param_2,
	.param .u64 __xla_fp8_e5m2_comparison_param_3,
	.param .u64 __xla_fp8_e5m2_comparison_param_4
)
{
	.reg .pred 	%p<11>;
	.reg .b16 	%rs<9>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd2, [__xla_fp8_e5m2_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_fp8_e5m2_comparison_param_1];
	ld.param.f32 	%f12, [__xla_fp8_e5m2_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_fp8_e5m2_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_fp8_e5m2_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB1_9;

	cvta.to.global.u64 	%rd6, %rd2;
	add.s64 	%rd7, %rd6, %rd1;
	ld.global.u8 	%rs3, [%rd7];
	shl.b16 	%rs4, %rs3, 8;
	and.b16  	%rs5, %rs3, 127;
	setp.gt.u16 	%p2, %rs5, 124;
	selp.b16 	%rs1, 32767, %rs4, %p2;
	// begin inline asm
	{  cvt.f32.f16 %f27, %rs1;}

	// end inline asm
	cvta.to.global.u64 	%rd8, %rd3;
	add.s64 	%rd9, %rd8, %rd1;
	ld.global.u8 	%rs6, [%rd9];
	shl.b16 	%rs7, %rs6, 8;
	and.b16  	%rs8, %rs6, 127;
	setp.gt.u16 	%p3, %rs8, 124;
	selp.b16 	%rs2, 32767, %rs7, %p3;
	// begin inline asm
	{  cvt.f32.f16 %f29, %rs2;}

	// end inline asm
	abs.f32 	%f15, %f27;
	setp.gtu.f32 	%p4, %f15, 0f7F800000;
	@%p4 bra 	$L__BB1_3;

	mov.f32 	%f16, 0f477FE100;
	min.f32 	%f17, %f27, %f16;
	mov.f32 	%f18, 0fC77FE100;
	max.f32 	%f27, %f18, %f17;

$L__BB1_3:
	abs.f32 	%f28, %f29;
	setp.gtu.f32 	%p5, %f28, 0f7F800000;
	@%p5 bra 	$L__BB1_5;

	mov.f32 	%f19, 0f477FE100;
	min.f32 	%f20, %f29, %f19;
	mov.f32 	%f21, 0fC77FE100;
	max.f32 	%f29, %f21, %f20;
	abs.f32 	%f28, %f29;

$L__BB1_5:
	abs.f32 	%f10, %f27;
	setp.gtu.f32 	%p6, %f10, 0f7F800000;
	setp.gtu.f32 	%p7, %f28, 0f7F800000;
	and.pred  	%p8, %p6, %p7;
	@%p8 bra 	$L__BB1_9;

	sub.f32 	%f22, %f27, %f29;
	abs.f32 	%f23, %f22;
	max.f32 	%f24, %f10, %f28;
	add.f32 	%f25, %f24, 0f3F800000;
	div.rn.f32 	%f11, %f23, %f25;
	setp.gt.f32 	%p9, %f11, %f12;
	@%p9 bra 	$L__BB1_8;

	abs.f32 	%f26, %f11;
	setp.le.f32 	%p10, %f26, 0f7F800000;
	@%p10 bra 	$L__BB1_9;

$L__BB1_8:
	cvta.to.global.u64 	%rd10, %rd4;
	atom.global.add.u32 	%r5, [%rd10], 1;

$L__BB1_9:
	ret;

}
	// .globl	__xla_fp16_comparison
.visible .entry __xla_fp16_comparison(
	.param .u64 __xla_fp16_comparison_param_0,
	.param .u64 __xla_fp16_comparison_param_1,
	.param .f32 __xla_fp16_comparison_param_2,
	.param .u64 __xla_fp16_comparison_param_3,
	.param .u64 __xla_fp16_comparison_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .b16 	%rs<3>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd2, [__xla_fp16_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_fp16_comparison_param_1];
	ld.param.f32 	%f12, [__xla_fp16_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_fp16_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_fp16_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB2_9;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 1;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.u16 	%rs1, [%rd8];
	// begin inline asm
	{  cvt.f32.f16 %f27, %rs1;}

	// end inline asm
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.u16 	%rs2, [%rd10];
	// begin inline asm
	{  cvt.f32.f16 %f29, %rs2;}

	// end inline asm
	abs.f32 	%f15, %f27;
	setp.gtu.f32 	%p2, %f15, 0f7F800000;
	@%p2 bra 	$L__BB2_3;

	mov.f32 	%f16, 0f477FE100;
	min.f32 	%f17, %f27, %f16;
	mov.f32 	%f18, 0fC77FE100;
	max.f32 	%f27, %f18, %f17;

$L__BB2_3:
	abs.f32 	%f28, %f29;
	setp.gtu.f32 	%p3, %f28, 0f7F800000;
	@%p3 bra 	$L__BB2_5;

	mov.f32 	%f19, 0f477FE100;
	min.f32 	%f20, %f29, %f19;
	mov.f32 	%f21, 0fC77FE100;
	max.f32 	%f29, %f21, %f20;
	abs.f32 	%f28, %f29;

$L__BB2_5:
	abs.f32 	%f10, %f27;
	setp.gtu.f32 	%p4, %f10, 0f7F800000;
	setp.gtu.f32 	%p5, %f28, 0f7F800000;
	and.pred  	%p6, %p4, %p5;
	@%p6 bra 	$L__BB2_9;

	sub.f32 	%f22, %f27, %f29;
	abs.f32 	%f23, %f22;
	max.f32 	%f24, %f10, %f28;
	add.f32 	%f25, %f24, 0f3F800000;
	div.rn.f32 	%f11, %f23, %f25;
	setp.gt.f32 	%p7, %f11, %f12;
	@%p7 bra 	$L__BB2_8;

	abs.f32 	%f26, %f11;
	setp.le.f32 	%p8, %f26, 0f7F800000;
	@%p8 bra 	$L__BB2_9;

$L__BB2_8:
	cvta.to.global.u64 	%rd11, %rd4;
	atom.global.add.u32 	%r5, [%rd11], 1;

$L__BB2_9:
	ret;

}
	// .globl	__xla_fp32_comparison
.visible .entry __xla_fp32_comparison(
	.param .u64 __xla_fp32_comparison_param_0,
	.param .u64 __xla_fp32_comparison_param_1,
	.param .f32 __xla_fp32_comparison_param_2,
	.param .u64 __xla_fp32_comparison_param_3,
	.param .u64 __xla_fp32_comparison_param_4
)
{
	.reg .pred 	%p<10>;
	.reg .b16 	%rs<3>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<10>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd2, [__xla_fp32_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_fp32_comparison_param_1];
	ld.param.f32 	%f7, [__xla_fp32_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_fp32_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_fp32_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB3_8;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 2;
	add.s64 	%rd8, %rd6, %rd7;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.f32 	%f1, [%rd8];
	abs.f32 	%f2, %f1;
	setp.le.f32 	%p2, %f2, 0f7F800000;
	ld.global.f32 	%f3, [%rd10];
	abs.f32 	%f14, %f3;
	@%p2 bra 	$L__BB3_3;

	setp.gtu.f32 	%p3, %f14, 0f7F800000;
	@%p3 bra 	$L__BB3_8;

$L__BB3_3:
	setp.neu.f32 	%p4, %f2, 0f7F800000;
	setp.neu.f32 	%p5, %f14, 0f7F800000;
	or.pred  	%p6, %p4, %p5;
	@%p6 bra 	$L__BB3_5;

	mov.b32 	%r5, %f1;
	shr.u32 	%r6, %r5, 31;
	cvt.u16.u32 	%rs1, %r6;
	mov.b32 	%r7, %f3;
	shr.u32 	%r8, %r7, 31;
	cvt.u16.u32 	%rs2, %r8;
	setp.eq.s16 	%p7, %rs1, %rs2;
	mov.f32 	%f14, 0f7F800000;
	@%p7 bra 	$L__BB3_8;

$L__BB3_5:
	sub.f32 	%f9, %f1, %f3;
	abs.f32 	%f10, %f9;
	max.f32 	%f11, %f2, %f14;
	add.f32 	%f12, %f11, 0f3F800000;
	div.rn.f32 	%f6, %f10, %f12;
	setp.gt.f32 	%p8, %f6, %f7;
	@%p8 bra 	$L__BB3_7;

	abs.f32 	%f13, %f6;
	setp.le.f32 	%p9, %f13, 0f7F800000;
	@%p9 bra 	$L__BB3_8;

$L__BB3_7:
	cvta.to.global.u64 	%rd11, %rd4;
	atom.global.add.u32 	%r9, [%rd11], 1;

$L__BB3_8:
	ret;

}
	// .globl	__xla_fp64_comparison
.visible .entry __xla_fp64_comparison(
	.param .u64 __xla_fp64_comparison_param_0,
	.param .u64 __xla_fp64_comparison_param_1,
	.param .f32 __xla_fp64_comparison_param_2,
	.param .u64 __xla_fp64_comparison_param_3,
	.param .u64 __xla_fp64_comparison_param_4
)
{
	.reg .pred 	%p<13>;
	.reg .b16 	%rs<3>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<14>;
	.reg .f64 	%fd<13>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd2, [__xla_fp64_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_fp64_comparison_param_1];
	ld.param.f32 	%f1, [__xla_fp64_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_fp64_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_fp64_comparison_param_4];
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %ctaid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	cvt.s64.s32 	%rd1, %r6;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB4_9;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 3;
	add.s64 	%rd8, %rd6, %rd7;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.f64 	%fd1, [%rd10];
	ld.global.f64 	%fd2, [%rd8];
	abs.f64 	%fd3, %fd2;
	setp.le.f64 	%p2, %fd3, 0d7FF0000000000000;
	@%p2 bra 	$L__BB4_3;

	abs.f64 	%fd5, %fd1;
	setp.gtu.f64 	%p3, %fd5, 0d7FF0000000000000;
	@%p3 bra 	$L__BB4_9;

$L__BB4_3:
	{
	.reg .b32 %temp; 
	mov.b64 	{%r7, %temp}, %fd2;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r1}, %fd2;
	}
	and.b32  	%r8, %r1, 2147483647;
	setp.ne.s32 	%p4, %r8, 2146435072;
	setp.ne.s32 	%p5, %r7, 0;
	or.pred  	%p6, %p4, %p5;
	@%p6 bra 	$L__BB4_6;

	{
	.reg .b32 %temp; 
	mov.b64 	{%r9, %temp}, %fd1;
	}
	{
	.reg .b32 %temp; 
	mov.b64 	{%temp, %r2}, %fd1;
	}
	and.b32  	%r10, %r2, 2147483647;
	setp.ne.s32 	%p7, %r10, 2146435072;
	setp.ne.s32 	%p8, %r9, 0;
	or.pred  	%p9, %p7, %p8;
	@%p9 bra 	$L__BB4_6;

	shr.u32 	%r11, %r1, 31;
	cvt.u16.u32 	%rs1, %r11;
	shr.u32 	%r12, %r2, 31;
	cvt.u16.u32 	%rs2, %r12;
	setp.eq.s16 	%p10, %rs1, %rs2;
	@%p10 bra 	$L__BB4_9;

$L__BB4_6:
	sub.f64 	%fd6, %fd2, %fd1;
	abs.f64 	%fd7, %fd6;
	abs.f64 	%fd8, %fd1;
	max.f64 	%fd9, %fd3, %fd8;
	add.f64 	%fd10, %fd9, 0d3FF0000000000000;
	div.rn.f64 	%fd4, %fd7, %fd10;
	cvt.f64.f32 	%fd11, %f1;
	setp.gt.f64 	%p11, %fd4, %fd11;
	@%p11 bra 	$L__BB4_8;

	abs.f64 	%fd12, %fd4;
	setp.le.f64 	%p12, %fd12, 0d7FF0000000000000;
	@%p12 bra 	$L__BB4_9;

$L__BB4_8:
	cvta.to.global.u64 	%rd11, %rd4;
	atom.global.add.u32 	%r13, [%rd11], 1;

$L__BB4_9:
	ret;

}
	// .globl	__xla_bf16_comparison
.visible .entry __xla_bf16_comparison(
	.param .u64 __xla_bf16_comparison_param_0,
	.param .u64 __xla_bf16_comparison_param_1,
	.param .f32 __xla_bf16_comparison_param_2,
	.param .u64 __xla_bf16_comparison_param_3,
	.param .u64 __xla_bf16_comparison_param_4
)
{
	.reg .pred 	%p<9>;
	.reg .b16 	%rs<3>;
	.reg .f32 	%f<30>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd2, [__xla_bf16_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_bf16_comparison_param_1];
	ld.param.f32 	%f12, [__xla_bf16_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_bf16_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_bf16_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB5_9;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 1;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.u16 	%rs1, [%rd8];
	// begin inline asm
	{ mov.b32 %f27, {0,%rs1};}

	// end inline asm
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.u16 	%rs2, [%rd10];
	// begin inline asm
	{ mov.b32 %f29, {0,%rs2};}

	// end inline asm
	abs.f32 	%f15, %f27;
	setp.gtu.f32 	%p2, %f15, 0f7F800000;
	@%p2 bra 	$L__BB5_3;

	mov.f32 	%f16, 0f477FE100;
	min.f32 	%f17, %f27, %f16;
	mov.f32 	%f18, 0fC77FE100;
	max.f32 	%f27, %f18, %f17;

$L__BB5_3:
	abs.f32 	%f28, %f29;
	setp.gtu.f32 	%p3, %f28, 0f7F800000;
	@%p3 bra 	$L__BB5_5;

	mov.f32 	%f19, 0f477FE100;
	min.f32 	%f20, %f29, %f19;
	mov.f32 	%f21, 0fC77FE100;
	max.f32 	%f29, %f21, %f20;
	abs.f32 	%f28, %f29;

$L__BB5_5:
	abs.f32 	%f10, %f27;
	setp.gtu.f32 	%p4, %f10, 0f7F800000;
	setp.gtu.f32 	%p5, %f28, 0f7F800000;
	and.pred  	%p6, %p4, %p5;
	@%p6 bra 	$L__BB5_9;

	sub.f32 	%f22, %f27, %f29;
	abs.f32 	%f23, %f22;
	max.f32 	%f24, %f10, %f28;
	add.f32 	%f25, %f24, 0f3F800000;
	div.rn.f32 	%f11, %f23, %f25;
	setp.gt.f32 	%p7, %f11, %f12;
	@%p7 bra 	$L__BB5_8;

	abs.f32 	%f26, %f11;
	setp.le.f32 	%p8, %f26, 0f7F800000;
	@%p8 bra 	$L__BB5_9;

$L__BB5_8:
	cvta.to.global.u64 	%rd11, %rd4;
	atom.global.add.u32 	%r5, [%rd11], 1;

$L__BB5_9:
	ret;

}
	// .globl	__xla_int8_comparison
.visible .entry __xla_int8_comparison(
	.param .u64 __xla_int8_comparison_param_0,
	.param .u64 __xla_int8_comparison_param_1,
	.param .f32 __xla_int8_comparison_param_2,
	.param .u64 __xla_int8_comparison_param_3,
	.param .u64 __xla_int8_comparison_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .b16 	%rs<3>;
	.reg .f32 	%f<12>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd2, [__xla_int8_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_int8_comparison_param_1];
	ld.param.f32 	%f2, [__xla_int8_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_int8_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_int8_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB6_4;

	cvta.to.global.u64 	%rd6, %rd2;
	add.s64 	%rd7, %rd6, %rd1;
	ld.global.s8 	%rs1, [%rd7];
	cvt.rn.f32.s16 	%f3, %rs1;
	cvta.to.global.u64 	%rd8, %rd3;
	add.s64 	%rd9, %rd8, %rd1;
	ld.global.s8 	%rs2, [%rd9];
	cvt.rn.f32.s16 	%f4, %rs2;
	sub.f32 	%f5, %f3, %f4;
	abs.f32 	%f6, %f5;
	abs.f32 	%f7, %f3;
	abs.f32 	%f8, %f4;
	max.f32 	%f9, %f7, %f8;
	add.f32 	%f10, %f9, 0f3F800000;
	div.rn.f32 	%f1, %f6, %f10;
	setp.gt.f32 	%p2, %f1, %f2;
	@%p2 bra 	$L__BB6_3;

	abs.f32 	%f11, %f1;
	setp.le.f32 	%p3, %f11, 0f7F800000;
	@%p3 bra 	$L__BB6_4;

$L__BB6_3:
	cvta.to.global.u64 	%rd10, %rd4;
	atom.global.add.u32 	%r5, [%rd10], 1;

$L__BB6_4:
	ret;

}
	// .globl	__xla_int32_comparison
.visible .entry __xla_int32_comparison(
	.param .u64 __xla_int32_comparison_param_0,
	.param .u64 __xla_int32_comparison_param_1,
	.param .f32 __xla_int32_comparison_param_2,
	.param .u64 __xla_int32_comparison_param_3,
	.param .u64 __xla_int32_comparison_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .f32 	%f<12>;
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<12>;


	ld.param.u64 	%rd2, [__xla_int32_comparison_param_0];
	ld.param.u64 	%rd3, [__xla_int32_comparison_param_1];
	ld.param.f32 	%f2, [__xla_int32_comparison_param_2];
	ld.param.u64 	%rd5, [__xla_int32_comparison_param_3];
	ld.param.u64 	%rd4, [__xla_int32_comparison_param_4];
	mov.u32 	%r1, %ntid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r2, %r1, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd5;
	@%p1 bra 	$L__BB7_4;

	cvta.to.global.u64 	%rd6, %rd2;
	shl.b64 	%rd7, %rd1, 2;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.u32 	%r5, [%rd8];
	cvt.rn.f32.s32 	%f3, %r5;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd7;
	ld.global.u32 	%r6, [%rd10];
	cvt.rn.f32.s32 	%f4, %r6;
	sub.f32 	%f5, %f3, %f4;
	abs.f32 	%f6, %f5;
	abs.f32 	%f7, %f3;
	abs.f32 	%f8, %f4;
	max.f32 	%f9, %f7, %f8;
	add.f32 	%f10, %f9, 0f3F800000;
	div.rn.f32 	%f1, %f6, %f10;
	setp.gt.f32 	%p2, %f1, %f2;
	@%p2 bra 	$L__BB7_3;

	abs.f32 	%f11, %f1;
	setp.le.f32 	%p3, %f11, 0f7F800000;
	@%p3 bra 	$L__BB7_4;

$L__BB7_3:
	cvta.to.global.u64 	%rd11, %rd4;
	atom.global.add.u32 	%r7, [%rd11], 1;

$L__BB7_4:
	ret;

}
)";

template <typename ElementT>
using ComparisonKernelT =
    se::TypedKernel<se::DeviceMemory<ElementT>, se::DeviceMemory<ElementT>,
                    float, uint64_t, se::DeviceMemory<uint64_t>>;

// Compares two buffers on the GPU.
//
// Returns `true` if two buffers are equal, `false` otherwise.
template <typename ElementT>
static StatusOr<bool> DeviceCompare(se::Stream* stream,
                                    se::DeviceMemoryBase current,
                                    se::DeviceMemoryBase expected,
                                    const Shape& buffer_shape,
                                    const HloModuleConfig& config,
                                    absl::string_view kernel_name) {
  se::StreamExecutor* executor = stream->parent();

  se::ScopedDeviceMemory<uint64_t> out_param =
      executor->AllocateOwnedScalar<uint64_t>();

  stream->ThenMemZero(out_param.ptr(), sizeof(uint64_t));
  if (current.size() != expected.size()) {
    return InternalError("Mismatched buffer size: %d bytes vs. %d bytes",
                         current.size(), expected.size());
  }

  se::DeviceMemory<ElementT> current_typed(current);
  se::DeviceMemory<ElementT> expected_typed(expected);
  uint64_t buffer_size = current_typed.ElementCount();

  absl::Span<const uint8_t> compiled_ptx = {};
  StatusOr<absl::Span<const uint8_t>> compiled_ptx_or =
      se::CompileGpuAsmOrGetCached(
          executor->device_ordinal(), buffer_compare_ptx,
          PtxOptsFromDebugOptions(config.debug_options()));
  if (compiled_ptx_or.ok()) {
    compiled_ptx = std::move(compiled_ptx_or).value();
  } else {
    static absl::once_flag ptxas_not_found_logged;
    absl::call_once(ptxas_not_found_logged, [&]() {
      LOG(WARNING)
          << compiled_ptx_or.status().ToString()
          << "\nRelying on driver to perform ptx compilation. "
          << "\nSetting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda "
          << " or modifying $PATH can be used to set the location of ptxas"
          << "\nThis message will only be logged once.";
    });
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ComparisonKernelT<ElementT>> comparison_kernel,
      (executor->CreateTypedKernel<se::DeviceMemory<ElementT>,
                                   se::DeviceMemory<ElementT>, float, uint64_t,
                                   se::DeviceMemory<uint64_t>>(
          kernel_name, buffer_compare_ptx, compiled_ptx)));

  GpuDeviceInfo gpu_device_info;
  gpu_device_info.threads_per_block_limit =
      executor->GetDeviceDescription().threads_per_block_limit();
  gpu_device_info.threads_per_warp =
      executor->GetDeviceDescription().threads_per_warp();
  gpu_device_info.shared_memory_per_block =
      executor->GetDeviceDescription().shared_memory_per_block();
  gpu_device_info.threads_per_core_limit =
      executor->GetDeviceDescription().threads_per_core_limit();
  gpu_device_info.core_count = executor->GetDeviceDescription().core_count();
  gpu_device_info.block_dim_limit_x =
      executor->GetDeviceDescription().block_dim_limit().x;
  gpu_device_info.block_dim_limit_y =
      executor->GetDeviceDescription().block_dim_limit().y;
  gpu_device_info.block_dim_limit_z =
      executor->GetDeviceDescription().block_dim_limit().z;

  TF_ASSIGN_OR_RETURN(LaunchDimensions dim,
                      CalculateLaunchDimensions(buffer_shape, gpu_device_info));

  LaunchDimensions::Dim3D thread_counts = dim.thread_counts_per_block();
  LaunchDimensions::Dim3D block_counts = dim.block_counts();
  TF_RETURN_IF_ERROR(stream->ThenLaunch(
      se::ThreadDim(thread_counts.x, thread_counts.y, thread_counts.z),
      se::BlockDim(block_counts.x, block_counts.y, block_counts.z),
      *comparison_kernel, current_typed, expected_typed,
      static_cast<float>(kTolerance), buffer_size, out_param.cref()));

  uint64_t result = -1;
  CHECK_EQ(out_param->size(), sizeof(result));
  stream->ThenMemcpy(&result, *out_param, sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return result == 0;
}

// Host side comparison code that does the same thing, but reports some of the
// differences as well. It only print logs for debugging.
//
// Returns true if no differences were seen, false otherwise.
template <typename ElementType, typename ComparisonType>
StatusOr<bool> HostCompare(se::Stream* stream, se::DeviceMemoryBase current,
                           se::DeviceMemoryBase expected) {
  int64_t n = current.size() / sizeof(ElementType);
  std::vector<ElementType> host_current(n), host_expected(n);
  stream->ThenMemcpy(host_current.data(), current, current.size());
  stream->ThenMemcpy(host_expected.data(), expected, expected.size());
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  const auto canonicalize = [](ComparisonType a) -> ComparisonType {
    if (std::is_same<ElementType, Eigen::half>::value && a) {
      constexpr ComparisonType kMaxFp16Value = 65505;
      if (std::isnan(a)) {
        return a;
      }
      return std::max(-kMaxFp16Value, std::min(a, kMaxFp16Value));
    }
    return a;
  };
  int differences_seen = 0;
  for (int64_t i = 0; i < n && differences_seen < 10; ++i) {
    auto current_value = static_cast<ComparisonType>(host_current[i]);
    auto expected_value = static_cast<ComparisonType>(host_expected[i]);
    ComparisonType current_value_canonical = canonicalize(current_value);
    ComparisonType expected_value_canonical = canonicalize(expected_value);
    if (std::isnan(current_value_canonical) &&
        std::isnan(expected_value_canonical)) {
      continue;
    }
    if (std::isinf(current_value_canonical) &&
        std::isinf(expected_value_canonical) &&
        current_value_canonical == expected_value_canonical) {
      continue;
    }
    if (std::isfinite(current_value_canonical) !=
            std::isfinite(expected_value_canonical) ||
        !(std::abs(current_value_canonical - expected_value_canonical) /
              (std::max(std::abs(current_value_canonical),
                        std::abs(expected_value_canonical)) +
               1) <
          kTolerance)) {
      ++differences_seen;
      LOG(ERROR) << "Difference at " << i << ": " << current_value
                 << ", expected " << expected_value;
    }
  }
  return differences_seen == 0;
}

template <typename ElementT, typename ComparisonT>
static StatusOr<bool> CompareEqualParameterized(se::Stream* stream,
                                                se::DeviceMemoryBase current,
                                                se::DeviceMemoryBase expected,
                                                const Shape& shape,
                                                const HloModuleConfig& config,
                                                absl::string_view kernel_name) {
  XLA_SCOPED_LOGGING_TIMER("BufferComparator::CompareEqual");
  TF_ASSIGN_OR_RETURN(bool result,
                      DeviceCompare<ElementT>(stream, current, expected, shape,
                                              config, kernel_name));

  if (result) {
    return true;
  }

  TF_ASSIGN_OR_RETURN(bool host_return, (HostCompare<ElementT, ComparisonT>(
                                            stream, current, expected)));
  CHECK_EQ(host_return, result)
      << "Host comparison succeeded even though GPU comparison failed.";

  return false;
}

StatusOr<bool> BufferComparator::CompareEqual(
    se::Stream* stream, se::DeviceMemoryBase current,
    se::DeviceMemoryBase expected) const {
  switch (shape_.element_type()) {
    case xla::F8E4M3FN:
      return CompareEqualParameterized<tsl::float8_e4m3fn, float>(
          stream, current, expected, shape_, config_,
          "__xla_fp8_e4m3fn_comparison");
    case xla::F8E5M2:
      return CompareEqualParameterized<tsl::float8_e5m2, float>(
          stream, current, expected, shape_, config_,
          "__xla_fp8_e5m2_comparison");
    case xla::F16:
      return CompareEqualParameterized<Eigen::half, float>(
          stream, current, expected, shape_, config_, "__xla_fp16_comparison");
    case xla::BF16:
      return CompareEqualParameterized<Eigen::bfloat16, float>(
          stream, current, expected, shape_, config_, "__xla_bf16_comparison");
    case xla::F32:
      return CompareEqualParameterized<float, float>(
          stream, current, expected, shape_, config_, "__xla_fp32_comparison");
    case xla::F64:
      return CompareEqualParameterized<double, double>(
          stream, current, expected, shape_, config_, "__xla_fp64_comparison");
    case xla::S8:
      return CompareEqualParameterized<int8_t, float>(
          stream, current, expected, shape_, config_, "__xla_int8_comparison");
    case xla::S32:
      return CompareEqualParameterized<int32_t, float>(
          stream, current, expected, shape_, config_, "__xla_int32_comparison");
    default:
      return Unimplemented("Unimplemented element type");
  }
}

BufferComparator::BufferComparator(const Shape& shape,
                                   const HloModuleConfig& config)
    : shape_(shape), config_(config) {
  // Normalize complex shapes: since we treat the passed array as a contiguous
  // storage it does not matter which dimension are we doubling.
  auto double_dim_size = [&]() {
    int64_t prev_zero_dim_size = shape_.dimensions(0);
    shape_.set_dimensions(0, prev_zero_dim_size * 2);
  };

  if (shape_.element_type() == PrimitiveType::C64) {
    // C64 is just two F32s next to each other.
    shape_.set_element_type(PrimitiveType::F32);
    double_dim_size();
  } else if (shape_.element_type() == PrimitiveType::C128) {
    // C128 is just two F64s next to each other.
    shape_.set_element_type(PrimitiveType::F64);
    double_dim_size();
  }
}

}  // namespace gpu
}  // namespace xla
