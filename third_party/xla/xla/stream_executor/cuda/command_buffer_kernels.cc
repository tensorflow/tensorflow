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

#include "xla/stream_executor/cuda/command_buffer_kernels.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_spec.h"

namespace stream_executor::cuda {
namespace {

// Collection of helper kernels required by command buffers on CUDA backends. We
// use pre-compiled PTX instead of a CUDA C++ because conditional nodes require
// CUDA 12.3+ and trying to run with earlier CUDA versions leads to run time
// errors as all CUDA C++ kernels registered in a global static registry and a
// failure to load ONE kernel leads to failure to load ANY kernel at all. We
// should be able to switch to CUDA C++ once the minimum supported CUDA version
// will be larger than 12.3.

// In all kernels defined below we set conditional handle value to `1` when we
// want to execute a CUDA graph tied to it, and to `0` otherwise. For loops, the
// graph will keep being executed until the conditional handle becomes `0`.

// clang-format off
// PTX kernel compiled from:
//
// #include <cuda/std/array>
//
// extern "C"  __global__ void set_case_condition(
//     cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
//     cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3,
//     cudaGraphConditionalHandle h4, cudaGraphConditionalHandle h5,
//     cudaGraphConditionalHandle h6, cudaGraphConditionalHandle h7,
//     uint8_t* index, bool index_is_bool, int32_t index_offset,
//     int32_t num_handles, bool enable_default_condition) {
//   // Only handles in [0, num_handles) range are valid.
//   //
//   // index_offset specifies an offset that will be applied to the index value
//   // to determine which conditional handle to set:
//   //
//   // effective_index = index - index_offset
//   // handle[effective_index] = 1
//   //
//   // When enable_default_condition is true, the handle[num_handles-1] is set
//   // for any index < 0 or effective_index >= num_handles.
//   //
//   // We can't define a device function with dynamic number of handle
//   // arguments, so we always pass 8 handles, but only some of them are valid.
//   // Size 8 picked as a reasonable (but random) upper bound for what we see
//   // in XLA uses.
//   cuda::std::array<cudaGraphConditionalHandle, 8> handles = {h0, h1, h2, h3,
//                                                              h4, h5, h6, h7};
//
//   // If branch index is out of range activate the last valid handle.
//   int32_t index_int32;
//   if (index_is_bool) {
//     index_int32 = static_cast<int32_t>(*reinterpret_cast<bool*>(index));
//   } else {
//     index_int32 = *reinterpret_cast<int32_t*>(index);
//   }
//
//   int32_t effective_index = index_int32 - index_offset;
//   if (enable_default_condition &&
//       (index_int32 < 0 || effective_index >= num_handles)) {
//     effective_index = num_handles - 1;
//   }
//
//   for (int32_t i = 0; i < num_handles; ++i) {
//     if (effective_index == i) {
//       cudaGraphSetConditional(handles[i], 1);
//
//     } else {
//       cudaGraphSetConditional(handles[i], 0);
//     }
//   }
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
// May have to include these compiler options: -arch sm_50
// clang-format on
inline constexpr absl::string_view kSetCaseConditionKernel = R"(
.version 8.3
.target sm_50
.address_size 64

	// .globl	set_case_condition
.extern .func cudaGraphSetConditional
(
	.param .b64 cudaGraphSetConditional_param_0,
	.param .b32 cudaGraphSetConditional_param_1
)
;
.global .align 1 .b8 _ZN37_INTERNAL_d9baebcd_7_case_cu_50b664ab4cuda3std3__48in_placeE[1];
.global .align 1 .b8 _ZN37_INTERNAL_d9baebcd_7_case_cu_50b664ab4cuda3std6ranges3__45__cpo4swapE[1];
.global .align 1 .b8 _ZN37_INTERNAL_d9baebcd_7_case_cu_50b664ab4cuda3std6ranges3__45__cpo9iter_moveE[1];

.visible .entry set_case_condition(
	.param .u64 set_case_condition_param_0,
	.param .u64 set_case_condition_param_1,
	.param .u64 set_case_condition_param_2,
	.param .u64 set_case_condition_param_3,
	.param .u64 set_case_condition_param_4,
	.param .u64 set_case_condition_param_5,
	.param .u64 set_case_condition_param_6,
	.param .u64 set_case_condition_param_7,
	.param .u64 set_case_condition_param_8,
	.param .u8 set_case_condition_param_9,
	.param .u32 set_case_condition_param_10,
	.param .u32 set_case_condition_param_11,
	.param .u8 set_case_condition_param_12
)
{
	.local .align 8 .b8 	__local_depot0[64];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<17>;
	.reg .b16 	%rs<3>;
	.reg .b32 	%r<36>;
	.reg .b64 	%rd<27>;


	mov.u64 	%SPL, __local_depot0;
	ld.param.s8 	%rs1, [set_case_condition_param_12];
	ld.param.s8 	%rs2, [set_case_condition_param_9];
	ld.param.u64 	%rd14, [set_case_condition_param_0];
	ld.param.u64 	%rd15, [set_case_condition_param_1];
	ld.param.u64 	%rd16, [set_case_condition_param_2];
	ld.param.u64 	%rd17, [set_case_condition_param_3];
	ld.param.u64 	%rd18, [set_case_condition_param_4];
	ld.param.u64 	%rd19, [set_case_condition_param_5];
	ld.param.u64 	%rd20, [set_case_condition_param_6];
	ld.param.u64 	%rd21, [set_case_condition_param_7];
	ld.param.u64 	%rd22, [set_case_condition_param_8];
	ld.param.u32 	%r21, [set_case_condition_param_10];
	ld.param.u32 	%r22, [set_case_condition_param_11];
	cvta.to.global.u64 	%rd2, %rd22;
	add.u64 	%rd1, %SPL, 0;
	st.local.u64 	[%rd1], %rd14;
	st.local.u64 	[%rd1+8], %rd15;
	st.local.u64 	[%rd1+16], %rd16;
	st.local.u64 	[%rd1+24], %rd17;
	st.local.u64 	[%rd1+32], %rd18;
	st.local.u64 	[%rd1+40], %rd19;
	st.local.u64 	[%rd1+48], %rd20;
	st.local.u64 	[%rd1+56], %rd21;
	setp.eq.s16 	%p1, %rs2, 0;
	@%p1 bra 	$L__BB0_2;

	ld.global.s8 	%r29, [%rd2];
	bra.uni 	$L__BB0_3;

$L__BB0_2:
	ld.global.u32 	%r29, [%rd2];

$L__BB0_3:
	sub.s32 	%r23, %r29, %r21;
	setp.ge.s32 	%p2, %r23, %r22;
	setp.lt.s32 	%p3, %r29, 0;
	or.pred  	%p4, %p3, %p2;
	setp.ne.s16 	%p5, %rs1, 0;
	and.pred  	%p6, %p5, %p4;
	add.s32 	%r4, %r22, -1;
	selp.b32 	%r5, %r4, %r23, %p6;
	setp.lt.s32 	%p7, %r22, 1;
	@%p7 bra 	$L__BB0_25;

	and.b32  	%r35, %r22, 3;
	setp.lt.u32 	%p8, %r4, 3;
	mov.u32 	%r33, 0;
	@%p8 bra 	$L__BB0_19;

	sub.s32 	%r32, %r22, %r35;
	neg.s32 	%r30, %r5;
	mov.u32 	%r33, 0;
	mov.u64 	%rd25, %rd1;

$L__BB0_6:
	ld.local.u64 	%rd5, [%rd25];
	setp.eq.s32 	%p9, %r30, 0;
	@%p9 bra 	$L__BB0_8;

	{ // callseq 0, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd5;
	.param .b32 param1;
	st.param.b32 	[param1+0], 0;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 0
	bra.uni 	$L__BB0_9;

$L__BB0_8:
	{ // callseq 1, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd5;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 1

$L__BB0_9:
	add.s32 	%r26, %r33, 1;
	setp.eq.s32 	%p10, %r5, %r26;
	ld.local.u64 	%rd6, [%rd25+8];
	@%p10 bra 	$L__BB0_11;
	bra.uni 	$L__BB0_10;

$L__BB0_11:
	{ // callseq 3, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd6;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 3
	bra.uni 	$L__BB0_12;

$L__BB0_10:
	{ // callseq 2, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd6;
	.param .b32 param1;
	st.param.b32 	[param1+0], 0;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 2

$L__BB0_12:
	add.s32 	%r27, %r33, 2;
	setp.eq.s32 	%p11, %r5, %r27;
	ld.local.u64 	%rd7, [%rd25+16];
	@%p11 bra 	$L__BB0_14;
	bra.uni 	$L__BB0_13;

$L__BB0_14:
	{ // callseq 5, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd7;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 5
	bra.uni 	$L__BB0_15;

$L__BB0_13:
	{ // callseq 4, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd7;
	.param .b32 param1;
	st.param.b32 	[param1+0], 0;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 4

$L__BB0_15:
	add.s32 	%r28, %r33, 3;
	setp.eq.s32 	%p12, %r5, %r28;
	ld.local.u64 	%rd8, [%rd25+24];
	@%p12 bra 	$L__BB0_17;
	bra.uni 	$L__BB0_16;

$L__BB0_17:
	{ // callseq 7, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd8;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 7
	bra.uni 	$L__BB0_18;

$L__BB0_16:
	{ // callseq 6, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd8;
	.param .b32 param1;
	st.param.b32 	[param1+0], 0;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 6

$L__BB0_18:
	add.s64 	%rd25, %rd25, 32;
	add.s32 	%r33, %r33, 4;
	add.s32 	%r30, %r30, 4;
	add.s32 	%r32, %r32, -4;
	setp.ne.s32 	%p13, %r32, 0;
	@%p13 bra 	$L__BB0_6;

$L__BB0_19:
	setp.eq.s32 	%p14, %r35, 0;
	@%p14 bra 	$L__BB0_25;

	mul.wide.s32 	%rd24, %r33, 8;
	add.s64 	%rd26, %rd1, %rd24;
	sub.s32 	%r34, %r33, %r5;

$L__BB0_21:
	.pragma "nounroll";
	ld.local.u64 	%rd12, [%rd26];
	setp.eq.s32 	%p15, %r34, 0;
	@%p15 bra 	$L__BB0_23;

	{ // callseq 8, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32 	[param1+0], 0;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 8
	bra.uni 	$L__BB0_24;

$L__BB0_23:
	{ // callseq 9, 0
	.reg .b32 temp_param_reg;
	.param .b64 param0;
	st.param.b64 	[param0+0], %rd12;
	.param .b32 param1;
	st.param.b32 	[param1+0], 1;
	call.uni
	cudaGraphSetConditional,
	(
	param0,
	param1
	);
	} // callseq 9

$L__BB0_24:
	add.s32 	%r35, %r35, -1;
	add.s64 	%rd26, %rd26, 8;
	add.s32 	%r34, %r34, 1;
	setp.ne.s32 	%p16, %r35, 0;
	@%p16 bra 	$L__BB0_21;

$L__BB0_25:
	ret;

})";

// While condition kernel is the same as an `If` with a single branch.
inline constexpr absl::string_view kSetWhileConditionKernel = R"(
.version 4.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional
(
        .param .b64 cudaGraphSetConditional_param_0,
        .param .b32 cudaGraphSetConditional_param_1
)

.visible .entry set_while_condition(
        .param .u64 set_while_condition_param_0,
        .param .u64 set_while_condition_param_1
)
{
        .reg .pred      %p<2>;
        .reg .b16       %rs<2>;
        .reg .b64       %rd<4>;
        .loc    1 1 0

        ld.param.u64    %rd1, [set_while_condition_param_0];
        ld.param.u64    %rd2, [set_while_condition_param_1];
        .loc    1 3 3
        cvta.to.global.u64      %rd3, %rd2;
        ld.global.u8    %rs1, [%rd3];
        setp.eq.s16     %p1, %rs1, 0;
        @%p1 bra        $L__BB0_2;

        .loc    1 4 5
        { // callseq 0, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd1;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni
        cudaGraphSetConditional,
        (
        param0,
        param1
        );
        } // callseq 0
        bra.uni         $L__BB0_3;

$L__BB0_2:
        .loc    1 6 5
        { // callseq 1, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd1;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni
        cudaGraphSetConditional,
        (
        param0,
        param1
        );
        } // callseq 1

$L__BB0_3:
        .loc    1 8 1
        ret;

})";

// PTX kernel compiled from:
//
//  __global__ void noop() {}
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr absl::string_view kNoOpKernel = R"(
.version 4.0
.target sm_50
.address_size 64

.visible .entry noop()
{

        .loc    1 1 0

        .loc    1 4 1
        ret;

})";

}  // namespace

absl::StatusOr<KernelLoaderSpec> GetSetCaseConditionKernelLoaderSpec() {
  return KernelLoaderSpec::CreateCudaPtxInMemorySpec(
      cuda::kSetCaseConditionKernel, "set_case_condition", 13);
}

absl::StatusOr<KernelLoaderSpec> GetSetWhileConditionKernelLoaderSpec() {
  return KernelLoaderSpec::CreateCudaPtxInMemorySpec(
      cuda::kSetWhileConditionKernel, "set_while_condition", 2);
}

absl::StatusOr<KernelLoaderSpec> GetNoOpKernelLoaderSpec() {
  return KernelLoaderSpec::CreateCudaPtxInMemorySpec(cuda::kNoOpKernel, "noop",
                                                     0);
}

}  // namespace stream_executor::cuda
