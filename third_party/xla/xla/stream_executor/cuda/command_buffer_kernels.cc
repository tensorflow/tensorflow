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

namespace stream_executor {
namespace cuda {
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

// PTX kernel compiled from:
//
// __global__ void SetIfCondition(cudaGraphConditionalHandle then_handle,
//                                bool* predicate) {
//   if (*predicate) {
//     cudaGraphSetConditional(then_handle, 1);
//   } else {
//     cudaGraphSetConditional(then_handle, 0);
//   }
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr absl::string_view kSetIfConditionKernel = R"(
.version 4.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional
(
        .param .b64 cudaGraphSetConditional_param_0,
        .param .b32 cudaGraphSetConditional_param_1
)

.visible .entry set_if_condition(
        .param .u64 set_if_condition_param_0,
        .param .u64 set_if_condition_param_1
)
{
        .reg .pred      %p<2>;
        .reg .b16       %rs<2>;
        .reg .b64       %rd<4>;
        .loc    1 1 0

        ld.param.u64    %rd1, [set_if_condition_param_0];
        ld.param.u64    %rd2, [set_if_condition_param_1];
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
// __global__ void SetIfElseCondition(cudaGraphConditionalHandle then_handle,
//                                    cudaGraphConditionalHandle else_handle,
//                                    bool* predicate) {
//   if (*predicate) {
//     cudaGraphSetConditional(then_handle, 1);
//     cudaGraphSetConditional(else_handle, 0);
//   } else {
//     cudaGraphSetConditional(then_handle, 0);
//     cudaGraphSetConditional(else_handle, 1);
//   }
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr absl::string_view kSetIfElseConditionKernel = R"(
.version 4.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional
(
        .param .b64 cudaGraphSetConditional_param_0,
        .param .b32 cudaGraphSetConditional_param_1
)

.visible .entry set_if_else_condition(
        .param .u64 set_if_else_condition_param_0,
        .param .u64 set_if_else_condition_param_1,
        .param .u64 set_if_else_condition_param_2
)
{
        .reg .pred      %p<2>;
        .reg .b16       %rs<2>;
        .reg .b64       %rd<5>;
        .loc    1 1 0

        ld.param.u64    %rd1, [set_if_else_condition_param_0];
        ld.param.u64    %rd2, [set_if_else_condition_param_1];
        ld.param.u64    %rd3, [set_if_else_condition_param_2];
        .loc    1 4 3
        cvta.to.global.u64      %rd4, %rd3;
        ld.global.u8    %rs1, [%rd4];
        setp.eq.s16     %p1, %rs1, 0;
        @%p1 bra        $L__BB0_2;

        .loc    1 5 5
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
        .loc    1 6 5
        { // callseq 1, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd2;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni
        cudaGraphSetConditional,
        (
        param0,
        param1
        );
        } // callseq 1
        bra.uni         $L__BB0_3;

$L__BB0_2:
        .loc    1 8 5
        { // callseq 2, 0
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
        } // callseq 2
        .loc    1 9 5
        { // callseq 3, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd2;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni
        cudaGraphSetConditional,
        (
        param0,
        param1
        );
        } // callseq 3

$L__BB0_3:
        .loc    1 11 1
        ret;

})";

// PTX kernel compiled from:
//
// #include <cuda/std/array>
//
// __global__ void set_case_condition(
//     cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
//     cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3,
//     cudaGraphConditionalHandle h4, cudaGraphConditionalHandle h5,
//     cudaGraphConditionalHandle h6, cudaGraphConditionalHandle h7,
//     int32_t* index, int32_t index_offset, int32_t num_handles,
//     bool enable_default_condition) {
//   // Only handles in [0, num_handles) range are valid.
//   //
//   // index_offset specifies an offset that will be applied to the index value
//   to
//   // determine which conditional handle to set:
//   //
//   // effective_index = index - index_offset
//   // handle[effective_index] = 1
//   //
//   // When enable_default_condition is true, the handle[num_handles-1] is set
//   for
//   // any index < 0 or effective_index >= num_handles.
//   //
//   // We can't define a device function with dynamic number of handle
//   arguments,
//   // so we always pass 8 handles, but only some of them are valid. Size 8
//   picked
//   // as a reasonable (but random) upper bound for what we see in XLA uses.
//   cuda::std::array<cudaGraphConditionalHandle, 8> handles = {h0, h1, h2, h3,
//                                                        h4, h5, h6, h7};
//
//   // If branch index is out of range activate the last valid handle.
//   int32_t effective_index = *index - index_offset;
//   if (enable_default_condition &&
//       (*index < 0 || effective_index >= num_handles)) {
//     effective_index = num_handles - 1;
//   }
//
//   for (int32_t i = 0; i < num_handles; ++i) {
//     if (effective_index == i) {
//       cudaGraphSetConditional(handles[i], 1);
//     } else {
//       cudaGraphSetConditional(handles[i], 0);
//     }
//   }
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
// May have to include these compiler options: -arch sm_50
inline constexpr absl::string_view kSetCaseConditionKernel = R"(
.version 4.0
.target sm_50
.address_size 64

       // .globl       set_case_condition
.extern .func cudaGraphSetConditional
(
        .param .b64 cudaGraphSetConditional_param_0,
        .param .b32 cudaGraphSetConditional_param_1
)
;
.global .align 1 .b8 _ZN41_INTERNAL_1c7773a2_10_example_cu_50b664ab4cuda3std3__48in_placeE[1];
.global .align 1 .b8 _ZN41_INTERNAL_1c7773a2_10_example_cu_50b664ab4cuda3std6ranges3__45__cpo4swapE[1];

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
        .param .u32 set_case_condition_param_9,
        .param .u32 set_case_condition_param_10,
        .param .u8 set_case_condition_param_11
)
{
        .local .align 8 .b8     __local_depot0[64];
        .reg .b64       %SP;
        .reg .b64       %SPL;
        .reg .pred      %p<18>;
        .reg .b16       %rs<2>;
        .reg .b32       %r<33>;
        .reg .b64       %rd<27>;
        .loc    1 3 0

        mov.u64         %SPL, __local_depot0;
        ld.param.s8     %rs1, [set_case_condition_param_11];
        ld.param.u64    %rd13, [set_case_condition_param_0];
        ld.param.u64    %rd14, [set_case_condition_param_1];
        ld.param.u64    %rd15, [set_case_condition_param_2];
        ld.param.u64    %rd16, [set_case_condition_param_3];
        ld.param.u64    %rd17, [set_case_condition_param_4];
        ld.param.u64    %rd18, [set_case_condition_param_5];
        ld.param.u64    %rd19, [set_case_condition_param_6];
        ld.param.u64    %rd20, [set_case_condition_param_7];
        ld.param.u64    %rd21, [set_case_condition_param_8];
        ld.param.u32    %r21, [set_case_condition_param_9];
        ld.param.u32    %r20, [set_case_condition_param_10];
        cvta.to.global.u64      %rd22, %rd21;
        .loc    1 24 3
        add.u64         %rd1, %SPL, 0;
        st.local.u64    [%rd1], %rd13;
        st.local.u64    [%rd1+8], %rd14;
        st.local.u64    [%rd1+16], %rd15;
        st.local.u64    [%rd1+24], %rd16;
        .loc    1 25 56
        st.local.u64    [%rd1+32], %rd17;
        .loc    1 25 60
        st.local.u64    [%rd1+40], %rd18;
        .loc    1 25 64
        st.local.u64    [%rd1+48], %rd19;
        .loc    1 25 68
        st.local.u64    [%rd1+56], %rd20;
        .loc    1 28 3
        ld.global.u32   %r1, [%rd22];
        sub.s32         %r2, %r1, %r21;
        .loc    1 29 3
        setp.eq.s16     %p4, %rs1, 0;
        mov.pred        %p17, 0;
        @%p4 bra        $L__BB0_2;

        .loc    1 30 8
        setp.lt.s32     %p5, %r1, 0;
        setp.ge.s32     %p6, %r2, %r20;
        or.pred         %p17, %p6, %p5;

$L__BB0_2:
        .loc    1 31 5
        add.s32         %r3, %r20, -1;
        .loc    1 30 8
        selp.b32        %r4, %r3, %r2, %p17;
        .loc    1 34 3
        setp.lt.s32     %p7, %r20, 1;
        @%p7 bra        $L__BB0_24;

        .loc    1 35 5
        and.b32         %r32, %r20, 3;
        setp.lt.u32     %p8, %r3, 3;
        mov.u32         %r30, 0;
        @%p8 bra        $L__BB0_18;

        sub.s32         %r29, %r20, %r32;
        neg.s32         %r27, %r4;
        mov.u32         %r30, 0;
        mov.u64         %rd25, %rd1;

$L__BB0_5:
        .loc    1 0 0
        ld.local.u64    %rd4, [%rd25];
        .loc    1 35 5
        setp.eq.s32     %p9, %r27, 0;
        @%p9 bra        $L__BB0_7;

        .loc    1 38 7
        { // callseq 0, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd4;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 0
        bra.uni         $L__BB0_8;

$L__BB0_7:
        .loc    1 36 7
        { // callseq 1, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd4;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 1

$L__BB0_8:
        .loc    1 34 40
        add.s32         %r24, %r30, 1;
        .loc    1 35 5
        setp.eq.s32     %p10, %r4, %r24;
        .loc    1 0 0
        ld.local.u64    %rd5, [%rd25+8];
        .loc    1 35 5
        @%p10 bra       $L__BB0_10;
        bra.uni         $L__BB0_9;

$L__BB0_10:
        .loc    1 36 7
        { // callseq 3, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd5;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 3
        bra.uni         $L__BB0_11;

$L__BB0_9:
        .loc    1 38 7
        { // callseq 2, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd5;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 2

$L__BB0_11:
        .loc    1 34 40
        add.s32         %r25, %r30, 2;
        .loc    1 35 5
        setp.eq.s32     %p11, %r4, %r25;
        .loc    1 0 0
        ld.local.u64    %rd6, [%rd25+16];
        .loc    1 35 5
        @%p11 bra       $L__BB0_13;
        bra.uni         $L__BB0_12;

$L__BB0_13:
        .loc    1 36 7
        { // callseq 5, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd6;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 5
        bra.uni         $L__BB0_14;

$L__BB0_12:
        .loc    1 38 7
        { // callseq 4, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd6;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 4

$L__BB0_14:
        .loc    1 34 40
        add.s32         %r26, %r30, 3;
        .loc    1 35 5
        setp.eq.s32     %p12, %r4, %r26;
        .loc    1 0 0
        ld.local.u64    %rd7, [%rd25+24];
        .loc    1 35 5
        @%p12 bra       $L__BB0_16;
        bra.uni         $L__BB0_15;

$L__BB0_16:
        .loc    1 36 7
        { // callseq 7, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd7;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 7
        bra.uni         $L__BB0_17;

$L__BB0_15:
        .loc    1 38 7
        { // callseq 6, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd7;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 6

$L__BB0_17:
        .loc    1 34 40
        add.s64         %rd25, %rd25, 32;
        add.s32         %r30, %r30, 4;
        .loc    1 34 3
        add.s32         %r27, %r27, 4;
        add.s32         %r29, %r29, -4;
        setp.ne.s32     %p13, %r29, 0;
        @%p13 bra       $L__BB0_5;

$L__BB0_18:
        .loc    1 35 5
        setp.eq.s32     %p14, %r32, 0;
        @%p14 bra       $L__BB0_24;

        mul.wide.s32    %rd24, %r30, 8;
        add.s64         %rd26, %rd1, %rd24;
        sub.s32         %r31, %r30, %r4;

$L__BB0_20:
        .pragma "nounroll";
        .loc    1 0 0
        ld.local.u64    %rd11, [%rd26];
        .loc    1 35 5
        setp.eq.s32     %p15, %r31, 0;
        @%p15 bra       $L__BB0_22;

        .loc    1 38 7
        { // callseq 8, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd11;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 8
        bra.uni         $L__BB0_23;

$L__BB0_22:
        .loc    1 36 7
        { // callseq 9, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd11;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni 
        cudaGraphSetConditional, 
        (
        param0, 
        param1
        );
        } // callseq 9

$L__BB0_23:
        .loc    1 34 3
        add.s32         %r32, %r32, -1;
        add.s64         %rd26, %rd26, 8;
        add.s32         %r31, %r31, 1;
        setp.ne.s32     %p16, %r32, 0;
        @%p16 bra       $L__BB0_20;

$L__BB0_24:
        .loc    1 41 1
        ret;

})";

// PTX kernel compiled from:
//
// __global__ void SetForCondition(cudaGraphConditionalHandle handle,
//                                 int32_t* loop_index,
//                                 int32_t num_iterations) {
//   if (*loop_index < num_iterations) {
//     cudaGraphSetConditional(handle, 1);
//   } else {
//     cudaGraphSetConditional(handle, 0);
//   }
//   *loop_index += 1;
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr absl::string_view kSetForConditionKernel = R"(
.version 4.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional
(
        .param .b64 cudaGraphSetConditional_param_0,
        .param .b32 cudaGraphSetConditional_param_1
)

.visible .entry set_for_condition(
        .param .u64 set_for_condition_param_0,
        .param .u64 set_for_condition_param_1,
        .param .u32 set_for_condition_param_2
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<5>;
        .reg .b64       %rd<4>;
        .loc    1 1 0

        ld.param.u64    %rd2, [set_for_condition_param_0];
        ld.param.u64    %rd3, [set_for_condition_param_1];
        ld.param.u32    %r1, [set_for_condition_param_2];
        .loc    1 3 3
        cvta.to.global.u64      %rd1, %rd3;
        ld.global.u32   %r2, [%rd1];
        setp.lt.s32     %p1, %r2, %r1;
        @%p1 bra        $L__BB0_2;
        bra.uni         $L__BB0_1;

$L__BB0_2:
        .loc    1 4 5
        { // callseq 1, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd2;
        .param .b32 param1;
        st.param.b32    [param1+0], 1;
        call.uni
        cudaGraphSetConditional,
        (
        param0,
        param1
        );
        } // callseq 1
        bra.uni         $L__BB0_3;

$L__BB0_1:
        .loc    1 6 5
        { // callseq 0, 0
        .reg .b32 temp_param_reg;
        .param .b64 param0;
        st.param.b64    [param0+0], %rd2;
        .param .b32 param1;
        st.param.b32    [param1+0], 0;
        call.uni
        cudaGraphSetConditional,
        (
        param0,
        param1
        );
        } // callseq 0

$L__BB0_3:
        .loc    1 8 3
        ld.global.u32   %r3, [%rd1];
        add.s32         %r4, %r3, 1;
        st.global.u32   [%rd1], %r4;
        .loc    1 9 1
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

absl::StatusOr<MultiKernelLoaderSpec> GetSetIfConditionKernelLoaderSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/2);
  spec.AddCudaPtxInMemory(cuda::kSetIfConditionKernel, "set_if_condition");
  return spec;
}

absl::StatusOr<MultiKernelLoaderSpec> GetSetIfElseConditionKernelLoaderSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddCudaPtxInMemory(cuda::kSetIfElseConditionKernel,
                          "set_if_else_condition");
  return spec;
}

absl::StatusOr<MultiKernelLoaderSpec> GetSetCaseConditionKernelLoaderSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/12);
  spec.AddCudaPtxInMemory(cuda::kSetCaseConditionKernel, "set_case_condition");
  return spec;
}

absl::StatusOr<MultiKernelLoaderSpec> GetSetForConditionKernelLoaderSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddCudaPtxInMemory(cuda::kSetForConditionKernel, "set_for_condition");
  return spec;
}

absl::StatusOr<MultiKernelLoaderSpec> GetSetWhileConditionKernelLoaderSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/2);
  spec.AddCudaPtxInMemory(cuda::kSetWhileConditionKernel,
                          "set_while_condition");
  return spec;
}

absl::StatusOr<MultiKernelLoaderSpec> GetNoOpKernelLoaderSpec() {
  MultiKernelLoaderSpec spec(/*arity=*/0);
  spec.AddCudaPtxInMemory(cuda::kNoOpKernel, "noop");
  return spec;
}

}  // namespace cuda
}  // namespace stream_executor
