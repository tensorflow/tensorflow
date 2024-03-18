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

#include <string_view>

namespace stream_executor::gpu {

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
std::string_view GetSetIfConditionKernel() {
  return R"(
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
}

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
std::string_view GetSetIfElseConditionKernel() {
  return R"(
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
}

// PTX kernel compiled from:
//
// __global__ void SetCaseCondition(
//     cudaGraphConditionalHandle h0, cudaGraphConditionalHandle h1,
//     cudaGraphConditionalHandle h2, cudaGraphConditionalHandle h3,
//     cudaGraphConditionalHandle h4, cudaGraphConditionalHandle h5,
//     cudaGraphConditionalHandle h6, cudaGraphConditionalHandle h7,
//     int32_t* index, int32_t num_handles) {
//   // Only handles in [0, num_handles) range are valid.
//   //
//   // We can't define a device function with dynamic number of handle
//   // arguments, so we always pass 8 handles, but only some of them are valid.
//   // Size 8 picked as a reasonable (but random) upper bound for what we see
//   // in XLA uses.
//   std::array<cudaGraphConditionalHandle, 8> handles = {h0, h1, h2, h3,
//                                                        h4, h5, h6, h7};

//   // If branch index is out of range activate the last valid handle.
//   int32_t branch_index = *index;
//   if (branch_index < 0 || branch_index >= num_handles) {
//     branch_index = num_handles - 1;
//   }

//   for (int32_t i = 0; i < num_handles; ++i) {
//     if (branch_index == i) {
//       cudaGraphSetConditional(handles[i], 1);
//     } else {
//       cudaGraphSetConditional(handles[i], 0);
//     }
//   }
// }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
std::string_view GetSetCaseConditionKernel() {
  return R"(
.version 4.0
.target sm_50
.address_size 64

.extern .func cudaGraphSetConditional
(
        .param .b64 cudaGraphSetConditional_param_0,
        .param .b32 cudaGraphSetConditional_param_1
)

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
        .param .u32 set_case_condition_param_9
)
{
        .local .align 16 .b8    __local_depot0[64];
        .reg .b64       %SP;
        .reg .b64       %SPL;
        .reg .pred      %p<14>;
        .reg .b32       %r<31>;
        .reg .b64       %rd<27>;
        .loc    1 4 0

        mov.u64         %SPL, __local_depot0;
        ld.param.u64    %rd13, [set_case_condition_param_8];
        ld.param.u32    %r18, [set_case_condition_param_9];
        cvta.to.global.u64      %rd14, %rd13;
        .loc    1 15 3
        add.u64         %rd1, %SPL, 0;
        ld.param.u64    %rd16, [set_case_condition_param_1];
        ld.param.u64    %rd17, [set_case_condition_param_0];
        st.local.v2.u64         [%rd1], {%rd17, %rd16};
        ld.param.u64    %rd18, [set_case_condition_param_3];
        ld.param.u64    %rd19, [set_case_condition_param_2];
        st.local.v2.u64         [%rd1+16], {%rd19, %rd18};
        ld.param.u64    %rd20, [set_case_condition_param_5];
        ld.param.u64    %rd21, [set_case_condition_param_4];
        .loc    1 16 60
        st.local.v2.u64         [%rd1+32], {%rd21, %rd20};
        ld.param.u64    %rd22, [set_case_condition_param_7];
        ld.param.u64    %rd23, [set_case_condition_param_6];
        .loc    1 16 68
        st.local.v2.u64         [%rd1+48], {%rd23, %rd22};
        .loc    1 19 3
        ld.global.u32   %r19, [%rd14];
        .loc    1 20 3
        setp.lt.s32     %p1, %r19, 0;
        setp.ge.s32     %p2, %r19, %r18;
        or.pred         %p3, %p1, %p2;
        .loc    1 21 5
        add.s32         %r1, %r18, -1;
        .loc    1 20 3
        selp.b32        %r2, %r1, %r19, %p3;
        .loc    1 24 3
        setp.lt.s32     %p4, %r18, 1;
        @%p4 bra        $L__BB0_22;

        .loc    1 25 5
        and.b32         %r30, %r18, 3;
        setp.lt.u32     %p5, %r1, 3;
        mov.u32         %r28, 0;
        @%p5 bra        $L__BB0_16;

        sub.s32         %r27, %r18, %r30;
        neg.s32         %r25, %r2;
        mov.u32         %r28, 0;
        mov.u64         %rd25, %rd1;

$L__BB0_3:
        .loc    1 0 0
        ld.local.u64    %rd4, [%rd25];
        .loc    1 25 5
        setp.eq.s32     %p6, %r25, 0;
        @%p6 bra        $L__BB0_5;

        .loc    1 28 7
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
        bra.uni         $L__BB0_6;

$L__BB0_5:
        .loc    1 26 7
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

$L__BB0_6:
        .loc    1 24 40
        add.s32         %r22, %r28, 1;
        .loc    1 25 5
        setp.eq.s32     %p7, %r2, %r22;
        .loc    1 0 0
        ld.local.u64    %rd5, [%rd25+8];
        .loc    1 25 5
        @%p7 bra        $L__BB0_8;
        bra.uni         $L__BB0_7;

$L__BB0_8:
        .loc    1 26 7
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
        bra.uni         $L__BB0_9;

$L__BB0_7:
        .loc    1 28 7
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

$L__BB0_9:
        .loc    1 24 40
        add.s32         %r23, %r28, 2;
        .loc    1 25 5
        setp.eq.s32     %p8, %r2, %r23;
        .loc    1 0 0
        ld.local.u64    %rd6, [%rd25+16];
        .loc    1 25 5
        @%p8 bra        $L__BB0_11;
        bra.uni         $L__BB0_10;

$L__BB0_11:
        .loc    1 26 7
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
        bra.uni         $L__BB0_12;

$L__BB0_10:
        .loc    1 28 7
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

$L__BB0_12:
        .loc    1 24 40
        add.s32         %r24, %r28, 3;
        .loc    1 25 5
        setp.eq.s32     %p9, %r2, %r24;
        .loc    1 0 0
        ld.local.u64    %rd7, [%rd25+24];
        .loc    1 25 5
        @%p9 bra        $L__BB0_14;
        bra.uni         $L__BB0_13;

$L__BB0_14:
        .loc    1 26 7
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
        bra.uni         $L__BB0_15;

$L__BB0_13:
        .loc    1 28 7
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

$L__BB0_15:
        .loc    1 24 40
        add.s64         %rd25, %rd25, 32;
        add.s32         %r28, %r28, 4;
        .loc    1 24 3
        add.s32         %r25, %r25, 4;
        add.s32         %r27, %r27, -4;
        setp.ne.s32     %p10, %r27, 0;
        @%p10 bra       $L__BB0_3;

$L__BB0_16:
        .loc    1 25 5
        setp.eq.s32     %p11, %r30, 0;
        @%p11 bra       $L__BB0_22;

        mul.wide.s32    %rd24, %r28, 8;
        add.s64         %rd26, %rd1, %rd24;
        sub.s32         %r29, %r28, %r2;

$L__BB0_18:
        .pragma "nounroll";
        .loc    1 0 0
        ld.local.u64    %rd11, [%rd26];
        .loc    1 25 5
        setp.eq.s32     %p12, %r29, 0;
        @%p12 bra       $L__BB0_20;

        .loc    1 28 7
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
        bra.uni         $L__BB0_21;

$L__BB0_20:
        .loc    1 26 7
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

$L__BB0_21:
        .loc    1 24 3
        add.s32         %r30, %r30, -1;
        add.s64         %rd26, %rd26, 8;
        add.s32         %r29, %r29, 1;
        setp.ne.s32     %p13, %r30, 0;
        @%p13 bra       $L__BB0_18;

$L__BB0_22:
        .loc    1 31 1
        ret;

})";
}

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
std::string_view GetSetForConditionKernel() {
  return R"(
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
}

std::string_view GetSetWhileConditionKernel() {
  // While condition kernel is the same as an `If` with a single branch.
  return R"(
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
}

}  // namespace stream_executor::gpu
