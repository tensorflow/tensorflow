/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include <string>
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

namespace {

using ::tensorflow::StringPiece;

struct TestData {
  string test_name;
  string module_string;
};

string TestDataToString(const ::testing::TestParamInfo<TestData>& data) {
  return data.param.test_name;
}

// For each string below, we check that:
//  - we parse it to an HloModule successfully, and
//  - the stringification of the resulting HloModule is equal to our original
//    string.
std::vector<TestData> CreateTestCases() {
  // clang-format off
  return std::vector<TestData>({
// ax + y
{
"AxpyParam",
R"(HloModule axpy_module

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}

)"
},
// pred constant
{
"ConstantPred",
R"(HloModule constant_pred_module

ENTRY %constant_pred () -> pred[] {
  ROOT %constant = pred[] constant(true), metadata={op_type="const" op_name="\"it\'s not a problem\n" source_file="path/to/test.cc" source_line=68}, backend_config="foo\" bar"
}

)"
},
// s32 constant
{
"ConstantS32",
R"(HloModule constant_s32_module

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42)
}

)"
},
// f32 constant, but the value is not a decimal and there is a backend
// configuration
{
"ConstantF32",
R"(HloModule ConstantF32_module

ENTRY %ConstantF32.v4 () -> f32[] {
  ROOT %constant = f32[] constant(42), backend_config="this is a configuration"
}

)"
},
// f32 constant, rank 1 empty array.
{
"ConstantF32R1Empty",
R"(HloModule ConstantF32Empty_module

ENTRY %ConstantF32Empty.v4 () -> f32[0] {
  ROOT %constant = f32[0]{0} constant({})
}

)"
},
// f32 constant, rank 4 empty array.
{
"ConstantF32R4Empty",
R"(HloModule ConstantF32R4Empty_module

ENTRY %ConstantF32R4Empty.v4 () -> f32[2,0,4,3] {
  ROOT %constant = f32[2,0,4,3]{3,2,1,0} constant(f32[2,0,4,3] { { /*i0=0*/ }, { /*i0=1*/ } })
}

)"
},
// constant 4D
{
"Constant4D",
R"(HloModule Small_3x2x1x1_module

ENTRY %Small_3x2x1x1.v1 () -> f32[3,2,1,1] {
  ROOT %constant = f32[3,2,1,1]{3,2,1,0} constant(f32[3,2,1,1] { { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
}

)"
},
// non-finite constants: nan, inf, -inf
{
"ConstantNonFinite",
R"(HloModule IsFiniteR1F32s_module

ENTRY %IsFiniteR1F32s.v2 () -> pred[6] {
  %constant = f32[6]{0} constant({nan, 7, nan, -1, inf, -inf})
  ROOT %is-finite = pred[6]{0} is-finite(f32[6]{0} %constant)
}

)"
},
// constant f16
{
"ConstantF16",
R"(HloModule ConstantF16_module

ENTRY %ConstantF16.v4 () -> f16[] {
  ROOT %constant = f16[] constant(500)
}

)"
},
// bf16
{
"BF16",
R"(HloModule BF16

ENTRY %BF16.v4 () -> bf16[] {
  ROOT %constant = bf16[] constant(500)
}

)"
},
// constant + constant
{
"AddConstants",
R"(HloModule add_constants_module

ENTRY %add_constants () -> f32[] {
  %constant = f32[] constant(3.14)
  ROOT %add = f32[] add(f32[] %constant, f32[] %constant)
}

)"
},
// tuple constant
{
"TupleConstant",
R"(HloModule TupleConstant_module

ENTRY %TupleConstant.v1 () -> (f32[2,1], f32[2]) {
  ROOT %constant = (f32[2,1]{1,0}, f32[2]{0}) constant((f32[2,1], f32[2]) ( f32[2,1] { { 1 }, { 2 } }, {2, 42} ))
}

)"
},
// v1 > v2 ? v1 : v2
{
"SelectR1F32",
R"(HloModule SelectR1F32WithCmpR1F32sFromParamsSmall_module

ENTRY %SelectR1F32WithCmpR1F32sFromParamsSmall.v4 (v1: f32[4], v2: f32[4]) -> f32[4] {
  %v1 = f32[4]{0} parameter(0), sharding={maximal device=1}
  %v2 = f32[4]{0} parameter(1), sharding={maximal device=1}
  %greater-than = pred[4]{0} greater-than(f32[4]{0} %v1, f32[4]{0} %v2), sharding={replicated}
  ROOT %select = f32[4]{0} select(pred[4]{0} %greater-than, f32[4]{0} %v1, f32[4]{0} %v2), sharding={}
}

)"
},
// empty tuple
{
"EmptyTupleCreate",
R"(HloModule EmptyTupleCreate_module

ENTRY %EmptyTupleCreate.v1 () -> () {
  ROOT %tuple = () tuple()
}

)"
},
// tuple
{
"TupleCreate",
R"(HloModule TupleCreate_module

ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}

)"
},
{
"ShardedTupleCreate",
R"(HloModule ShardedTupleCreate_module

ENTRY %ShardedTupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3), sharding={{replicated}, {maximal device=0}, {replicated}}
}

)"
},
{
"DomainParsing",
R"(HloModule DomainParsing_module

ENTRY %DomainParsing (v1: f32[]) -> f32[] {
  %v1 = f32[] parameter(0)
  ROOT %dom = f32[] domain(f32[] %v1), domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
}

)"
},
// int32 result = 0;
// while (result < 5) { result = result + 1; }
{
"WhileWithScalarS32Result",
R"(HloModule WhileWithScalarS32Result_module

%body.v3 (prev.1: s32[]) -> s32[] {
  %constant = s32[] constant(1)
  %prev.1 = s32[] parameter(0)
  ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
}

%condition.v3 (prev.2: s32[]) -> pred[] {
  %constant.1 = s32[] constant(5)
  %prev.2 = s32[] parameter(0)
  ROOT %greater-than = pred[] greater-than(s32[] %constant.1, s32[] %prev.2)
}

ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
  %constant.2 = s32[] constant(0)
  ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
}

)"
},
// send and recv
{
"SendRecv",
R"(HloModule TwoSendRecvBothWayRecvFist_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %recv = (f32[], u32[]) recv(), channel_id=15, sharding={maximal device=1}
  ROOT %recv-done = f32[] recv-done((f32[], u32[]) %recv), channel_id=15, sharding={maximal device=1}
  %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = (f32[], u32[]) send(f32[] %constant), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
  %send-done = () send-done((f32[], u32[]) %send), channel_id=16, sharding={maximal device=0}
}

)"
},
// get-tuple-element
{
"GetTupleElement",
R"(HloModule GetTupleElement_module

ENTRY %GetTupleElement.v4 () -> s32[2,3] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = s32[2,3]{1,0} constant(s32[2,3] { { 1, 2, 3 }, { 4, 5, 6 } })
  %tuple = (f32[3]{0}, s32[2,3]{1,0}) tuple(f32[3]{0} %constant, s32[2,3]{1,0} %constant.1)
  ROOT %get-tuple-element = s32[2,3]{1,0} get-tuple-element((f32[3]{0}, s32[2,3]{1,0}) %tuple), index=1, sharding={maximal device=0}
}

)"
},
// call
{
"Call",
R"(HloModule CallR0F32IdentityScalar_module

%Identity.v1 (x: f32[]) -> f32[] {
  ROOT %x = f32[] parameter(0)
}

ENTRY %CallR0F32IdentityScalar.v2 () -> f32[] {
  %constant = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant), to_apply=%Identity.v1
}

)"
},
// reduce window
{
"ReduceWindow",
R"(HloModule R4UnitWindow_module

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %R4UnitWindow.v3 (operand: f32[13,12,8,15]) -> f32[13,3,8,15] {
  %operand = f32[13,12,8,15]{0,3,2,1} parameter(0)
  %constant = f32[] constant(0)
  ROOT %reduce-window = f32[13,3,8,15]{0,3,2,1} reduce-window(f32[13,12,8,15]{0,3,2,1} %operand, f32[] %constant), window={size=1x1x7x1 stride=1x4x1x1 pad=0_0x0_0x3_3x0_0}, to_apply=%add_F32.v3
}

)"
},
// reduce window on scalar
{
"ReduceWindowScalar",
R"(HloModule reduce_window_scalar

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %R4UnitWindowScalar () -> f32[] {
  %constant = f32[] constant(42)
  %constant.1 = f32[] constant(1)
  ROOT %reduce-window = f32[] reduce-window(f32[] %constant, f32[] %constant.1), to_apply=%add_F32.v3
}

)"
},
// convolution
{
"Convolution",
R"(HloModule Convolve1D1Window_0_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f
}

)"
},
// convolution rank 2
{
"ConvolutionR2",
R"(HloModule ConvolveR2_module

ENTRY %ConvolveR2.v3 (input: f32[1,2], filter: f32[1,1]) -> f32[1,2] {
  %input = f32[1,2]{1,0} parameter(0)
  %filter = f32[1,1]{1,0} parameter(1)
  ROOT %convolution = f32[1,2]{0,1} convolution(f32[1,2]{1,0} %input, f32[1,1]{1,0} %filter), dim_labels=bf_io->bf
}

)"
},
// convolution backward
{
"ConvolutionBackward",
R"(HloModule ConvolveBackward_module

ENTRY %ConvolveBackward (input: f32[128,7,7,512], filter: f32[3,3,512,512]) -> f32[128,14,14,512] {
  %input = f32[128,7,7,512]{0,3,2,1} parameter(0)
  %filter = f32[3,3,512,512]{3,2,1,0} parameter(1)
  ROOT %convolution-base-dilated = f32[128,14,14,512]{0,3,2,1} convolution(f32[128,7,7,512]{0,3,2,1} %input, f32[3,3,512,512]{3,2,1,0} %filter), window={size=3x3 pad=1_2x1_2 lhs_dilate=2x2 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f
}

)"
},
// reverse(constant)
{
"Reverse4D",
R"(HloModule Reverse4DFloatArrayOnDim01_module

ENTRY %Reverse4DFloatArrayOnDim01.v2 () -> f32[4,3,2,1] {
  %constant = f32[4,3,2,1]{0,1,2,3} constant(f32[4,3,2,1] { { /*i0=0*/ { /*i1=0*/ {1}, {2} }, { /*i1=1*/ {3}, {4} }, { /*i1=2*/ {5}, {6} } }, { /*i0=1*/ { /*i1=0*/ {7}, {8} }, { /*i1=1*/ {9}, {10} }, { /*i1=2*/ {11}, {12} } }, { /*i0=2*/ { /*i1=0*/ {13}, {14} }, { /*i1=1*/ {15}, {16} }, { /*i1=2*/ {17}, {18} } }, { /*i0=3*/ { /*i1=0*/ {19}, {20} }, { /*i1=1*/ {21}, {22} }, { /*i1=2*/ {23}, {24} } } })
  ROOT %reverse = f32[4,3,2,1]{0,1,2,3} reverse(f32[4,3,2,1]{0,1,2,3} %constant), dimensions={0,1}
}

)"
},
// concat
{
"Concat",
R"(HloModule Concat2x3With2x5_module

ENTRY %Concat2x3With2x5.v3 () -> f32[2,8] {
  %constant = f32[2,3]{1,0} constant(f32[2,3] { { 0, 1, 2 }, { 1000, 1001, 1002 } })
  %constant.1 = f32[2,5]{1,0} constant(f32[2,5] { { 64, 65, 66, 67, 68 }, { 1064, 1065, 1066, 1067, 1068 } })
  ROOT %concatenate = f32[2,8]{1,0} concatenate(f32[2,3]{1,0} %constant, f32[2,5]{1,0} %constant.1), dimensions={1}
}

)"
},
// select and scatter
{
"SelectAndScatter",
R"(HloModule R4F32OverlapSmall_module

%ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %greater-than-or-equal-to = pred[] greater-than-or-equal-to(f32[] %lhs, f32[] %rhs)
}

%add_F32.v3 (lhs.1: f32[], rhs.1: f32[]) -> f32[] {
  %lhs.1 = f32[] parameter(0)
  %rhs.1 = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs.1, f32[] %rhs.1)
}

ENTRY %R4F32OverlapSmall.v4 () -> f32[4,5,1,1] {
  %constant = f32[4,5,1,1]{3,2,1,0} constant(f32[4,5,1,1] { { /*i0=0*/ { /*i1=0*/ {7} }, { /*i1=1*/ {2} }, { /*i1=2*/ {5} }, { /*i1=3*/ {3} }, { /*i1=4*/ {8} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {8} }, { /*i1=2*/ {9} }, { /*i1=3*/ {3} }, { /*i1=4*/ {4} } }, { /*i0=2*/ { /*i1=0*/ {1} }, { /*i1=1*/ {5} }, { /*i1=2*/ {7} }, { /*i1=3*/ {5} }, { /*i1=4*/ {6} } }, { /*i0=3*/ { /*i1=0*/ {0} }, { /*i1=1*/ {6} }, { /*i1=2*/ {2} }, { /*i1=3*/ {10} }, { /*i1=4*/ {2} } } })
  %constant.1 = f32[2,2,1,1]{3,2,1,0} constant(f32[2,2,1,1] { { /*i0=0*/ { /*i1=0*/ {2} }, { /*i1=1*/ {6} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {1} } } })
  %constant.2 = f32[] constant(0)
  ROOT %select-and-scatter = f32[4,5,1,1]{3,2,1,0} select-and-scatter(f32[4,5,1,1]{3,2,1,0} %constant, f32[2,2,1,1]{3,2,1,0} %constant.1, f32[] %constant.2), window={size=2x3x1x1 stride=2x2x1x1}, select=%ge_F32.v3, scatter=%add_F32.v3
}

)"
},
// select and scatter on scalar
{
"SelectAndScatterScalar",
R"(HloModule select_and_scatter_scalar

%ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %greater-than-or-equal-to = pred[] greater-than-or-equal-to(f32[] %lhs, f32[] %rhs)
}

%add_F32.v3 (lhs.1: f32[], rhs.1: f32[]) -> f32[] {
  %lhs.1 = f32[] parameter(0)
  %rhs.1 = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs.1, f32[] %rhs.1)
}

ENTRY %SelectAndScatterScalar () -> f32[] {
  %constant = f32[] constant(42)
  %constant.1 = f32[] constant(1)
  %constant.2 = f32[] constant(2)
  ROOT %select-and-scatter = f32[] select-and-scatter(f32[] %constant, f32[] %constant.1, f32[] %constant.2), select=%ge_F32.v3, scatter=%add_F32.v3
}

)"
},
// slice
{
"Slice",
R"(HloModule slice_module

ENTRY %slice.v2 (p0: f32[3,3,4,4]) -> f32[3,3,2,4] {
  %p0 = f32[3,3,4,4]{3,2,1,0} parameter(0)
  ROOT %slice = f32[3,3,2,4]{3,2,1,0} slice(f32[3,3,4,4]{3,2,1,0} %p0), slice={[0:3:1], [0:3:1], [0:4:2], [0:4:1]}
}

)"
},
// slice, no stride
{
"SliceNoStride",
R"(HloModule Slice3x3x3_To_1x3x3_F32_module

ENTRY %Slice3x3x3_To_1x3x3_F32.v2 () -> f32[1,3,3] {
  %constant = f32[3,3,3]{2,1,0} constant(f32[3,3,3] { { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } }, { { 9, 10, 11 }, { 12, 13, 14 }, { 15, 16, 17 } }, { { 18, 19, 20 }, { 21, 22, 23 }, { 24, 25, 26 } } })
  ROOT %slice = f32[1,3,3]{2,1,0} slice(f32[3,3,3]{2,1,0} %constant), slice={[0:1], [0:3], [0:3]}
}

)"
},
// slice R0
{
"SliceR0",
R"(HloModule SliceR0_module

ENTRY %SliceR0.v2 () -> s32[] {
  %constant = s32[] constant(1)
  ROOT %slice = s32[] slice(s32[] %constant), slice={}
}

)"
},
// transpose
{
"Transpose",
R"(HloModule Transpose_module

ENTRY %Transpose.v2 () -> s32[1,2,3] {
  %constant = s32[1,2,3]{2,1,0} constant(s32[1,2,3] { { { 1, 2, 3 }, { 4, 5, 6 } } })
  ROOT %transpose = s32[1,2,3]{2,1,0} transpose(s32[1,2,3]{2,1,0} %constant), dimensions={0,1,2}
}

)"
},
// Dynamic slice
{
"DynamicSlice",
R"(HloModule DynamicSlice_module

ENTRY %DynamicSlice.v5 (original_parameter: s32[2,2,258], start_index: s32[1]) -> s32[2,2,258] {
  %original_parameter = s32[2,2,258]{2,1,0} parameter(0)
  %constant = s32[1]{0} constant({0})
  %start_index = s32[1]{0} parameter(1)
  %concatenate = s32[3]{0} concatenate(s32[1]{0} %constant, s32[1]{0} %constant, s32[1]{0} %start_index), dimensions={0}
  ROOT %dynamic-slice = s32[2,2,258]{2,1,0} dynamic-slice(s32[2,2,258]{2,1,0} %original_parameter, s32[3]{0} %concatenate), dynamic_slice_sizes={2,2,258}
}

)"
},
// Dynamic update slice
{
"DynamicUpdateSlice",
R"(HloModule DynamicUpdateSlice_module

ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_indices: s32[4]) -> s32[1,1,25,1] {
  %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
  %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
  %start_indices = s32[4]{0} parameter(2)
  ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[4]{0} %start_indices)
}

)"
},
// batch norm training
{
"BatchNormTraining",
R"(HloModule BasicTraining_module

ENTRY %BasicTraining.v4 () -> (f32[2,2,1,2], f32[2], f32[2]) {
  %constant = f32[2,2,1,2]{3,2,1,0} constant(f32[2,2,1,2] { { /*i0=0*/ { /*i1=0*/ {1, 2} }, { /*i1=1*/ {3, 4} } }, { /*i0=1*/ { /*i1=0*/ {5, 6} }, { /*i1=1*/ {7, 8} } } })
  %constant.1 = f32[2]{0} constant({2, 3})
  %constant.2 = f32[2]{0} constant({1, 2})
  ROOT %batch-norm-training = (f32[2,2,1,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[2,2,1,2]{3,2,1,0} %constant, f32[2]{0} %constant.1, f32[2]{0} %constant.2), epsilon=0.001, feature_index=3
}

)"
},
// batch norm inference
{
"BatchNormInference",
R"(HloModule BatchNormInference_module

ENTRY %BatchNormInference.v6 (input: f32[2,2,2,2], offset: f32[2], scale: f32[2], mean: f32[2], variance: f32[2]) -> f32[2,2,2,2] {
  %input = f32[2,2,2,2]{3,2,1,0} parameter(0)
  %offset = f32[2]{0} parameter(1)
  %scale = f32[2]{0} parameter(2)
  %mean = f32[2]{0} parameter(3)
  %variance = f32[2]{0} parameter(4)
  ROOT %batch-norm-inference = f32[2,2,2,2]{3,2,1,0} batch-norm-inference(f32[2,2,2,2]{3,2,1,0} %input, f32[2]{0} %offset, f32[2]{0} %scale, f32[2]{0} %mean, f32[2]{0} %variance), epsilon=0.001, feature_index=0
}

)"
},
// batch norm grad
{
"BatchNormGrad",
R"(HloModule BatchNormGrad_module

ENTRY %BatchNormGrad.v4 (input: f32[2,2,2,2], scale: f32[2], mean: f32[2], variance: f32[2], grad_output: f32[2,2,2,2]) -> (f32[2,2,2,2], f32[2], f32[2]) {
  %input = f32[2,2,2,2]{3,2,1,0} parameter(0)
  %scale = f32[2]{0} parameter(1)
  %mean = f32[2]{0} parameter(2)
  %variance = f32[2]{0} parameter(3)
  %grad_output = f32[2,2,2,2]{3,2,1,0} parameter(4)
  ROOT %batch-norm-grad = (f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad(f32[2,2,2,2]{3,2,1,0} %input, f32[2]{0} %scale, f32[2]{0} %mean, f32[2]{0} %variance, f32[2,2,2,2]{3,2,1,0} %grad_output), epsilon=0.001, feature_index=0
}

)"
},
// fft
{
"Fft",
R"(HloModule Fft_module

ENTRY %Fft (input: c64[8,32]) -> c64[8,32] {
  %input = c64[8,32]{1,0} parameter(0)
  ROOT %fft = c64[8,32]{1,0} fft(c64[8,32]{1,0} %input), fft_type=FFT, fft_length={32}
}

)"
},
// ifft
{
"Ifft2d",
R"(HloModule Ifft2d_module

ENTRY %Ifft2d (input: c64[5,8,32]) -> c64[5,8,32] {
  %input = c64[5,8,32]{2,1,0} parameter(0)
  ROOT %fft = c64[5,8,32]{2,1,0} fft(c64[5,8,32]{2,1,0} %input), fft_type=IFFT, fft_length={8,32}
}

)"
},
// rfft2d
{
"Rfft2d",
R"(HloModule Rfft2d_module

ENTRY %Rfft2d (input: f32[5,64,32]) -> c64[5,64,17] {
  %input = f32[5,64,32]{2,1,0} parameter(0)
  ROOT %fft = c64[5,64,17]{2,1,0} fft(f32[5,64,32]{2,1,0} %input), fft_type=RFFT, fft_length={64,32}
}

)"
},
// irfft3d
{
"Irfft3d",
R"(HloModule Irfft3d_module

ENTRY %Irfft3d (input: c64[5,64,128,33]) -> f32[5,64,128,64] {
  %input = c64[5,64,128,33]{3,2,1,0} parameter(0)
  ROOT %fft = f32[5,64,128,64]{3,2,1,0} fft(c64[5,64,128,33]{3,2,1,0} %input), fft_type=IRFFT, fft_length={64,128,64}
}

)"
},
// pad
{
"Pad",
R"(HloModule Pad1DS3Array_module

ENTRY %Pad1DS3Array.v3 () -> f32[8] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = f32[] constant(0.1)
  ROOT %pad = f32[8]{0} pad(f32[3]{0} %constant, f32[] %constant.1), padding=3_1
}

)"
},
// pad has interior
{
"PadHasInterior",
R"(HloModule PadHasInterior_module

ENTRY %PadHasInterior.v3 (input: f32[1,25,7,7]) -> f32[1,25,17,11] {
  %input = f32[1,25,7,7]{3,2,1,0} parameter(0)
  %constant = f32[] constant(-5.123)
  ROOT %pad = f32[1,25,17,11]{3,2,1,0} pad(f32[1,25,7,7]{3,2,1,0} %input, f32[] %constant), padding=0_0_0x0_0_0x2_2_1x2_2_0
}

)"
},
// Negative padding
{
"PadHasNegativePadding",
R"(HloModule PadHasNegativePadding_module

ENTRY %PadHasNegativePadding (input: f32[1,25,7,7,10]) -> f32[1,15,6,3,29] {
  %input = f32[1,25,7,7,10]{4,3,2,1,0} parameter(0)
  %constant = f32[] constant(-5.123)
  ROOT %pad = f32[1,15,6,3,29]{4,3,2,1,0} pad(f32[1,25,7,7,10]{4,3,2,1,0} %input, f32[] %constant), padding=0_0_0x0_-10_0x0_-1_0x-2_-2_0x-1_-1_3
}

)"
},
// fusion
{
"Fusion",
R"(HloModule fusion_module

%fused_computation (constant.param_0: f32[3,2,1,1], constant.1.param_1: f32[2]) -> f32[3,2,1,1] {
  %constant.param_0 = f32[3,2,1,1]{3,2,1,0} parameter(0)
  %constant.1.param_1 = f32[2]{0} parameter(1)
  %broadcast = f32[3,2,1,1]{3,2,1,0} broadcast(f32[2]{0} %constant.1.param_1), dimensions={1}
  ROOT %subtract = f32[3,2,1,1]{3,2,1,0} subtract(f32[3,2,1,1]{3,2,1,0} %constant.param_0, f32[3,2,1,1]{3,2,1,0} %broadcast)
}

ENTRY %fusion.v3 () -> f32[3,2,1,1] {
  %constant = f32[3,2,1,1]{3,2,1,0} constant(f32[3,2,1,1] { { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
  %constant.1 = f32[2]{0} constant({3.14, 4.25})
  ROOT %fusion = f32[3,2,1,1]{3,2,1,0} fusion(f32[3,2,1,1]{3,2,1,0} %constant, f32[2]{0} %constant.1), kind=kLoop, calls=%fused_computation
}

)"
},
{
"Sparse",
R"(HloModule sparse_f32

ENTRY %sparse () -> f32[2,3,4] {
  ROOT %foo = f32[2,3,4]sparse{10} constant(f32[2,3,4]{[0, 1, 2]: 1, [1, 2, 3]: 2, [2, 3, 4]: 3})
}

)"
},
{
"SparseEmpty",
R"(HloModule sparse_f32_empty

ENTRY %sparse_f32_empty () -> f32[2,3,4] {
  ROOT %foo = f32[2,3,4]sparse{10} constant(f32[2,3,4]{})
}

)"
},
{
"SparseR1",
R"(HloModule sparse_f32_r1

ENTRY %sparse_f32_r1 () -> f32[9] {
  ROOT %foo = f32[9]sparse{10} constant(f32[9]{1: 2, 3: 4, 5: 6})
}

)"
},
{
"gather",
R"(HloModule StringifyGather

ENTRY %Gather (input_tensor: f32[50,49,48,47,46], gather_indices: s64[10,9,8,7,5]) -> f32[10,9,8,7,30,29,28,27,26] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %gather_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %gather_indices), output_window_dims={4,5,6,7,8}, elided_window_dims={}, gather_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, window_bounds={30,29,28,27,26}
}

)"
},
  });
  // clang-format on
}

std::vector<TestData> CreateShortTestCases() {
  // clang-format off
  return std::vector<TestData>({
// map
{
"Map",
R"(HloModule MapBinaryAdder_module

add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY MapBinaryAdder.v3 {
  param0 = f32[4]{0} parameter(0)
  param1 = f32[4]{0} parameter(1)
  ROOT map = f32[4]{0} map(param0, param1), dimensions={0}, to_apply=add_F32.v3
}

)"
},
// reduce
{
"Reduce",
R"(HloModule ReduceR3ToR2_module

add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY ReduceR3ToR2.v3 {
  input = f32[8,16,256]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[8,16]{1,0} reduce(input, constant), dimensions={2}, to_apply=add_F32.v3
}

)"
},
// infeed/outfeed
{
"InfeedOutfeed",
R"(HloModule outfeed_module

ENTRY InfeedToOutfeed {
  token = token[] after-all()
  infeed = ((u32[3]{0}, pred[]), token[]) infeed(token)
  infeed.data = (u32[3]{0}, pred[]) get-tuple-element(infeed), index=0
  outfeed = token[] outfeed(infeed.data, token)
  ROOT infeed.1 = ((u32[3]{0}, pred[]), token[]) infeed(token)
  infeed.1.data = (u32[3]{0}, pred[]) get-tuple-element(infeed.1), index=0
  infeed.1.token = token[] get-tuple-element(infeed.1), index=1
  outfeed.1 = token[] outfeed(infeed.1.data, infeed.1.token)
}

)"
},
// Rng
{
"Rng",
R"(HloModule rng_module

ENTRY Rng {
  constant = f32[] constant(0)
  constant.1 = f32[] constant(1)
  ROOT rng = f32[8]{0} rng(constant, constant.1), distribution=rng_uniform
}

)"
},
// Reduce precision
{
"ReducePrevison",
R"(HloModule reduce_precision

ENTRY ReducePrecision {
  constant = f32[1]{0} constant({3.14159})
  ROOT reduce-precision = f32[1]{0} reduce-precision(constant), exponent_bits=8, mantissa_bits=10
}

)"
},
// Conditional
{
"Conditional",
R"(HloModule conditional

Negate {
  x = f32[] parameter(0)
  ROOT negate = f32[] negate(x)
}

Identity {
  y = f32[] parameter(0)
  ROOT copy = f32[] copy(y)
}

ENTRY Parameters1.v4 {
  constant = pred[] constant(true)
  constant.1 = f32[] constant(56)
  constant.2 = f32[] constant(12)
  ROOT conditional = f32[] conditional(constant, constant.1, constant.2), true_computation=Negate, false_computation=Identity
}

)"
},
// CustomCall
{
"CustomCall",
R"(HloModule custom_call

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar"
}

)"
},
// Variables with non-default names
{
"NonDefaultNames",
R"(HloModule add_constants_module

ENTRY add_constants {
  foo = f32[] constant(3.14)
  ROOT bar = f32[] add(foo, foo)
}

)"
},
{
"Dot",
R"(HloModule dot

ENTRY dot {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[10,3]{1,0} parameter(1)
  ROOT dot = f32[2,3]{1,0} dot(a, b), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

)"
},
{
"gather",
R"(HloModule gather

ENTRY Gather {
  input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  gather_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(input_tensor, gather_indices), output_window_dims={4,5,6,7,8}, elided_window_dims={}, gather_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, window_bounds={30,29,28,27,26}
}

)"
},
// cross-replica-sum
{
"CrossReplicaSum",
R"(HloModule CRS

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT crs = f32[8]{0} cross-replica-sum(input), replica_group_ids={}, to_apply=add
}

)"
},
// cross-replica-sum with subgroups
{
"CrossReplicaSumWithSubgroups",
R"(HloModule CRS_Subgroups

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CrossReplicaSumWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT cross-replica-sum = f32[128,32]{0,1} cross-replica-sum(input), replica_group_ids={0,0,1,1}, barrier="abc", to_apply=add
}

)"
}
  });
  // clang-format on
}

class HloParserTest : public ::testing::Test,
                      public ::testing::WithParamInterface<TestData> {
 protected:
  static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
    EXPECT_TRUE(tensorflow::str_util::StrContains(s, expected))
        << "'" << s << "' does not contain '" << expected << "'";
  }

  // Expects "ToString(ParseHloString(string)) == string", that is, parses the
  // string, asserts that it succeeded, stringifies the parsed module, and
  // checks that the it equals the original string.
  void ExpectEqual() {
    const string& original = GetParam().module_string;
    auto result = ParseHloString(original);
    TF_ASSERT_OK(result.status());
    EXPECT_EQ(original, result.ValueOrDie()->ToString(
                            HloPrintOptions().set_print_large_constants(true)));
  }
};

class HloParserShortTest : public HloParserTest {
 protected:
  void ExpectEqualShort() {
    const string& original = GetParam().module_string;
    auto result = ParseHloString(original);
    TF_ASSERT_OK(result.status());
    EXPECT_EQ(original,
              result.ValueOrDie()->ToString(HloPrintOptions::ShortParsable()));
  }
};

TEST_P(HloParserTest, Run) { ExpectEqual(); }

TEST_P(HloParserShortTest, Run) { ExpectEqualShort(); }

INSTANTIATE_TEST_CASE_P(HloParserTestSuccessInstantiation, HloParserTest,
                        ::testing::ValuesIn(CreateTestCases()),
                        TestDataToString);

INSTANTIATE_TEST_CASE_P(HloParserTestSuccessInstantiation, HloParserShortTest,
                        ::testing::ValuesIn(CreateShortTestCases()),
                        TestDataToString);

TEST_F(HloParserTest, Empty) {
  const string original = "";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, Garbage) {
  const string original = "HloModule thi$ str1ng makes# N0 sen$e @all!*&^%$";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, WrongOpcode) {
  const string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: f32[], y: f32[]) -> f32[] {
  %x = f32[]{} parameter(0)
  %y = f32[]{} parameter(1)
  %le = pred[]{} le(f32[]{} %x, f32[]{} %y)
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, WrongShape) {
  const string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: g32[]) -> g32[] {
  %x = g32[]{} parameter(0)
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, WrongOperandsSize) {
  const string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: f32[]) -> pred[] {
  %x = f32[]{} parameter(0)
  %eq = pred[]{} equal-to(f32[]{} %x)
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, OperandNotFound) {
  const string original = R"(HloModule operand_not_found:
ENTRY %blabla (x: f32[]) -> pred[] {
  %x = f32[]{} parameter(0)
  %eq = pred[]{} equal-to(f32[]{} %x, f32[]{} %y)
}
)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, MoreConstants) {
  const string original = R"(HloModule SelectScalarS32True_module

ENTRY %SelectScalarS32True.v4 () -> s32[] {
  %constant.2 = pred[] constant(true)
  %constant.1 = s32[] constant(-42), sharding={s32[5,6] devices=[2,3]1,2,3,4}
  %constant = s32[] constant(42)
  %select = s32[] select(pred[] %constant.2, s32[] %constant.1, s32[] %constant)
}

)";
  auto result = ParseHloString(original);
  TF_EXPECT_OK(result.status());
  // Constant instructions have no name. The string will be parsed successfully
  // but the constant names will not be exactly the same.
}

TEST_F(HloParserTest, ConfigurationField) {
  const string original = R"(HloModule AModule
ENTRY %configuration_test() -> s32[] {
  %constant = s32[] constant(42), backend_config="foo bar"
})";
  auto result = ParseHloString(original);
  TF_ASSERT_OK(result.status());
  EXPECT_EQ("foo bar", result.ValueOrDie()
                           ->entry_computation()
                           ->root_instruction()
                           ->raw_backend_config_string());
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_1) {
  const string original = R"(HloModule some_2_module

ENTRY %some_2 () -> f32[2] {
  ROOT %constant = f32[2]{0} constant({1,{2}})
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects nested array in rank 1, but sees larger");
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_2) {
  const string original = R"(HloModule some_2x3_module

ENTRY %some_2x3 () -> f32[2,3] {
  ROOT %constant = f32[2,3]{1,0} constant(f32[2,3] {1, 2, 3, 4, 5, 6})
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects nested array in rank 2, but sees 1");
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_3) {
  const string original = R"(HloModule some_2x3x2_module

ENTRY %some_2x3x2 () -> f32[2,3,2] {
  ROOT %constant = f32[2,3,2]{2,1,0} constant(f32[2,3,2] {{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}})
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects 3 elements in the [0]th element");
}

TEST_F(HloParserTest, ConstantF16Overflow) {
  const string original =
      R"(HloModule ConstantF16Overflow_module

ENTRY %ConstantF16Overflow.v4 () -> f16[] {
  ROOT %constant = f16[] constant(-65505)
}

)";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "is out of range for literal's primitive type F16");
}

TEST_F(HloParserTest, ConstantWithExp) {
  const string original = R"(HloModule ConstantWithExp_module

ENTRY %ConstantWithExp.v4 () -> f32[] {
  %constant.1 = f32[] constant(3e+2)
}

)";
  auto result = ParseHloString(original);
  TF_EXPECT_OK(result.status());
  // The string will be parsed successfully but the output strings are not
  // exactly the same, because "3e2" is parsed into value 300 and will be
  // printed as "300".
}

TEST_F(HloParserTest, AttibutesAnyOrder) {
  const string original = R"(HloModule any_order_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), sharding={maximal device=1}, backend_config="foo", dim_labels=b0f_0io->b0f, window={pad=1_1 size=2}
}

)";
  TF_EXPECT_OK(ParseHloString(original).status());
}

TEST_F(HloParserTest, InvalidDimLabels) {
  string prefix = R"(HloModule invalid_dim_labels_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1} )";
  string suffix = R"(
}

)";

  ExpectHasSubstr(ParseHloString(tensorflow::strings::StrCat(
                                     prefix, ",dim_labels=00_01_10", suffix))
                      .status()
                      .error_message(),
                  "expects dim labels pattern");

  ExpectHasSubstr(
      ParseHloString(tensorflow::strings::StrCat(
                         prefix, ",dim_labels=010_1100->010", suffix))
          .status()
          .error_message(),
      "must have the same rank");
}

TEST_F(HloParserTest, UnexpectedAttribute) {
  const string original = R"(HloModule unexpected_attr_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %recv = (f32[], u32[]) recv(), channel_id=15
  %recv-done = f32[] recv-done((f32[], u32[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(2.1)
  %send = (f32[], u32[]) send(f32[] %constant), channel_id=16, calls=%recv
  %send-done = () send-done((f32[], u32[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "unexpected attribute \"calls\"");
}

TEST_F(HloParserTest, MissingAttribute) {
  const string original = R"(HloModule missing_attr_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %recv = (f32[], u32[]) recv(), channel_id=15
  %recv-done = f32[] recv-done((f32[], u32[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(-2.1)
  %send = (f32[], u32[]) send(f32[] %constant)
  %send-done = () send-done((f32[], u32[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "attribute channel_id is expected but not seen");
}

TEST_F(HloParserTest, PredecessorUndefined) {
  const string original = R"(HloModule pre_not_found_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %recv = (f32[], u32[]) recv(), channel_id=15
  %recv-done = f32[] recv-done((f32[], u32[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(2.1)
  %send = (f32[], u32[]) send(f32[] %constant), channel_id=16, control-predecessors={%done}
  %send-done = () send-done((f32[], u32[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "'done' is not defined");
}

TEST_F(HloParserTest, SliceAllowOmitStride1) {
  const string original = R"(HloModule slice_module

ENTRY %slice.v2 (p0: f32[3,3,4,4]) -> f32[3,3,2,4] {
  %p0 = f32[3,3,4,4]{3,2,1,0} parameter(0)
  ROOT %slice = f32[3,3,2,4]{3,2,1,0} slice(f32[3,3,4,4]{3,2,1,0} %p0), slice={[0:3], [0:3], [0:4:2], [0:4]}
}

)";
  TF_EXPECT_OK(ParseHloString(original).status());
}

TEST_F(HloParserTest, PaddingConfigIsNotWindowPad) {
  const string original = R"(HloModule window_pad_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), dim_labels=b0f_0io->b0f, window={pad=1_1_0 size=1}
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "expects padding_low and padding_high separated by '_'");
}

TEST_F(HloParserTest, CommaBetweenSubAttributes) {
  const string original = R"(HloModule test_comma_module

ENTRY %test_comma.v4 () -> f32[] {
  ROOT %constant = f32[] constant(-4.2), metadata={source_line=5, op_type="::const"}
}

)";
  TF_EXPECT_OK(ParseHloString(original).status());
}

TEST_F(HloParserTest, ComputationShapeDoesNotMatchRootShape) {
  const string original = R"(HloModule custom_call:

ENTRY %CustomCall () -> f32[1] {
  %constant = f32[1]{0} constant({12345})
  ROOT %foo = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar"
})";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "Shape of computation CustomCall, f32[1], is not compatible "
                  "with that of its root instruction foo, f32[1,2,3]");
}

TEST_F(HloParserTest, EntryComputationWithLayout) {
  const string original = R"(HloModule layout:
add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY %Reduce (input: f32[8,16,256]) -> f32[8,16] {
  input = f32[8,16,256]{0,1,2} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[8,16]{0,1} reduce(input, constant), dimensions={2}, to_apply=add_F32.v3
})";

  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
  auto program_layout = module.ValueOrDie()->entry_computation_layout();
  ASSERT_EQ(program_layout.parameter_count(), 1);
  auto param_layout = program_layout.parameter_layout(0).layout();
  auto result_layout = program_layout.result_layout().layout();
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1, 2}), param_layout))
      << "actual layout of parameter(0) is "
      << LayoutUtil::HumanString(param_layout);
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}), result_layout))
      << "actual layout of result is "
      << LayoutUtil::HumanString(result_layout);
}

TEST_F(HloParserTest, NoEntry) {
  const string original = R"(HloModule no_entry:
c1 {
  const1 = f32[1]{0} constant({12345})
}
c2 {
  const2 = f32[1]{0} constant({67890})
})";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
  EXPECT_EQ(module.ValueOrDie()->entry_computation()->name(), "c2");
}

TEST_F(HloParserTest, NoRoot) {
  const string original = R"(HloModule no_root:
ENTRY consts {
  first = f32[1]{0} constant({12345})
  last = f32[1]{0} constant({67890})
})";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
  EXPECT_EQ(
      module.ValueOrDie()->entry_computation()->root_instruction()->name(),
      "last");
}

TEST_F(HloParserTest, MultipleEntries) {
  const string original = R"(HloModule multiple_entries:
ENTRY c1 {
  const1 = f32[1]{0} constant({12345})
}
ENTRY c2 {
  const2 = f32[1]{0} constant({67890})
})";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "expects only one ENTRY");
}

TEST_F(HloParserTest, MultipleRoots) {
  const string original = R"(HloModule multiple_roots:
ENTRY consts {
  ROOT const1 = f32[1]{0} constant({12345})
  ROOT const2 = f32[1]{0} constant({12345})
})";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "one computation should have only one ROOT");
}

TEST_F(HloParserTest, ComputationExists) {
  const string original = R"(HloModule comp_exists
comp {
  const1 = f32[1]{0} constant({12345})
}
comp {
  const2 = f32[1]{0} constant({67890})
})";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  R"(was parsing 2:1: error: computation previously defined here
comp {
^)");
}

TEST_F(HloParserTest, CrossComputationLookup) {
  const string original = R"(HloModule cross_computation_lookup:
tcalla (a: (s32[], s32[])) -> (s32[], s32[]) {
  ROOT aparam = (s32[], s32[]) parameter(0)
}

tcallb (b: (s32[], s32[])) -> s32[] {
  rparam = (s32[], s32[]) parameter(0)
  ROOT gte0 = s32[] get-tuple-element(aparam), index=0
}

ENTRY entry {
  param = (s32[], s32[]) parameter(0)
  call0 = (s32[], s32[]) call(param), to_apply=tcalla
  ROOT call1 = s32[] call(param), to_apply=tcallb
})";
  ExpectHasSubstr(
      ParseHloString(original).status().error_message(),
      "was parsing 8:39: error: instruction does not exist: aparam");
}

TEST_F(HloParserTest, ParseSharding) {
  const string original = "{maximal device=42}";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, ParseSharding(original));
  EXPECT_EQ(sharding.ToString(), original);
}

TEST_F(HloParserTest, ParseWindow) {
  Window original = window_util::MakeWindow({1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(Window parsed,
                          ParseWindow(window_util::ToString(original)))
  EXPECT_EQ(window_util::ToString(original), window_util::ToString(parsed));
}

TEST_F(HloParserTest, ParseConvolutionDimensionNumbers) {
  const string original = "b0f_0io->b0f";
  TF_ASSERT_OK_AND_ASSIGN(ConvolutionDimensionNumbers dnums,
                          ParseConvolutionDimensionNumbers(original));
  EXPECT_EQ(original, ConvolutionDimensionNumbersToString(dnums));
}

TEST_F(HloParserTest, NontupleInfeed) {
  const string original = R"(HloModule nontuple_infeed:
ENTRY nontuple_infeed {
  token = token[] after-all()
  ROOT infeed = pred[] infeed(token)
})";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "infeed must have a non-empty tuple shape");
}

}  // namespace
}  // namespace xla
