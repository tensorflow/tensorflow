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
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

namespace m = ::xla::match;
using absl::string_view;

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
// pred array constant
{
"ConstantPredArray",
R"(HloModule module

ENTRY %constant_pred_array () -> pred[2,3] {
  ROOT %constant = pred[2,3]{1,0} constant({ { 0, 1, 0 }, { 1, 0, 1 } })
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
  ROOT %constant = f32[2,0,4,3]{3,2,1,0} constant({ { /*i0=0*/ }, { /*i0=1*/ } })
}

)"
},
// constant 4D
{
"Constant4D",
R"(HloModule Small_3x2x1x1_module

ENTRY %Small_3x2x1x1.v1 () -> f32[3,2,1,1] {
  ROOT %constant = f32[3,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
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
  ROOT %constant = (f32[2,1]{1,0}, f32[2]{0}) constant(( { {1}, {2} }, {2, 42} ))
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

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={maximal device=1}
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={maximal device=1}
  %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={maximal device=0}, control-predecessors={%recv}
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}
}

)"
},
{
"SendRecvWithHostTransfer",
R"(HloModule HostTransferSendRecv_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, is_host_transfer=true
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, is_host_transfer=true
  %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, is_host_transfer=true
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, is_host_transfer=true
}

)"
},
// get-tuple-element
{
"GetTupleElement",
R"(HloModule GetTupleElement_module

ENTRY %GetTupleElement.v4 () -> s32[2,3] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 } })
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
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}
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
  %constant = f32[4,3,2,1]{0,1,2,3} constant({ { /*i0=0*/ { /*i1=0*/ {1}, {2} }, { /*i1=1*/ {3}, {4} }, { /*i1=2*/ {5}, {6} } }, { /*i0=1*/ { /*i1=0*/ {7}, {8} }, { /*i1=1*/ {9}, {10} }, { /*i1=2*/ {11}, {12} } }, { /*i0=2*/ { /*i1=0*/ {13}, {14} }, { /*i1=1*/ {15}, {16} }, { /*i1=2*/ {17}, {18} } }, { /*i0=3*/ { /*i1=0*/ {19}, {20} }, { /*i1=1*/ {21}, {22} }, { /*i1=2*/ {23}, {24} } } })
  ROOT %reverse = f32[4,3,2,1]{0,1,2,3} reverse(f32[4,3,2,1]{0,1,2,3} %constant), dimensions={0,1}
}

)"
},
// concat
{
"Concat",
R"(HloModule Concat2x3With2x5_module

ENTRY %Concat2x3With2x5.v3 () -> f32[2,8] {
  %constant = f32[2,3]{1,0} constant({ { 0, 1, 2 }, { 1000, 1001, 1002 } })
  %constant.1 = f32[2,5]{1,0} constant({ { 64, 65, 66, 67, 68 }, { 1064, 1065, 1066, 1067, 1068 } })
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
  %constant = f32[4,5,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {7} }, { /*i1=1*/ {2} }, { /*i1=2*/ {5} }, { /*i1=3*/ {3} }, { /*i1=4*/ {8} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {8} }, { /*i1=2*/ {9} }, { /*i1=3*/ {3} }, { /*i1=4*/ {4} } }, { /*i0=2*/ { /*i1=0*/ {1} }, { /*i1=1*/ {5} }, { /*i1=2*/ {7} }, { /*i1=3*/ {5} }, { /*i1=4*/ {6} } }, { /*i0=3*/ { /*i1=0*/ {0} }, { /*i1=1*/ {6} }, { /*i1=2*/ {2} }, { /*i1=3*/ {10} }, { /*i1=4*/ {2} } } })
  %constant.1 = f32[2,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {2} }, { /*i1=1*/ {6} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {1} } } })
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
  %constant = f32[3,3,3]{2,1,0} constant({ { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } }, { { 9, 10, 11 }, { 12, 13, 14 }, { 15, 16, 17 } }, { { 18, 19, 20 }, { 21, 22, 23 }, { 24, 25, 26 } } })
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
  %constant = s32[1,2,3]{2,1,0} constant({ { { 1, 2, 3 }, { 4, 5, 6 } } })
  ROOT %transpose = s32[1,2,3]{2,1,0} transpose(s32[1,2,3]{2,1,0} %constant), dimensions={0,1,2}
}

)"
},
{
"TransposeC128",
R"(HloModule TransposeC128_module

ENTRY %Transpose.v3 (input: c128[1,2,3]) -> c128[1,2,3] {
  %input = c128[1,2,3]{2,1,0} parameter(0)
  ROOT %transpose = c128[1,2,3]{2,1,0} transpose(c128[1,2,3]{2,1,0} %input), dimensions={0,1,2}
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
// Dynamic slice with scalar indices
{
"DynamicSliceScalarIndices",
R"(HloModule DynamicSlice_module

ENTRY %DynamicSlice.v5 (original_parameter: s32[2,2,258], start_index: s32[]) -> s32[2,2,258] {
  %original_parameter = s32[2,2,258]{2,1,0} parameter(0)
  %constant = s32[] constant(0)
  %start_index = s32[] parameter(1)
  ROOT %dynamic-slice = s32[2,2,258]{2,1,0} dynamic-slice(s32[2,2,258]{2,1,0} %original_parameter, s32[] %constant, s32[] %constant, s32[] %start_index), dynamic_slice_sizes={2,2,258}
}

)"
},
// Dynamic update slice
{
"DynamicUpdateSlice",
R"(HloModule DynamicSlice_module

ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_indices: s32[4]) -> s32[1,1,25,1] {
  %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
  %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
  %start_indices = s32[4]{0} parameter(2)
  ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[4]{0} %start_indices)
}

)"
},
// Dynamic update slice with scalar indices
{
"DynamicUpdateSliceScalarIndex",
R"(HloModule DynamicUpdateSlice_module

ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_index.0: s32[], start_index.1: s32[], start_index.2: s32[], start_index.3: s32[]) -> s32[1,1,25,1] {
  %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
  %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
  %start_index.0 = s32[] parameter(2)
  %start_index.1 = s32[] parameter(3)
  %start_index.2 = s32[] parameter(4)
  %start_index.3 = s32[] parameter(5)
  ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[] %start_index.0, s32[] %start_index.1, s32[] %start_index.2, s32[] %start_index.3)
}

)"
},
// batch norm training
{
"BatchNormTraining",
R"(HloModule BasicTraining_module

ENTRY %BasicTraining.v4 () -> (f32[2,2,1,2], f32[2], f32[2]) {
  %constant = f32[2,2,1,2]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ { 1, 2 } }, { /*i1=1*/ { 3, 4 } } }, { /*i0=1*/ { /*i1=0*/ { 5, 6 } }, { /*i1=1*/ { 7, 8 } } } })
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
  %constant = f32[3,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
  %constant.1 = f32[2]{0} constant({3.14, 4.25})
  ROOT %fusion = f32[3,2,1,1]{3,2,1,0} fusion(f32[3,2,1,1]{3,2,1,0} %constant, f32[2]{0} %constant.1), kind=kLoop, calls=%fused_computation
}

)"
},
{
"Sparse",
R"(HloModule sparse_f32

ENTRY %sparse () -> f32[2,3,4] {
  ROOT %foo = f32[2,3,4]sparse{10} constant({[0, 1, 2]: 1, [1, 2, 2]: 2, [1, 2, 3]: 3})
}

)"
},
{
"SparseC128",
R"(HloModule sparse_c128

ENTRY %sparse () -> c128[2,3,4] {
  ROOT %foo = c128[2,3,4]sparse{10} constant({[0, 1, 2]: (1, 0), [1, 2, 2]: (2, 5), [1, 2, 3]: (3, 10)})
}

)"
},
{
"SparseEmpty",
R"(HloModule sparse_f32_empty

ENTRY %sparse_f32_empty () -> f32[2,3,4] {
  ROOT %foo = f32[2,3,4]sparse{10} constant({})
}

)"
},
{
"SparseR1",
R"(HloModule sparse_f32_r1

ENTRY %sparse_f32_r1 () -> f32[9] {
  ROOT %foo = f32[9]sparse{10} constant({1: 2, 3: 4, 5: 6})
}

)"
},
{
"gather",
R"(HloModule StringifyGather

ENTRY %Gather (input_tensor: f32[50,49,48,47,46], start_indices: s64[10,9,8,7,5]) -> f32[10,9,8,7,30,29,28,27,26] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}
}

)"
},
{
"scatter",
R"(HloModule StringifyScatter

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates: f32[10,9,8,7,30,29,28,27,26]) -> f32[50,49,48,47,46] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, to_apply=%add_F32.v3
}

)"
},
{
  "ConstantUnsignedNoUnderflow",
  R"(HloModule ConstantUnsignedNoUnderflow_module

ENTRY %ConstantUnsignedNoUnderflow () -> u64[] {
  ROOT %constant = u64[] constant(1)
}

)"
},

{
  "ConstantUnsignedNoOverflow",
  R"(HloModule ConstantUnsignedNoOverflow_module

ENTRY %ConstantUnsignedNoOverflow () -> u64[] {
  ROOT %constant = u64[] constant(9223372036854775807)
}

)"
},
// CustomCallWithLayoutConstraints
{
"CustomCallWithLayoutConstraints",
R"(HloModule CustomCallWithLayoutConstraints

ENTRY %CustomCallWithLayoutConstraints (p0: f32[42,2,3], p1: f32[123,4]) -> f32[1,2,3] {
  %p0 = f32[42,2,3]{0,1,2} parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[42,2,3]{0,1,2} %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={f32[42,2,3]{0,1,2}, f32[123,4]{1,0}}
}

)"
},
// CustomCallWithLayoutConstraintsNoOperands
{
"CustomCallWithLayoutConstraintsNoOperands",
R"(HloModule CustomCallWithLayoutConstraintsNoOperands

ENTRY %CustomCallWithLayoutConstraints () -> f32[1,2,3] {
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(), custom_call_target="baz", operand_layout_constraints={}
}

)"
},
// CustomCallWithLayoutConstraintsTupleShapes
{
"CustomCallWithLayoutConstraintsTupleShapes",
R"(HloModule CustomCallWithLayoutConstraintsTupleShapes

ENTRY %CustomCallWithLayoutConstraints (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[1,2,3], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={(f32[2,2]{1,0}, f32[42,2,3]{2,0,1}), f32[123,4]{1,0}}
}

)"
},
// Parse c64 literal
{
"ParseC64Literal",
R"(HloModule ParseC64Literal

ENTRY %ParseC64Literal () -> c64[2] {
  ROOT %c = c64[2]{0} constant({(1, 2), (-inf, nan)})
}

)"
},
// Parse c128 literal
{
"ParseC128Literal",
R"(HloModule ParseC128Literal

ENTRY %ParseC128Literal () -> c128[2] {
  ROOT %c = c128[2]{0} constant({(1, 2), (-inf, nan)})
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
// tuple reduce
{
"TupleReduce",
R"(HloModule TupleReduce

max_argmax {
  value = f32[] parameter(2)
  prev_max = f32[] parameter(0)
  is_next_larger = pred[] greater-than-or-equal-to(value, prev_max)
  max = f32[] select(is_next_larger, value, prev_max)
  index = s32[] parameter(3)
  prev_argmax = s32[] parameter(1)
  argmax = s32[] select(is_next_larger, index, prev_argmax)
  ROOT pair = (f32[], s32[]) tuple(max, argmax)
}

ENTRY reduce_entry {
  values = f32[1024]{0} parameter(0)
  indices = f32[1024]{0} parameter(1)
  init_value = f32[] constant(-inf)
  init_index = s32[] constant(-1)
  ROOT result = (f32[], s32[]) reduce(values, indices, init_value, init_index), dimensions={0}, to_apply=max_argmax
}

)"
},
// infeed/outfeed
{
"InfeedOutfeed",
R"(HloModule outfeed_module

ENTRY InfeedToOutfeed {
  token0 = token[] after-all()
  infeed = ((u32[3]{0}, pred[]), token[]) infeed(token0)
  infeed.data = (u32[3]{0}, pred[]) get-tuple-element(infeed), index=0
  outfeed = token[] outfeed(infeed.data, token0)
  ROOT infeed.1 = ((u32[3]{0}, pred[]), token[]) infeed(token0)
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
// Sort (Key)
{
"SortKey",
R"(HloModule sort

ENTRY Sort {
  x = f32[1024]{0} parameter(0)
  ROOT sorted = f32[1024]{0} sort(x), dimensions={0}
}

)"
},
// Sort (Key, Value)
{
"SortKeyValue",
R"(HloModule sort

ENTRY Sort {
  keys = f32[1024]{0} parameter(0)
  values = s32[1024]{0} parameter(1)
  ROOT sorted = (f32[1024]{0}, s32[1024]{0}) sort(keys, values), dimensions={0}
}

)"
},
// R2 Sort (Key)
{
"SortKeyR2",
R"(HloModule sort

ENTRY Sort {
  x = f32[1024,16]{0,1} parameter(0)
  ROOT sorted = f32[1024,16]{0,1} sort(x), dimensions={0}
}

)"
},
// R2 Sort (Key, Value)
{
"SortKeyValueR2",
R"(HloModule sort

ENTRY Sort {
  keys = f32[1024,16]{0,1} parameter(0)
  values = s32[1024,16]{0,1} parameter(1)
  ROOT sorted = (f32[1024,16]{0,1}, s32[1024,16]{0,1}) sort(keys, values), dimensions={0}
}

)"
},
// Sort (Key, Value, Value, Value)
{
"SortManyValues",
R"(HloModule sort

ENTRY Sort {
  keys = f32[1024,16]{0,1} parameter(0)
  values.0 = s32[1024,16]{0,1} parameter(1)
  values.1 = u32[1024,16]{0,1} parameter(2)
  values.2 = f32[1024,16]{0,1} parameter(3)
  ROOT sorted = (f32[1024,16]{0,1}, s32[1024,16]{0,1}, u32[1024,16]{0,1}, f32[1024,16]{0,1}) sort(keys, values.0, values.1, values.2), dimensions={0}
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
// CustomCall with opaque value.
{
"CustomCallWithOpaque",
R"(HloModule custom_call

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar", opaque="this string is opaque"
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
  start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(input_tensor, start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}
}

)"
},
// all-reduce
{
"AllReduce",
R"(HloModule CRS

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT crs = f32[8]{0} all-reduce(input), replica_groups={}, to_apply=add
}

)"
},
// all-reduce with subgroups
{
"AllReduceWithSubgroups",
R"(HloModule CRS_Subgroups

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY AllReduceWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT all-reduce = f32[128,32]{0,1} all-reduce(input), replica_groups={{0,1},{2,3}}, barrier="abc", to_apply=add
}

)"
},
// all-reduce with all-reduce-id
{
"AllReduceAllReduce",
R"(HloModule CRS

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  crs.1 = f32[8]{0} all-reduce(input), replica_groups={{0}}, all_reduce_id=1, to_apply=add
  ROOT crs.0 = f32[8]{0} all-reduce(input), replica_groups={{0}}, all_reduce_id=1, to_apply=add
}

)"
},
// all-to-all
{
"AllToAll",
R"(HloModule AllToAll

ENTRY AllToAll {
  input = f32[128,32]{0,1} parameter(0)
  ROOT a2a = f32[128,32]{0,1} all-to-all(input), replica_groups={}
}

)"
},
// all-to-all with subgroups
{
"AllToAllWithSubgroups",
R"(HloModule AllToAllWithSubgroups

ENTRY AllToAllWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT a2a = f32[128,32]{0,1} all-to-all(input), replica_groups={{1,2},{3,0}}
}

)"
},
// collective-permute
{
"CollectivePermute",
R"(HloModule CollectivePermute

ENTRY CollectivePermute {
  input = f32[128,32]{0,1} parameter(0)
  ROOT root = f32[128,32]{0,1} collective-permute(input), source_target_pairs={{0,1},{1,2},{2,3}}
}

)"
},
// replica-id
{
"ReplicaId",
R"(HloModule replica-id

ENTRY Replica-id {
  ROOT replica-id = u32[] replica-id()
}

)"
},
// Iota
{
"Iota",
R"(HloModule iota

ENTRY Iota {
  ROOT iota = f32[100]{0} iota(), iota_dimension=0
}

)"
},
// custom-call with window, dim_labels and feature_group_count
{
"CustomCallWithWindowAndDimLabelsAndFeatureGroupCount",
R"(HloModule CustomCallWithWindowAndDimLabelsAndFeatureGroupCount

ENTRY Computation {
  ROOT r = f32[100]{0} custom-call(), window={size=2x2}, dim_labels=b01f_01io->b01f, feature_group_count=2, custom_call_target="target"
}

)"
    },
// is_scheduled=true attribute
{
"ScheduledModule",
R"(HloModule scheduled_module, is_scheduled=true

ENTRY Sort {
  keys = f32[1024]{0} parameter(0)
  values = s32[1024]{0} parameter(1)
  ROOT sorted = (f32[1024]{0}, s32[1024]{0}) sort(keys, values), dimensions={0}
}

)"
    },
// AfterAll with multiple operands
{
"AfterAllWithMultipleOperands",
R"(HloModule AfterAllWithMultipleOperands

ENTRY AfterAllWithMultipleOperands {
  p0 = f32[] parameter(0)
  token0 = token[] after-all()
  token1 = token[] after-all()
  ROOT after-all = token[] after-all(p0, token0, token1)
}

)"
},
// AddDependency
// A dependency chain is created from 'neg' to 'exp' using tokens.
{
"AddDependency",
R"(HloModule AddDependency

ENTRY AddDependency {
  p = f32[] parameter(0)
  neg = f32[] negate(p)
  token0 = token[] after-all(neg)
  p_after_token = f32[] add-dependency(p, token0)
  exp = f32[] exponential(p_after_token)
  ROOT sum = f32[] add(neg, exp)
}

)"
},

// A module containing constants equal to the min/max values of various data
// types.
{
"MinMaxValues",
R"(HloModule MinMaxValues

ENTRY MinMaxValues {
  x.s8 = s8[2]{0} constant({-128, 127})
  x.s16 = s16[2]{0} constant({-32768, 32767})
  x.s32 = s32[2]{0} constant({-2147483648, 2147483647})
  x.u8 = u8[2]{0} constant({0, 255})
  x.u16 = u16[2]{0} constant({0, 65535})
  x.u32 = u32[2]{0} constant({0, 4294967295})
  x.f16 = f16[2]{0} constant({-65504, 65504})
  x.bf16 = bf16[2]{0} constant({-3.38953e+38, 3.38953e+38})
  x.f32 = f32[2]{0} constant({-3.40282e+38, 3.40282e+38})
  x.f64 = f64[2]{0} constant({-1.79769e+308, 1.79769e+308})
  x.c64 = c64[2]{0} constant({(-3.40282e+38, 3.40282e+38), (3.40282e+38, -3.40282e+38)})
  ROOT c.c128 = c128[2]{0} constant({(-1.79769e+308, 1.79769e+308), (1.79769e+308, -1.79769e+308)})
}

)"
},
});
  // clang-format on
}

// The test class for those tests defined above which round-trip through the
// parser and ToString is templatized on two bool parameters:
//
//  short_form : used for the "short" test cases which use the ShortParsable
//    output form.
//  proto_round_trip : whether the module should also be round-tripped through
//    HloProto form. This provides much better coverage for the proto
//    serialization/deserialization.
//
// The proto_round_trip=true case also technically covers the Parser->ToString
// roundtrip as well, but separating out the Parser->ToString roundtrip as its
// own test provides better isolation and could conceivably catch weirdo bugs
// which are hidden by interaction between the textual and proto roundtripping.
template <bool short_form, bool proto_round_trip>
class HloParameterizedParserTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<TestData> {
 protected:
  // Expects "ToString(ParseHloString(string)) == string", that is, parses the
  // string, asserts that it succeeded, stringifies the parsed module, and
  // checks that the it equals the original string.
  void ExpectEqual() {
    const string& original = GetParam().module_string;
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseHloString(original));
    if (proto_round_trip) {
      TF_ASSERT_OK_AND_ASSIGN(module, HloModule::CreateFromProto(
                                          module->ToProto(), module->config()));
    }
    if (short_form) {
      EXPECT_EQ(original, module->ToString(HloPrintOptions::ShortParsable()));
    } else {
      EXPECT_EQ(
          original,
          module->ToString(HloPrintOptions().set_print_large_constants(true)));
    }
  }
};

// These using shenanigans are required because the TEST_P macro doesn't like
// template instantiations which contain commas.
using HloParserTestLong = HloParameterizedParserTest<false, false>;
using HloParserTestLongProto = HloParameterizedParserTest<false, true>;
using HloParserTestShort = HloParameterizedParserTest<true, false>;
using HloParserTestShortProto = HloParameterizedParserTest<true, true>;

TEST_P(HloParserTestLong, Run) { ExpectEqual(); }
TEST_P(HloParserTestLongProto, Run) { ExpectEqual(); }
TEST_P(HloParserTestShort, Run) { ExpectEqual(); }
TEST_P(HloParserTestShortProto, Run) { ExpectEqual(); }

INSTANTIATE_TEST_SUITE_P(HloParserTestSuccessInstantiation, HloParserTestLong,
                         ::testing::ValuesIn(CreateTestCases()),
                         TestDataToString);
INSTANTIATE_TEST_SUITE_P(HloParserTestSuccessInstantiation,
                         HloParserTestLongProto,
                         ::testing::ValuesIn(CreateTestCases()),
                         TestDataToString);
INSTANTIATE_TEST_SUITE_P(HloParserTestSuccessInstantiation, HloParserTestShort,
                         ::testing::ValuesIn(CreateShortTestCases()),
                         TestDataToString);
INSTANTIATE_TEST_SUITE_P(HloParserTestSuccessInstantiation,
                         HloParserTestShortProto,
                         ::testing::ValuesIn(CreateShortTestCases()),
                         TestDataToString);

class HloParserTest : public ::testing::Test {
 protected:
  static void ExpectHasSubstr(string_view s, string_view expected) {
    EXPECT_TRUE(absl::StrContains(s, expected))
        << "'" << s << "' does not contain '" << expected << "'";
  }
};

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
  %constant.1 = s32[] constant(-42), sharding={devices=[2,2]1,2,3,4}
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
  ROOT %constant = f32[2,3]{1,0} constant({1, 2, 3, 4, 5, 6})
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
  ROOT %constant = f32[2,3,2]{2,1,0} constant({{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}})
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

TEST_F(HloParserTest, ConstantBf16NoOverflow) {
  // 65505 is in range for bf16.
  const string original = R"(
  HloModule test_module
  ENTRY test {
    ROOT c = bf16[] constant(-65505)
  })";
  EXPECT_EQ(Status::OK(), ParseHloString(original).status());
}

TEST_F(HloParserTest, ConstantBf16Overflow) {
  // 1e100 is out of range for bf16.
  const string original = R"(
  HloModule test_module
  ENTRY test {
    ROOT c = bf16[] constant(1e100)
  })";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "out of range");
}

TEST_F(HloParserTest, ConstantF16OverflowInSparseArray) {
  const string original = R"(
    HloModule test_module
    ENTRY test {
      ROOT c = f16[5]sparse{10} constant({[0]: 0, [1]: -65505})
    })";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "is out of range for literal's primitive type F16");
}

TEST_F(HloParserTest, ConstantUnsignedUnderflow) {
  const string original = R"(
      HloModule ConstantUnsignedUnderflow_module
      ENTRY %ConstantUnsignedUnderflow () -> u64[] {
        ROOT %constant = u64[] constant(-1)
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "is out of range for literal's primitive type U64");
}

TEST_F(HloParserTest, ConstantUnsignedOverflow) {
  const string original = R"(
      HloModule ConstantUnsignedOverflow_module
      ENTRY %ConstantUnsignedOverflow () -> u32[] {
        ROOT %constant = u32[] constant(4294967296)
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "is out of range for literal's primitive type U32");
}

TEST_F(HloParserTest, ConstantUnsignedInt64Overflow) {
  const string original = R"(
      HloModule ConstantUnsignedOverflow_module
      ENTRY %ConstantUnsignedOverflow () -> u64[] {
        ROOT %constant = u64[] constant(9223372036854775808)
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, ConstantC64Overflow) {
  const string original = R"(
      HloModule test_module
      ENTRY test () -> c64[] {
        ROOT c = c64[] constant((1e100, 0))
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, ConstantC64Underflow) {
  const string original = R"(
      HloModule test_module
      ENTRY test () -> c64[] {
        ROOT c = c64[] constant((0, -1e100))
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, ConstantF64Overflow) {
  const string original = R"(
      HloModule test_module
      ENTRY test {
        ROOT c = f64[] constant(1.8e308)
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
}

TEST_F(HloParserTest, ConstantF64Underflow) {
  const string original = R"(
      HloModule test_module
      ENTRY test {
        ROOT c = f64[] constant(-1.8e308)
      })";
  auto result = ParseHloString(original);
  EXPECT_NE(Status::OK(), result.status());
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

TEST_F(HloParserTest, ShortConstant) {
  const string original = R"(HloModule ShortCOnstant_module

ENTRY %ShortConstant.v4 () -> f32[67,89] {
  ROOT %constant.1 = f32[67,89]{1,0} constant({...})
}

)";
  auto result = ParseHloString(original);
  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie()->ToString(HloPrintOptions()), original);
}

TEST_F(HloParserTest, AttibutesAnyOrder) {
  const string original = R"(HloModule any_order_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), feature_group_count=1, sharding={maximal device=1}, backend_config="foo", dim_labels=b0f_0io->b0f, window={pad=1_1 size=2}
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

  ExpectHasSubstr(
      ParseHloString(absl::StrCat(prefix, ",dim_labels=00_01_10", suffix))
          .status()
          .error_message(),
      "expects dim labels pattern");

  ExpectHasSubstr(
      ParseHloString(absl::StrCat(prefix, ",dim_labels=010_1100->010", suffix))
          .status()
          .error_message(),
      "must have the same rank");
}

TEST_F(HloParserTest, UnexpectedAttribute) {
  const string original = R"(HloModule unexpected_attr_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, calls=%recv
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "unexpected attribute \"calls\"");
}

TEST_F(HloParserTest, MissingAttribute) {
  const string original = R"(HloModule missing_attr_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(-2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0)
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "attribute channel_id is expected but not seen");
}

TEST_F(HloParserTest, PredecessorUndefined) {
  const string original = R"(HloModule pre_not_found_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, control-predecessors={%done}
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
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

TEST_F(HloParserTest, Comments) {
  const string original = R"(/* module description. */
HloModule comments:

ENTRY /*comment*/ c1 {
  /* blah */
  ROOT const1 = /*foo*/f32[1]{0} constant({12345 /*bar*/})
  /* comment */
}

/* something else */

)";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, MultilineComments) {
  const string original = R"(HloModule multiline_comment:
ENTRY c1 {
  /*
     ROOT foo = f32[1]{0} constant({12345})
  */
  ROOT const1 = f32[1]{0} constant({12345})
/*
a
b
c
d

*/
})";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, UnterminatedComment) {
  const string original = R"(HloModule unterminated_comment:
ENTRY c1 {
/* unterminated
  ROOT const1 = f32[1]{0} constant({12345})
})";
  // Verify that the error message points to the beginning of the unterminated
  // comment.
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "/* unterminated\n^");
}

TEST_F(HloParserTest, SlashSlashComments) {
  const string original = R"(HloModule slash_slash_comment:
// Garbage
ENTRY c1 {
  // Foo bar
  ROOT const1 = f32[1]{0} constant({12345}) // Something else
})";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, SlashSlashCommentMsDosEolFormat) {
  const string original =
      "HloModule slash_slash_comment:\r\n// Garbage\r\nENTRY c1 {\r\n// Foo "
      "bar\r\nROOT const1 = f32[1]{0} constant({12345}) // Something else\r\n}";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, SlashSlashCommentMacEolFormat) {
  const string original =
      "HloModule slash_slash_comment:\r// Garbage\rENTRY c1 {\r// Foo "
      "bar\rROOT const1 = f32[1]{0} constant({12345}) // Something else\r}";
  auto module = ParseHloString(original);
  TF_ASSERT_OK(module.status());
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

TEST_F(HloParserTest, SameNameDiffComputations) {
  const string original = R"(HloModule same_names:
add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT result = f32[] add(p0, p1)
}

ENTRY ReduceR3ToR2 {
  p0 = f32[8,16,256]{2,1,0} parameter(0)
  p1 = f32[] constant(0)
  ROOT result = f32[8,16]{1,0} reduce(p0, p1), dimensions={2}, to_apply=add
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(original));
  ASSERT_NE(module->entry_computation(), nullptr);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reduce()));
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

TEST_F(HloParserTest, ParsePaddingConfigNoInteriorPadding) {
  const string original = "0_1x2_3";
  TF_ASSERT_OK_AND_ASSIGN(PaddingConfig dnums, ParsePaddingConfig(original));
  EXPECT_EQ(original, PaddingConfigToString(dnums));
}

TEST_F(HloParserTest, ParsePaddingConfigInteriorPadding) {
  const string original = "0_1_0x2_3_4";
  TF_ASSERT_OK_AND_ASSIGN(PaddingConfig dnums, ParsePaddingConfig(original));
  EXPECT_EQ(original, PaddingConfigToString(dnums));
}

TEST_F(HloParserTest, ParsePaddingConfigInteriorPaddingImplicitZeroDim) {
  TF_ASSERT_OK_AND_ASSIGN(PaddingConfig dnums, ParsePaddingConfig("0_1x2_3_4"));
  // The extra "_0" gets added to the canonical string because the other dim has
  // interior padding.
  EXPECT_EQ("0_1_0x2_3_4", PaddingConfigToString(dnums));
}

TEST_F(HloParserTest, NontupleInfeed) {
  const string original = R"(HloModule nontuple_infeed:
ENTRY nontuple_infeed {
  token0 = token[] after-all()
  ROOT infeed = pred[] infeed(token0)
})";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "infeed must have a non-empty tuple shape");
}

TEST(HloParserSingleOpTest, SingleOp) {
  const string text =
      "%multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, "
      "f32[2,4]{1,0} %x)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
}

TEST(HloParserSingleOpTest, SingleOpNoShapeProducesError) {
  const string text = "multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)";
  StatusOr<std::unique_ptr<HloModule>> module = ParseHloString(text);
  ASSERT_TRUE(!module.status().ok());
  LOG(INFO) << "Status: " << module.status();
  EXPECT_THAT(module.status().ToString(),
              ::testing::HasSubstr("expects '=' in instruction"));
}

TEST(HloParserSingleOpTest, SingleOpNoOperandShapesProducesError) {
  const string text = "%multiply = f32[2,4]{1,0} multiply(%broadcast, %x)";
  StatusOr<std::unique_ptr<HloModule>> module = ParseHloString(text);
  ASSERT_TRUE(!module.status().ok());
  LOG(INFO) << "Status: " << module.status();
  EXPECT_THAT(module.status().ToString(),
              ::testing::HasSubstr("Operand had no shape in HLO text"));
}

TEST(HloParserSingleOpTest, SingleOpNoNames) {
  const string text =
      "%multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0}, f32[2,4]{1,0})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
}

TEST(HloParserSingleOpTest, CanonicalOp) {
  const string text = "f32[2,4]{1,0} multiply(f32[2,4]{1,0}, f32[2,4]{1,0})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(
      computation->root_instruction()->ToString(HloPrintOptions::Canonical()),
      text);
}

TEST(HloParserSingleOpTest, CanonicalOpWithNested) {
  const string text =
      R"(f32[5,20]{1,0} while(f32[5,10]{1,0}), condition=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  ROOT tmp_2 = f32[5,20]{1,0} fusion(f32[5,10]{1,0} tmp_0, f32[20,10]{1,0} tmp_1), kind=kLoop, calls=
  {
    tmp_0 = f32[5,10]{1,0} parameter(0)
    tmp_1 = f32[20,10]{1,0} parameter(1)
    tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
    ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
}, body=
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  tmp_1 = f32[20,10]{1,0} parameter(1)
  ROOT tmp_2 = f32[5,20]{1,0} fusion(f32[5,10]{1,0} tmp_0, f32[20,10]{1,0} tmp_1), kind=kLoop, calls=
  {
    tmp_0 = f32[5,10]{1,0} parameter(0)
    tmp_1 = f32[20,10]{1,0} parameter(1)
    tmp_2 = f32[10,20]{1,0} transpose(f32[20,10]{1,0} tmp_1), dimensions={1,0}
    ROOT tmp_3 = f32[5,20]{1,0} dot(f32[5,10]{1,0} tmp_0, f32[10,20]{1,0} tmp_2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_EQ(
      computation->root_instruction()->ToString(HloPrintOptions::Canonical()),
      text);
}

TEST(HloParserSingleOpTest, SingleOpWithNested) {
  const string text =
      R"(%fusion = f32[3,2,1,1]{3,2,1,0} fusion(f32[3,2,1,1]{3,2,1,0} %p0, f32[2]{0} %p1), kind=kLoop, calls=
{
  %param_0 = f32[3,2,1,1]{3,2,1,0} parameter(0)
  %param_1 = f32[2]{0} parameter(1)
  %broadcast = f32[3,2,1,1]{3,2,1,0} broadcast(f32[2]{0} %param_1), dimensions={1}
  ROOT %subtract = f32[3,2,1,1]{3,2,1,0} subtract(f32[3,2,1,1]{3,2,1,0} %param_0, f32[3,2,1,1]{3,2,1,0} %broadcast)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Op()
                             .WithOpcode(HloOpcode::kFusion)
                             .WithNumOperands(2)
                             .WithOperand(0, m::Parameter(0))
                             .WithOperand(1, m::Parameter(1))));
}

TEST(HloParserSingleOpTest, SingleOpWithNested_DoesNotExist) {
  const string text =
      R"(reduce = f32[] reduce(f32[10], f32[]), dimensions={1}, to_apply=
{
  result = f32[] add(f32[] x, f32[] y)
})";
  auto status = ParseHloString(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("does not exist: x"));
}

TEST(HloParserSingleOpTest, SingleOpWithNested_NoLhs) {
  const string text =
      R"(reduce = f32[] reduce(f32[10], f32[]), dimensions={1}, to_apply=
{
  f32[] add(f32[] x, f32[] y)
})";
  auto status = ParseHloString(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("expects name"));
}

TEST(HloParserSingleOpTest, SingleOpWithNested_NoOperandName) {
  const string text =
      R"(reduce = f32[] reduce(f32[10], f32[]), dimensions={1}, to_apply=
{
  result = f32[] add(f32[], f32[])
})";
  auto status = ParseHloString(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("expects name"));
}

TEST(HloParserSingleOpTest, ConvolutionTrivialFeatureGroupCount) {
  const string text =
      R"(%convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convolution(m::Parameter(0), m::Parameter(1))));
  auto* convolution =
      Cast<HloConvolutionInstruction>(computation->root_instruction());
  EXPECT_EQ(convolution->feature_group_count(), 1);
}

TEST_F(HloParserTest, IsScheduledIsFalse) {
  const string text = R"(
HloModule axpy_module, is_scheduled=false

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
  ASSERT_FALSE(module->has_schedule());
}

TEST_F(HloParserTest, IsScheduledNotPresent) {
  const string text = R"(
HloModule axpy_module

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
  ASSERT_FALSE(module->has_schedule());
}

TEST_F(HloParserTest, IsScheduledIsTrue) {
  const string text = R"(
HloModule axpy_module, is_scheduled=true

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());
  EXPECT_EQ(module->schedule().sequences().size(), 1);
  ASSERT_TRUE(
      module->schedule().is_computation_scheduled(module->entry_computation()));
  EXPECT_THAT(
      module->schedule().sequence(module->entry_computation()).instructions(),
      ::testing::ElementsAre(
          GmockMatch(m::Parameter()), GmockMatch(m::Broadcast()),
          GmockMatch(m::Parameter()), GmockMatch(m::Multiply()),
          GmockMatch(m::Parameter()), GmockMatch(m::Add())));
}

TEST_F(HloParserTest, IsScheduledIsTrueDifferentOrder) {
  // As above but in with a different schedule order.
  const string text = R"(
HloModule axpy_module, is_scheduled=true

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %x = f32[2,4]{1,0} parameter(1)
  %y = f32[2,4]{1,0} parameter(2)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());
  EXPECT_EQ(module->schedule().sequences().size(), 1);
  ASSERT_TRUE(
      module->schedule().is_computation_scheduled(module->entry_computation()));
  EXPECT_THAT(
      module->schedule().sequence(module->entry_computation()).instructions(),
      ::testing::ElementsAre(
          GmockMatch(m::Parameter()), GmockMatch(m::Parameter()),
          GmockMatch(m::Parameter()), GmockMatch(m::Broadcast()),
          GmockMatch(m::Multiply()), GmockMatch(m::Add())));
}

TEST_F(HloParserTest, CustomCallWrongNumberofOperandConstraints) {
  const string original = R"(HloModule CustomCallWrongNumberofOperandConstraints

ENTRY %CustomCallWrongNumberofOperandConstraints (p0: f32[42,2,3], p1: f32[123,4]) -> f32[1,2,3] {
  %p0 = f32[42,2,3]{0,1,2} parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = f32[1,2,3]{0,1,2} custom-call(f32[42,2,3]{0,1,2} %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={f32[42,2,3]{0,1,2}}
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "Expected 2 operand layout constraints, 1 given");
}

TEST_F(HloParserTest, CustomCallIncompatibleOperandConstraints) {
  const string original = R"(HloModule CustomCallIncompatibleOperandConstraints

ENTRY %CustomCallIncompatibleOperandConstraints (p0: f32[42,2,3], p1: f32[123,4]) -> f32[1,2,3] {
  %p0 = f32[42,2,3]{0,1,2} parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = f32[1,2,3]{0,1,2} custom-call(f32[42,2,3]{0,1,2} %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={f32[42,2,3]{0,1,2}, f32[555,5]{1,0}}
}

)";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "operand 1 is not compatible with operand shape");
}

TEST_F(HloParserTest, AllowShapeWhitespace) {
  const string text = R"(
HloModule module

ENTRY entry {
  ROOT root = f32[ 1, 2,3, 4, 5]{0, 1, 2,3, 4 } parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseHloString(text));
}

TEST_F(HloParserTest, ShapeMismatchInOperand) {
  const string text = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2] parameter(0)
  %constant.1 = f32[2,2] constant({{1, 2}, {3, 4}})
  ROOT %add.1 = f32[2,2] add(f32[2,2] %p, f32[2,5] %constant.1)
}
)";

  ExpectHasSubstr(ParseHloString(text).status().error_message(),
                  "The declared operand shape f32[2,5]{1,0} is not compatible"
                  " with the shape of the operand instruction f32[2,2]{1,0}.");
}

TEST_F(HloParserTest, OutOfRangeSparseIndex) {
  const string original = R"(
    HloModule test_module
    ENTRY test {
      ROOT c = f16[5]sparse{10} constant({[100]: 0})
    })";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "Invalid sparse index");
}

TEST_F(HloParserTest, NegativeSparseIndex) {
  const string original = R"(
    HloModule test_module
    ENTRY test {
      ROOT c = f16[5]sparse{10} constant({-1: 0})
    })";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "Invalid sparse index");
}

TEST_F(HloParserTest, SparseIndexWithRankTooLarge) {
  const string original = R"(
    HloModule test_module
    ENTRY test {
      ROOT c = f16[5]sparse{10} constant({[0, 0]: 0})
    })";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "Invalid sparse index");
}

TEST_F(HloParserTest, SparseIndexWithRankTooSmall) {
  const string original = R"(
    HloModule test_module
    ENTRY test {
      ROOT c = f16[5, 5]sparse{10} constant({[0]: 0})
    })";
  ExpectHasSubstr(ParseHloString(original).status().error_message(),
                  "Invalid sparse index");
}

TEST_F(HloParserTest, ParseShapeStringR2F32) {
  string shape_string = "f32[123,456]";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShape(F32, {123, 456});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringTupleOfArrays) {
  string shape_string = "(f32[1572864],s8[5120,1024])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1572864}),
                                 ShapeUtil::MakeShape(S8, {5120, 1024})});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringNestedTuple) {
  string shape_string = "(f32[1],(f32[2], token[]), opaque[], f32[3])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(F32, {1}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeTokenShape()}),
      ShapeUtil::MakeOpaqueShape(),
      ShapeUtil::MakeShape(F32, {3}),
  });
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringWithLayout) {
  string shape_string = "f32[123,456]{0,1}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithLayout(F32, {123, 456}, {0, 1});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringWithSparseLayout) {
  string shape_string = "f32[123,456]sparse{10}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithSparseLayout(F32, {123, 456}, 10);
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual: " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseOpaqueType) {
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape("opaque[]"));
  Shape expected = ShapeUtil::MakeOpaqueShape();
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseTokenType) {
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape("token[]"));
  Shape expected = ShapeUtil::MakeTokenShape();
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseInvalidShapeString) {
  string shape_strings[] = {
      "f32[123,456]foobar{0,1}", "f32[123,456]sparse{0,1}", "f32[123,456]{foo}",
      "f32[123,456]dense{foo}",  "f32[123,456]sparse{foo}",
  };
  for (const string& shape_string : shape_strings) {
    StatusOr<Shape> result = ParseShape(shape_string);
    ASSERT_FALSE(result.ok()) << "shape: " << shape_string;
  }
}

TEST_F(HloParserTest, ParseDynamicArray) {
  string shape_string = "f32[123,<=456]";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShape(F32, {123, 456}, {false, true});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseDynamicTuple) {
  string shape_string = "(f32[42], u32[<=123,<=456])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeShape(U32, {123, 456}, {true, true})});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, NegativeParameterNumber) {
  const string hlo_string = "par0 = f32[3,5] parameter(-1)";
  auto result = ParseHloString(hlo_string);
  ASSERT_FALSE(result.status().ok());
  EXPECT_THAT(result.status().error_message(),
              ::testing::HasSubstr("parameter number must be >= 0"));
}

}  // namespace
}  // namespace xla
