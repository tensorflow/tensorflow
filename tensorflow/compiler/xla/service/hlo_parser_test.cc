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

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

namespace m = ::xla::match;

using ::absl::string_view;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct TestData {
  std::string test_name;
  std::string module_string;
  int64_t replica_count = 1;
  bool enable_verification = true;
};

std::string TestDataToString(const ::testing::TestParamInfo<TestData>& data) {
  return data.param.test_name;
}

// Tests where the input module string doesn't match the output.
//
// In general we want to avoid these because we want HLO text to be
// round-trippable!  But nested instructions, e.g. add(sqrt(x), y), cannot be
// round-triped without modification.
struct NonRoundtripTestData {
  std::string test_name;
  std::string input_module_string;
  std::string output_module_string;
};

std::string NonRoundtripTestDataToString(
    const ::testing::TestParamInfo<NonRoundtripTestData>& data) {
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
R"(HloModule axpy_module, entry_computation_layout={(f32[],f32[2,4]{1,0},f32[2,4]{1,0})->f32[2,4]{1,0}}

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
// parameter replication
{
"ParamReplication",
R"(HloModule param_replication_module, entry_computation_layout={(f32[],(f32[2,4]{1,0}, (f32[2,4]{1,0})))->(f32[], (f32[2,4]{1,0}, (f32[2,4]{1,0})))}

ENTRY %param_replication (a: f32[], b: (f32[2,4], (f32[2,4]))) -> (f32[], (f32[2,4], (f32[2,4]))) {
  %a = f32[] parameter(0), parameter_replication={true}
  %b = (f32[2,4]{1,0}, (f32[2,4]{1,0})) parameter(1), parameter_replication={false,true}
  ROOT %tuple = (f32[], (f32[2,4]{1,0}, (f32[2,4]{1,0}))) tuple(f32[] %a, (f32[2,4]{1,0}, (f32[2,4]{1,0})) %b)
}

)"
},
// pred constant
{
"ConstantPred",
R"(HloModule constant_pred_module, entry_computation_layout={()->pred[]}

ENTRY %constant_pred () -> pred[] {
  ROOT %constant = pred[] constant(true), metadata={op_type="const" op_name="\"it\'s not a problem\n" source_file="path/to/test.cc" source_line=68}, backend_config="foo\" bar"
}

)"
},
// pred array constant
{
"ConstantPredArray",
R"(HloModule module, entry_computation_layout={()->pred[2,3]{1,0}}

ENTRY %constant_pred_array () -> pred[2,3] {
  ROOT %constant = pred[2,3]{1,0} constant({ { 0, 1, 0 }, { 1, 0, 1 } })
}

)"
},

// s32 constant
{
"ConstantS32",
R"(HloModule constant_s32_module, entry_computation_layout={()->s32[]}

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42)
}

)"
},
// f32 constant, but the value is not a decimal and there is a backend
// configuration
{
"ConstantF32",
R"(HloModule ConstantF32_module, entry_computation_layout={()->f32[]}

ENTRY %ConstantF32.v4 () -> f32[] {
  ROOT %constant = f32[] constant(42), backend_config="this is a configuration"
}

)"
},
// f32 constant, rank 1 empty array.
{
"ConstantF32R1Empty",
R"(HloModule ConstantF32Empty_module, entry_computation_layout={()->f32[0]{0}}

ENTRY %ConstantF32Empty.v4 () -> f32[0] {
  ROOT %constant = f32[0]{0} constant({})
}

)"
},
// f32 constant, rank 4 empty array.
{
"ConstantF32R4Empty",
R"(HloModule ConstantF32R4Empty_module, entry_computation_layout={()->f32[2,0,4,3]{3,2,1,0}}

ENTRY %ConstantF32R4Empty.v4 () -> f32[2,0,4,3] {
  ROOT %constant = f32[2,0,4,3]{3,2,1,0} constant({ { /*i0=0*/ }, { /*i0=1*/ } })
}

)"
},
// constant 4D
{
"Constant4D",
R"(HloModule Small_3x2x1x1_module, entry_computation_layout={()->f32[3,2,1,1]{3,2,1,0}}

ENTRY %Small_3x2x1x1.v1 () -> f32[3,2,1,1] {
  ROOT %constant = f32[3,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
}

)"
},
// non-finite constants: nan, inf, -inf
{
"ConstantNonFinite",
R"(HloModule IsFiniteR1F32s_module, entry_computation_layout={()->pred[6]{0}}

ENTRY %IsFiniteR1F32s.v2 () -> pred[6] {
  %constant = f32[6]{0} constant({nan, 7, nan, -1, inf, -inf})
  ROOT %is-finite = pred[6]{0} is-finite(f32[6]{0} %constant)
}

)"
},
// constant f16
{
"ConstantF16",
R"(HloModule ConstantF16_module, entry_computation_layout={()->f16[]}

ENTRY %ConstantF16.v4 () -> f16[] {
  ROOT %constant = f16[] constant(500)
}

)"
},
// bf16
{
"BF16",
R"(HloModule BF16, entry_computation_layout={()->bf16[]}

ENTRY %BF16.v4 () -> bf16[] {
  ROOT %constant = bf16[] constant(500)
}

)"
},
// constant + constant
{
"AddConstants",
R"(HloModule add_constants_module, entry_computation_layout={()->f32[]}

ENTRY %add_constants () -> f32[] {
  %constant = f32[] constant(3.14)
  ROOT %add = f32[] add(f32[] %constant, f32[] %constant)
}

)"
},
// tuple constant
{
"TupleConstant",
R"(HloModule TupleConstant_module, entry_computation_layout={()->(f32[2,1]{1,0}, f32[2]{0})}

ENTRY %TupleConstant.v1 () -> (f32[2,1], f32[2]) {
  ROOT %constant = (f32[2,1]{1,0}, f32[2]{0}) constant(( { {1}, {2} }, {2, 42} ))
}

)"
},
// v1 > v2 ? v1 : v2
{
"SelectR1F32",
R"(HloModule SelectR1F32WithCmpR1F32sFromParamsSmall_module, entry_computation_layout={(f32[4]{0},f32[4]{0})->f32[4]{0}}

ENTRY %SelectR1F32WithCmpR1F32sFromParamsSmall.v4 (v1: f32[4], v2: f32[4]) -> f32[4] {
  %v1 = f32[4]{0} parameter(0), sharding={maximal device=1}
  %v2 = f32[4]{0} parameter(1), sharding={maximal device=1}
  %greater-than = pred[4]{0} compare(f32[4]{0} %v1, f32[4]{0} %v2), direction=GT, type=TOTALORDER, sharding={replicated}
  ROOT %select = f32[4]{0} select(pred[4]{0} %greater-than, f32[4]{0} %v1, f32[4]{0} %v2), sharding={replicated}
}

)"
},
// empty tuple
{
"EmptyTupleCreate",
R"(HloModule EmptyTupleCreate_module, entry_computation_layout={()->()}

ENTRY %EmptyTupleCreate.v1 () -> () {
  ROOT %tuple = () tuple()
}

)"
},
// tuple
{
"TupleCreate",
R"(HloModule TupleCreate_module, entry_computation_layout={(f32[],f32[3]{0},f32[2,3]{1,0})->(f32[], f32[3]{0}, f32[2,3]{1,0})}

ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}

)"
},
// tuple
{
"LargeTupleRoundTrip",
R"(HloModule LargeTupleRoundTrip_module, entry_computation_layout={(f32[])->(f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[])}

ENTRY %TupleCreate.v4 (v: f32[]) -> (f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[]) {
  %v = f32[] parameter(0)
  ROOT %tuple = (f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[]) tuple(f32[] %v, f32[] %v, f32[] %v, f32[] %v, f32[] %v, /*index=5*/f32[] %v)
}

)"
},
{
"ShardedTupleCreate",
R"(HloModule ShardedTupleCreate_module, entry_computation_layout={(f32[],f32[3]{0},f32[2,3]{1,0})->(f32[], f32[3]{0}, f32[2,3]{1,0})}

ENTRY %ShardedTupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0), sharding={manual}
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3), sharding={{manual}, {maximal device=0}, {replicated}}
}

)"
},
{
"DomainParsing",
R"(HloModule DomainParsing_module, entry_computation_layout={(f32[])->f32[]}

ENTRY %DomainParsing (v1: f32[]) -> f32[] {
  %v1 = f32[] parameter(0)
  ROOT %dom = f32[] domain(f32[] %v1), domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
}

)"
},
// int32_t result = 0;
// while (result < 5) { result = result + 1; }
{
"WhileWithScalarS32Result",
R"(HloModule WhileWithScalarS32Result_module, entry_computation_layout={()->s32[]}

%body.v3 (prev.1: s32[]) -> s32[] {
  %constant = s32[] constant(1)
  %prev.1 = s32[] parameter(0)
  ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
}

%condition.v3 (prev.2: s32[]) -> pred[] {
  %constant.1 = s32[] constant(5)
  %prev.2 = s32[] parameter(0)
  ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %prev.2), direction=GT
}

ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
  %constant.2 = s32[] constant(0)
  ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
}

)"
},
// copy-start and copy-done
{
"CopyStartAndCopyDone",

R"(HloModule CopyStartAndCopyDone_module, entry_computation_layout={(f32[],f32[2,3]{1,0:S(1)})->(f32[], f32[2,3]{1,0:S(2)})}

ENTRY %CopyStartAndCopyDone (v1: f32[], v2: f32[2,3]) -> (f32[], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %copy-start.1 = (f32[], f32[], u32[]) copy-start(f32[] %v1), is_cross_program_prefetch=true
  %copy-done.1 = f32[] copy-done((f32[], f32[], u32[]) %copy-start.1)
  %v2 = f32[2,3]{1,0:S(1)} parameter(1)
  %copy-start.2 = (f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) copy-start(f32[2,3]{1,0:S(1)} %v2)
  %copy-done.2 = f32[2,3]{1,0:S(2)} copy-done((f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) %copy-start.2)
  ROOT %tuple = (f32[], f32[2,3]{1,0:S(2)}) tuple(f32[] %copy-done.1, f32[2,3]{1,0:S(2)} %copy-done.2)
}

)"
},
// send and recv
{
"SendRecv",
R"(HloModule TwoSendRecvBothWayRecvFist_module, entry_computation_layout={()->(f32[], token[])}

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={{maximal device=1}, {replicated}, {replicated}}
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={{maximal device=1}, {replicated}}
  %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={{maximal device=1}, {replicated}, {replicated}}, control-predecessors={%recv}
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}
}

)"
},
{
"SendRecvWithHostTransfer",
R"(HloModule HostTransferSendRecv_module, entry_computation_layout={()->(f32[], token[])}

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
R"(HloModule GetTupleElement_module, entry_computation_layout={()->s32[2,3]{1,0}}

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
R"(HloModule CallR0F32IdentityScalar_module, entry_computation_layout={()->f32[]}

%Identity.v1 (x: f32[]) -> f32[] {
  ROOT %x = f32[] parameter(0)
}

ENTRY %CallR0F32IdentityScalar.v2 () -> f32[] {
  %constant = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant), to_apply=%Identity.v1
}

)"
},
// CustomCall with backend_config.
{
"CustomCallWithOpaque",
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", backend_config="this string is opaque"
}

)"
},

// CustomCall with literal.
{
"CustomCallWithLiteral",
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", literal=s32[2]{0} {1, 2}
}

)"
},

// CustomCall with literal tuple.
{
"CustomCallWithLiteralTuple",
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", literal=( s32[4]{0} {4, 128, 128, 3}, pred[4]{0} {1, 0, 0, 0} )
}

)"
},

// CustomCall with literal R0.
{
"CustomCallWithLiteralR0",
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", literal=f32[] 0.1
}

)"
},
// reduce window
{
"ReduceWindow",
R"(HloModule R4UnitWindow_module, entry_computation_layout={(f32[13,12,8,15]{0,3,2,1})->f32[13,3,8,15]{0,3,2,1}}

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
R"(HloModule reduce_window_scalar, entry_computation_layout={()->f32[]}

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
// reduce window on scalar
{
"ReduceWindowVariadic",
R"(HloModule reduce_window_variadic, entry_computation_layout={()->(f32[], f32[])}

%add_F32.v3 (lhs1: f32[], lhs2: f32[], rhs1: f32[], rhs2: f32[]) -> (f32[], f32[]) {
  %lhs1 = f32[] parameter(0)
  %rhs1 = f32[] parameter(2)
  %add1 = f32[] add(f32[] %lhs1, f32[] %rhs1)
  %lhs2 = f32[] parameter(1)
  %rhs2 = f32[] parameter(3)
  %add2 = f32[] add(f32[] %lhs2, f32[] %rhs2)
  ROOT %tuple1 = (f32[], f32[]) tuple(f32[] %add1, f32[] %add2)
}

ENTRY %R4UnitWindowScalar () -> (f32[], f32[]) {
  %constant = f32[] constant(42)
  %constant.1 = f32[] constant(1)
  ROOT %reduce-window = (f32[], f32[]) reduce-window(f32[] %constant, f32[] %constant, f32[] %constant.1, f32[] %constant.1), to_apply=%add_F32.v3
}

)"
},
// convolution
{
"Convolution",
R"(HloModule Convolve1D1Window_0_module, entry_computation_layout={(f32[1,2,1]{2,1,0},f32[1,1,1]{2,1,0})->f32[1,2,1]{2,0,1}}

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}
}

)"
},
// convolution dynamic
{
"ConvolutionDynamic",
R"(HloModule Convolve1D1Window_0_module, entry_computation_layout={(f32[1,2,1]{2,1,0},f32[1,1,1]{2,1,0})->f32[1,2,1]{2,0,1}}

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %custom-call.52 = f32[1,2,1]{2,0,1} custom-call(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}, custom_call_target="DynamicConvolutionForward", metadata={op_type="Conv2D" op_name="conv1d"}
}

)"
},
// convolution rank 2
{
"ConvolutionR2",
R"(HloModule ConvolveR2_module, entry_computation_layout={(f32[1,2]{1,0},f32[2,2]{1,0})->f32[1,2]{0,1}}

ENTRY %ConvolveR2.v3 (input: f32[1,2], filter: f32[2,2]) -> f32[1,2] {
  %input = f32[1,2]{1,0} parameter(0)
  %filter = f32[2,2]{1,0} parameter(1)
  ROOT %convolution = f32[1,2]{0,1} convolution(f32[1,2]{1,0} %input, f32[2,2]{1,0} %filter), dim_labels=bf_io->bf
}

)"
},
// convolution backward
{
"ConvolutionBackward",
R"(HloModule ConvolveBackward_module, entry_computation_layout={(f32[128,7,7,512]{0,3,2,1},f32[3,3,512,512]{3,2,1,0})->f32[128,14,14,512]{0,3,2,1}}

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
R"(HloModule Reverse4DFloatArrayOnDim01_module, entry_computation_layout={()->f32[4,3,2,1]{0,1,2,3}}

ENTRY %Reverse4DFloatArrayOnDim01.v2 () -> f32[4,3,2,1] {
  %constant = f32[4,3,2,1]{0,1,2,3} constant({ { /*i0=0*/ { /*i1=0*/ {1}, {2} }, { /*i1=1*/ {3}, {4} }, { /*i1=2*/ {5}, {6} } }, { /*i0=1*/ { /*i1=0*/ {7}, {8} }, { /*i1=1*/ {9}, {10} }, { /*i1=2*/ {11}, {12} } }, { /*i0=2*/ { /*i1=0*/ {13}, {14} }, { /*i1=1*/ {15}, {16} }, { /*i1=2*/ {17}, {18} } }, { /*i0=3*/ { /*i1=0*/ {19}, {20} }, { /*i1=1*/ {21}, {22} }, { /*i1=2*/ {23}, {24} } } })
  ROOT %reverse = f32[4,3,2,1]{0,1,2,3} reverse(f32[4,3,2,1]{0,1,2,3} %constant), dimensions={0,1}
}

)"
},
// concat
{
"Concat",
R"(HloModule Concat2x3With2x5_module, entry_computation_layout={()->f32[2,8]{1,0}}

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
R"(HloModule R4F32OverlapSmall_module, entry_computation_layout={()->f32[4,5,1,1]{3,2,1,0}}

%ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %greater-than-or-equal-to = pred[] compare(f32[] %lhs, f32[] %rhs), direction=GE, type=TOTALORDER
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
R"(HloModule select_and_scatter_scalar, entry_computation_layout={()->f32[]}

%ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %greater-than-or-equal-to = pred[] compare(f32[] %lhs, f32[] %rhs), direction=GE
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
R"(HloModule slice_module, entry_computation_layout={(f32[3,3,4,4]{3,2,1,0})->f32[3,3,2,4]{3,2,1,0}}

ENTRY %slice.v2 (p0: f32[3,3,4,4]) -> f32[3,3,2,4] {
  %p0 = f32[3,3,4,4]{3,2,1,0} parameter(0)
  ROOT %slice = f32[3,3,2,4]{3,2,1,0} slice(f32[3,3,4,4]{3,2,1,0} %p0), slice={[0:3:1], [0:3:1], [0:4:2], [0:4:1]}
}

)"
},
// slice, no stride
{
"SliceNoStride",
R"(HloModule Slice3x3x3_To_1x3x3_F32_module, entry_computation_layout={()->f32[1,3,3]{2,1,0}}

ENTRY %Slice3x3x3_To_1x3x3_F32.v2 () -> f32[1,3,3] {
  %constant = f32[3,3,3]{2,1,0} constant({ { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } }, { { 9, 10, 11 }, { 12, 13, 14 }, { 15, 16, 17 } }, { { 18, 19, 20 }, { 21, 22, 23 }, { 24, 25, 26 } } })
  ROOT %slice = f32[1,3,3]{2,1,0} slice(f32[3,3,3]{2,1,0} %constant), slice={[0:1], [0:3], [0:3]}
}

)"
},
// slice R0
{
"SliceR0",
R"(HloModule SliceR0_module, entry_computation_layout={()->s32[]}

ENTRY %SliceR0.v2 () -> s32[] {
  %constant = s32[] constant(1)
  ROOT %slice = s32[] slice(s32[] %constant), slice={}
}

)"
},
// transpose
{
"Transpose",
R"(HloModule Transpose_module, entry_computation_layout={()->s32[1,2,3]{2,1,0}}

ENTRY %Transpose.v2 () -> s32[1,2,3] {
  %constant = s32[1,2,3]{2,1,0} constant({ { { 1, 2, 3 }, { 4, 5, 6 } } })
  ROOT %transpose = s32[1,2,3]{2,1,0} transpose(s32[1,2,3]{2,1,0} %constant), dimensions={0,1,2}
}

)"
},
{
"TransposeC128",
R"(HloModule TransposeC128_module, entry_computation_layout={(c128[1,2,3]{2,1,0})->c128[1,2,3]{2,1,0}}

ENTRY %Transpose.v3 (input: c128[1,2,3]) -> c128[1,2,3] {
  %input = c128[1,2,3]{2,1,0} parameter(0)
  ROOT %transpose = c128[1,2,3]{2,1,0} transpose(c128[1,2,3]{2,1,0} %input), dimensions={0,1,2}
}

)"
},
// Triangular solve
{
"TriangularSolve",
R"(HloModule TriangularSolve_module, entry_computation_layout={(f32[4,4]{1,0},f32[3,4]{1,0})->f32[3,4]{1,0}}

ENTRY %SimpleRightLowerNotranspose.4 (a.1: f32[4,4], b.2: f32[3,4]) -> f32[3,4] {
  %a.1 = f32[4,4]{1,0} parameter(0)
  %b.2 = f32[3,4]{1,0} parameter(1)
  ROOT %triangular-solve.3 = f32[3,4]{1,0} triangular-solve(f32[4,4]{1,0} %a.1, f32[3,4]{1,0} %b.2), lower=true, transpose_a=NO_TRANSPOSE
}

)"
},
// Dynamic slice
{
"DynamicSlice",
R"(HloModule DynamicSlice_module, entry_computation_layout={(s32[2,2,258]{2,1,0},s32[1]{0})->s32[2,2,258]{2,1,0}}

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
R"(HloModule DynamicSlice_module, entry_computation_layout={(s32[2,2,258]{2,1,0},s32[])->s32[2,2,258]{2,1,0}}

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
R"(HloModule DynamicSlice_module, entry_computation_layout={(s32[1,1,25,1]{3,2,1,0},s32[1,1,2,1]{3,2,1,0},s32[4]{0})->s32[1,1,25,1]{3,2,1,0}}

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
R"(HloModule DynamicUpdateSlice_module, entry_computation_layout={(s32[1,1,25,1]{3,2,1,0},s32[1,1,2,1]{3,2,1,0},s32[],s32[],s32[],s32[])->s32[1,1,25,1]{3,2,1,0}}

ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_index.0: s32[], start_index.1: s32[], start_index.2: s32[], start_index.3: s32[]) -> s32[1,1,25,1] {
  %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
  %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
  %start_index.0 = s32[] parameter(2)
  %start_index.1 = s32[] parameter(3)
  %start_index.2 = s32[] parameter(4)
  %start_index.3 = s32[] parameter(5)
  ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[] %start_index.0, s32[] %start_index.1, s32[] %start_index.2, /*index=5*/s32[] %start_index.3)
}

)"
},
// batch norm training
{
"BatchNormTraining",
R"(HloModule BasicTraining_module, entry_computation_layout={()->(f32[2,2,1,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})}

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
R"(HloModule BatchNormInference_module, entry_computation_layout={(f32[2,2,2,2]{3,2,1,0},f32[2]{0},f32[2]{0},f32[2]{0},f32[2]{0})->f32[2,2,2,2]{3,2,1,0}}

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
R"(HloModule BatchNormGrad_module, entry_computation_layout={(f32[2,2,2,2]{3,2,1,0},f32[2]{0},f32[2]{0},f32[2]{0},f32[2,2,2,2]{3,2,1,0})->(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})}

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
R"(HloModule Fft_module, entry_computation_layout={(c64[8,32]{1,0})->c64[8,32]{1,0}}

ENTRY %Fft (input: c64[8,32]) -> c64[8,32] {
  %input = c64[8,32]{1,0} parameter(0)
  ROOT %fft = c64[8,32]{1,0} fft(c64[8,32]{1,0} %input), fft_type=FFT, fft_length={32}
}

)"
},
// ifft
{
"Ifft2d",
R"(HloModule Ifft2d_module, entry_computation_layout={(c64[5,8,32]{2,1,0})->c64[5,8,32]{2,1,0}}

ENTRY %Ifft2d (input: c64[5,8,32]) -> c64[5,8,32] {
  %input = c64[5,8,32]{2,1,0} parameter(0)
  ROOT %fft = c64[5,8,32]{2,1,0} fft(c64[5,8,32]{2,1,0} %input), fft_type=IFFT, fft_length={8,32}
}

)"
},
// rfft2d
{
"Rfft2d",
R"(HloModule Rfft2d_module, entry_computation_layout={(f32[5,64,32]{2,1,0})->c64[5,64,17]{2,1,0}}

ENTRY %Rfft2d (input: f32[5,64,32]) -> c64[5,64,17] {
  %input = f32[5,64,32]{2,1,0} parameter(0)
  ROOT %fft = c64[5,64,17]{2,1,0} fft(f32[5,64,32]{2,1,0} %input), fft_type=RFFT, fft_length={64,32}
}

)"
},
// irfft3d
{
"Irfft3d",
R"(HloModule Irfft3d_module, entry_computation_layout={(c64[5,64,128,33]{3,2,1,0})->f32[5,64,128,64]{3,2,1,0}}

ENTRY %Irfft3d (input: c64[5,64,128,33]) -> f32[5,64,128,64] {
  %input = c64[5,64,128,33]{3,2,1,0} parameter(0)
  ROOT %fft = f32[5,64,128,64]{3,2,1,0} fft(c64[5,64,128,33]{3,2,1,0} %input), fft_type=IRFFT, fft_length={64,128,64}
}

)"
},
// pad
{
"Pad",
R"(HloModule Pad1DS3Array_module, entry_computation_layout={()->f32[7]{0}}

ENTRY %Pad1DS3Array.v3 () -> f32[7] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = f32[] constant(0.1)
  ROOT %pad = f32[7]{0} pad(f32[3]{0} %constant, f32[] %constant.1), padding=3_1
}

)"
},
// pad has interior
{
"PadHasInterior",
R"(HloModule PadHasInterior_module, entry_computation_layout={(f32[1,25,7,7]{3,2,1,0})->f32[1,25,17,11]{3,2,1,0}}

ENTRY %PadHasInterior.v3 (input: f32[1,25,7,7]) -> f32[1,25,17,11] {
  %input = f32[1,25,7,7]{3,2,1,0} parameter(0)
  %constant = f32[] constant(-5.123)
  ROOT %pad = f32[1,25,17,11]{3,2,1,0} pad(f32[1,25,7,7]{3,2,1,0} %input, f32[] %constant), padding=0_0_0x0_0_0x2_2_1x2_2_0
}

)"
},
// round to nearest even
{
"RoundNearestEven",
R"(HloModule RoundNearestEven_module, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

ENTRY %RoundNearestEven (input: f32[2,2]) -> f32[2,2] {
  %input = f32[2,2]{1,0} parameter(0)
  ROOT %round-nearest-even = f32[2,2]{1,0} round-nearest-even(f32[2,2]{1,0} %input)
}

)"
},
// Negative padding
{
"PadHasNegativePadding",
R"(HloModule PadHasNegativePadding_module, entry_computation_layout={(f32[1,25,7,7,10]{4,3,2,1,0})->f32[1,15,6,3,35]{4,3,2,1,0}}

ENTRY %PadHasNegativePadding (input: f32[1,25,7,7,10]) -> f32[1,15,6,3,35] {
  %input = f32[1,25,7,7,10]{4,3,2,1,0} parameter(0)
  %constant = f32[] constant(-5.123)
  ROOT %pad = f32[1,15,6,3,35]{4,3,2,1,0} pad(f32[1,25,7,7,10]{4,3,2,1,0} %input, f32[] %constant), padding=0_0_0x0_-10_0x0_-1_0x-2_-2_0x-1_-1_3
}

)"
},
// fusion
{
"Fusion",
R"(HloModule fusion_module, entry_computation_layout={()->f32[3,2,1,1]{3,2,1,0}}

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
// FusionWithAliasing
{
"FusionWithAliasing",
R"(HloModule FusionWithAliasing, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}),f32[123,4]{0,1})->(f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2})}

%FusedComp (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[123,4], f32[2,2], f32[1,2,3]) {
  %p1 = f32[123,4]{0,1} parameter(1)
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %elem1 = f32[2,2]{0,1} get-tuple-element((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0), index=0
  %constant0 = f32[] constant(1)
  %broadcast0 = f32[1,2,3]{0,1,2} broadcast(f32[] %constant0), dimensions={}
  ROOT %tuple = (f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2}) tuple(f32[123,4]{0,1} %p1, f32[2,2]{0,1} %elem1, f32[1,2,3]{0,1,2} %broadcast0)
}

ENTRY %FusionWithAliasing (p0.1: (f32[2,2], f32[42,2,3]), p1.1: f32[123,4]) -> (f32[123,4], f32[2,2], f32[1,2,3]) {
  %p0.1 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1.1 = f32[123,4]{0,1} parameter(1)
  ROOT %fusion = (f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2}) fusion((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0.1, f32[123,4]{0,1} %p1.1), kind=kLoop, output_to_operand_aliasing={{0}: (1, {}), {1}: (0, {0})}, calls=%FusedComp
}

)"
},
{
"Gather",
R"(HloModule StringifyGather, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}}

ENTRY %Gather (input_tensor: f32[50,49,48,47,46], start_indices: s64[10,9,8,7,5]) -> f32[10,9,8,7,30,29,28,27,26] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}
}

)"
},
{
"SortedGather",
R"(HloModule StringifyGather, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}}

ENTRY %Gather (input_tensor: f32[50,49,48,47,46], start_indices: s64[10,9,8,7,5]) -> f32[10,9,8,7,30,29,28,27,26] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}, indices_are_sorted=true
}

)"
},
{
"Scatter",
R"(HloModule StringifyScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0},f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46]{4,3,2,1,0}}

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
"TupleScatter",
R"(HloModule TupleScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},bf16[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0},f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0},bf16[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->(f32[50,49,48,47,46]{4,3,2,1,0}, bf16[50,49,48,47,46]{4,3,2,1,0})}

%add_F32_mul_BF16 (lhs_0: f32[], lhs_1: bf16[], rhs_0: f32[], rhs_1: bf16[]) -> (f32[], bf16[]) {
  %lhs_0 = f32[] parameter(0)
  %rhs_0 = f32[] parameter(2)
  %add = f32[] add(f32[] %lhs_0, f32[] %rhs_0)
  %lhs_1 = bf16[] parameter(1)
  %rhs_1 = bf16[] parameter(3)
  %mul = bf16[] multiply(bf16[] %lhs_1, bf16[] %rhs_1)
  ROOT %tuple = (f32[], bf16[]) tuple(f32[] %add, bf16[] %mul)
}

ENTRY %Scatter (input_0: f32[50,49,48,47,46], input_1: bf16[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates_0: f32[10,9,8,7,30,29,28,27,26], updates_1: bf16[10,9,8,7,30,29,28,27,26]) -> (f32[50,49,48,47,46], bf16[50,49,48,47,46]) {
  %input_0 = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %input_1 = bf16[50,49,48,47,46]{4,3,2,1,0} parameter(1)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(2)
  %updates_0 = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(3)
  %updates_1 = bf16[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(4)
  ROOT %scatter = (f32[50,49,48,47,46]{4,3,2,1,0}, bf16[50,49,48,47,46]{4,3,2,1,0}) scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_0, bf16[50,49,48,47,46]{4,3,2,1,0} %input_1, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates_0, bf16[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates_1), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, to_apply=%add_F32_mul_BF16
}

)"
},
{
"SortedScatter",
R"(HloModule StringifySortedScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0},f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46]{4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates: f32[10,9,8,7,30,29,28,27,26]) -> f32[50,49,48,47,46] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, indices_are_sorted=true, to_apply=%add_F32.v3
}

)"
},
{
"UniqueIndicesScatter",
R"(HloModule StringifyUniqueIndicesScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0},f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46]{4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates: f32[10,9,8,7,30,29,28,27,26]) -> f32[50,49,48,47,46] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, unique_indices=true, to_apply=%add_F32.v3
}

)"
},
{
  "ConstantUnsignedNoUnderflow",
  R"(HloModule ConstantUnsignedNoUnderflow_module, entry_computation_layout={()->u64[]}

ENTRY %ConstantUnsignedNoUnderflow () -> u64[] {
  ROOT %constant = u64[] constant(1)
}

)"
},

{
  "ConstantUnsignedNoOverflow",
  R"(HloModule ConstantUnsignedNoOverflow_module, entry_computation_layout={()->u64[]}

ENTRY %ConstantUnsignedNoOverflow () -> u64[] {
  ROOT %constant = u64[] constant(9223372036854775807)
}

)"
},
// CustomCallWithLayoutConstraints
{
"CustomCallWithLayoutConstraints",
R"(HloModule CustomCallWithLayoutConstraints, entry_computation_layout={(f32[42,2,3]{0,1,2},f32[123,4]{0,1})->f32[1,2,3]{0,2,1}}

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
R"(HloModule CustomCallWithLayoutConstraintsNoOperands, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCallWithLayoutConstraints () -> f32[1,2,3] {
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(), custom_call_target="baz", operand_layout_constraints={}
}

)"
},
// CustomCallWithLayoutConstraintsTupleShapes
{
"CustomCallWithLayoutConstraintsTupleShapes",
R"(HloModule CustomCallWithLayoutConstraintsTupleShapes, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}),f32[123,4]{0,1})->(f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0})}

ENTRY %CustomCallWithLayoutConstraints (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[1,2,3], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={(f32[2,2]{1,0}, f32[42,2,3]{2,0,1}), f32[123,4]{1,0}}
}

)"
},
// CustomCallWithHasSideEffect
{
"CustomCallWithHasSideEffect",
R"(HloModule CustomCallWithHasSideEffect, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}),f32[123,4]{0,1})->(f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0})}

ENTRY %CustomCallWithHasSideEffect (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[1,2,3], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", custom_call_has_side_effect=true
}

)"
},
// CustomCallWithAliasing
{
"CustomCallWithAliasing",
R"(HloModule CustomCallWithAliasing, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}),f32[123,4]{0,1})->(f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2})}

ENTRY %CustomCallWithAliasing (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[123,4], f32[2,2], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", output_to_operand_aliasing={{0}: (1, {}), {1}: (0, {0})}
}

)"
},
// CustomCall with schedule.
{
"CustomCallWithSchedule",
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  %custom-call.0 = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo", schedule=SCHEDULE_EARLIEST
  ROOT %custom-call.1 = f32[1,2,3]{0,2,1} custom-call(f32[1,2,3]{0,2,1} %custom-call.0), custom_call_target="bar", schedule=SCHEDULE_LATEST
}

)"
},
// CustomCall that returns a status.
{
"CustomCallWithStatusReturningVersion",
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call.1 = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo", api_version=API_VERSION_STATUS_RETURNING
}

)"
},
// Parse c64 literal
{
"ParseC64Literal",
R"(HloModule ParseC64Literal, entry_computation_layout={()->c64[2]{0}}

ENTRY %ParseC64Literal () -> c64[2] {
  ROOT %c = c64[2]{0} constant({(1, 2), (-inf, nan)})
}

)"
},
// Parse c128 literal
{
"ParseC128Literal",
R"(HloModule ParseC128Literal, entry_computation_layout={()->c128[2]{0}}

ENTRY %ParseC128Literal () -> c128[2] {
  ROOT %c = c128[2]{0} constant({(1, 2), (-inf, nan)})
}

)"
},
// Indexed Conditional
{
"IndexedConditional",
R"(HloModule indexed_conditional, entry_computation_layout={()->f32[]}

%Negate (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %negate = f32[] negate(f32[] %x)
}

%Identity (y: f32[]) -> f32[] {
  %y = f32[] parameter(0)
  ROOT %copy = f32[] copy(f32[] %y)
}

%Floor (z: f32[]) -> f32[] {
  %z = f32[] parameter(0)
  ROOT %floor = f32[] floor(f32[] %z)
}

ENTRY %Parameters1.v4 () -> f32[] {
  %constant = s32[] constant(1)
  %constant.1 = f32[] constant(56)
  %constant.2 = f32[] constant(12)
  %constant.3 = f32[] constant(13)
  ROOT %conditional = f32[] conditional(s32[] %constant, f32[] %constant.1, f32[] %constant.2, f32[] %constant.3), branch_computations={%Negate, %Identity, %Floor}
}

)"
},
// rng-get-and-update-state
{
"RngGetAndUpdateState",
R"(HloModule rng_get_and_update_state, entry_computation_layout={()->u64[2]{0}}

ENTRY %RngGetAndUpdateState () -> u64[2] {
  ROOT %rng-get-and-update-state = u64[2]{0} rng-get-and-update-state(), delta=4096
}

)"
},
{
"RngBitGenerator",
R"(HloModule gng_bit_generator, entry_computation_layout={(u64[2]{0})->(u64[2]{0}, u32[11,17]{1,0})}

ENTRY %RngBitGenerator (p0: u64[2]) -> (u64[2], u32[11,17]) {
  %p0 = u64[2]{0} parameter(0)
  ROOT %rand = (u64[2]{0}, u32[11,17]{1,0}) rng-bit-generator(u64[2]{0} %p0), algorithm=rng_three_fry
}

)"
},
// Async ops with syntax sugar.
{
"AsyncOpsWithSyntaxSugar",
R"(HloModule AsyncOpsWithSyntaxSugar, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start), custom_call_target="foo"
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update), custom_call_target="foo"
}

)"
},
{
"AsyncOpsWithSyntaxSugarAndGroupId",
R"(HloModule AsyncOpsWithSyntaxSugarAndGroupId, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), async_group_id=3, custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start), async_group_id=3, custom_call_target="foo"
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update), async_group_id=3, custom_call_target="foo"
}

)"
},
// Async ops with syntax sugar and async thread name.
{
"AsyncOpsWithSyntaxSugarAndThreadName",
R"(HloModule AsyncOpsWithSyntaxSugarAndThreadName, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), async_execution_thread="parallel_thread", custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start), async_execution_thread="parallel_thread", custom_call_target="foo"
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update), async_execution_thread="parallel_thread", custom_call_target="foo"
}

)"
},
// HloComputation with thread name as attribute.
{
"HloComputationWithParallelThreadName",
R"(HloModule HloComputationWithParallelThreadName, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), async_execution_thread="parallel_thread", custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start), async_execution_thread="parallel_thread", custom_call_target="foo"
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update), async_execution_thread="parallel_thread", custom_call_target="foo"
}, execution_thread="main_thread"

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
R"(HloModule MapBinaryAdder_module, entry_computation_layout={(f32[4]{0},f32[4]{0})->f32[4]{0}}

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
R"(HloModule ReduceR3ToR2_module, entry_computation_layout={(f32[8,16,256]{2,1,0})->f32[8,16]{1,0}}

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
R"(HloModule TupleReduce, entry_computation_layout={(f32[1024]{0},s32[1024]{0})->(f32[], s32[])}

max_argmax {
  value = f32[] parameter(2)
  prev_max = f32[] parameter(0)
  is_next_larger = pred[] compare(value, prev_max), direction=GE
  max = f32[] select(is_next_larger, value, prev_max)
  index = s32[] parameter(3)
  prev_argmax = s32[] parameter(1)
  argmax = s32[] select(is_next_larger, index, prev_argmax)
  ROOT pair = (f32[], s32[]) tuple(max, argmax)
}

ENTRY reduce_entry {
  values = f32[1024]{0} parameter(0)
  indices = s32[1024]{0} parameter(1)
  init_value = f32[] constant(-inf)
  init_index = s32[] constant(-1)
  ROOT result = (f32[], s32[]) reduce(values, indices, init_value, init_index), dimensions={0}, to_apply=max_argmax
}

)"
},
// infeed/outfeed
{
"InfeedOutfeed",
R"(HloModule outfeed_module, entry_computation_layout={()->((u32[3]{0}, pred[]), token[])}

ENTRY InfeedToOutfeed {
  token0 = token[] after-all()
  infeed = ((u32[3]{0}, pred[]), token[]) infeed(token0)
  infeed.data = (u32[3]{0}, pred[]) get-tuple-element(infeed), index=0
  outfeed = token[] outfeed(infeed.data, token0), outfeed_shape=(u32[3]{0}, pred[])
  ROOT infeed.1 = ((u32[3]{0}, pred[]), token[]) infeed(token0)
  infeed.1.data = (u32[3]{0}, pred[]) get-tuple-element(infeed.1), index=0
  infeed.1.token = token[] get-tuple-element(infeed.1), index=1
  outfeed.1 = token[] outfeed(infeed.1.data, infeed.1.token), outfeed_shape=(u32[3]{0}, pred[])
}

)"
},
// Rng
{
"Rng",
R"(HloModule rng_module, entry_computation_layout={()->f32[8]{0}}

ENTRY Rng {
  constant = f32[] constant(0)
  constant.1 = f32[] constant(1)
  ROOT rng = f32[8]{0} rng(constant, constant.1), distribution=rng_uniform
}

)"
},
// Reduce precision
{
"ReducePrecision",
R"(HloModule reduce_precision, entry_computation_layout={()->f32[1]{0}}

ENTRY ReducePrecision {
  constant = f32[1]{0} constant({3.14159})
  ROOT reduce-precision = f32[1]{0} reduce-precision(constant), exponent_bits=8, mantissa_bits=10
}

)"
},
// Sort (Key)
{
"SortKey",
R"(HloModule sort, entry_computation_layout={(f32[1024]{0})->f32[1024]{0}}

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  x = f32[1024]{0} parameter(0)
  ROOT sorted = f32[1024]{0} sort(x), dimensions={0}, to_apply=compare
}

)"
},
// Sort (Key, Value)
{
"SortKeyValue",
R"(HloModule sort, entry_computation_layout={(f32[1024]{0},s32[1024]{0})->(f32[1024]{0}, s32[1024]{0})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024]{0} parameter(0)
  values = s32[1024]{0} parameter(1)
  ROOT sorted = (f32[1024]{0}, s32[1024]{0}) sort(keys, values), dimensions={0}, to_apply=compare
}

)"
},
// R2 Sort (Key)
{
"SortKeyR2",
R"(HloModule sort, entry_computation_layout={(f32[1024,16]{0,1})->f32[1024,16]{0,1}}

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  x = f32[1024,16]{0,1} parameter(0)
  ROOT sorted = f32[1024,16]{0,1} sort(x), dimensions={0}, to_apply=compare
}

)"
},
// R2 Sort (Key, Value)
{
"SortKeyValueR2",
R"(HloModule sort, entry_computation_layout={(f32[1024,16]{0,1},s32[1024,16]{0,1})->(f32[1024,16]{0,1}, s32[1024,16]{0,1})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024,16]{0,1} parameter(0)
  values = s32[1024,16]{0,1} parameter(1)
  ROOT sorted = (f32[1024,16]{0,1}, s32[1024,16]{0,1}) sort(keys, values), dimensions={0}, to_apply=compare
}

)"
},
// Sort (Key, Value, Value, Value)
{
"SortManyValues",
R"(HloModule sort, entry_computation_layout={(f32[1024,16]{0,1},s32[1024,16]{0,1},u32[1024,16]{0,1},f32[1024,16]{0,1})->(f32[1024,16]{0,1}, s32[1024,16]{0,1}, u32[1024,16]{0,1}, f32[1024,16]{0,1})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.2.lhs = u32[] parameter(4)
  p.2.rhs = u32[] parameter(5)
  p.3.lhs = f32[] parameter(6)
  p.3.rhs = f32[] parameter(7)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024,16]{0,1} parameter(0)
  values.0 = s32[1024,16]{0,1} parameter(1)
  values.1 = u32[1024,16]{0,1} parameter(2)
  values.2 = f32[1024,16]{0,1} parameter(3)
  ROOT sorted = (f32[1024,16]{0,1}, s32[1024,16]{0,1}, u32[1024,16]{0,1}, f32[1024,16]{0,1}) sort(keys, values.0, values.1, values.2), dimensions={0}, to_apply=compare
}

)"
},
// Sort (Key) is_stable=true
{
"SortKeyStable",
R"(HloModule sort, entry_computation_layout={(f32[1024]{0})->f32[1024]{0}}

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  x = f32[1024]{0} parameter(0)
  ROOT sorted = f32[1024]{0} sort(x), dimensions={0}, is_stable=true, to_apply=compare
}

)"
},
// Indexed Conditional
{
"IndexedConditional",
R"(HloModule indexed_conditional, entry_computation_layout={()->f32[]}

Negate {
  x = f32[] parameter(0)
  ROOT negate = f32[] negate(x)
}

Identity {
  y = f32[] parameter(0)
  ROOT copy = f32[] copy(y)
}

Floor {
  z = f32[] parameter(0)
  ROOT floor = f32[] floor(z)
}

ENTRY Parameters1.v4 {
  constant = s32[] constant(1)
  constant.1 = f32[] constant(56)
  constant.2 = f32[] constant(12)
  constant.3 = f32[] constant(13)
  ROOT conditional = f32[] conditional(constant, constant.1, constant.2, constant.3), branch_computations={Negate, Identity, Floor}
}

)"
},
// Predicated Conditional
{
"PredicatedConditional",
R"(HloModule pred_conditional, entry_computation_layout={()->f32[]}

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
R"(HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar"
}

)"
},
// CustomCall with single computation.
{
"CustumCallSingleComp",
R"(HloModule custom_call_with_comp, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

max_F32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar", called_computations={max_F32}
}

)"
},
// CustomCall with multiple computations.
{
"CustumCallMultipleComps",
R"(HloModule custom_call_with_comps, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

max_F32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar", called_computations={max_F32, max_F32}
}

)"
},
// Variables with non-default names
{
"NonDefaultNames",
R"(HloModule add_constants_module, entry_computation_layout={()->f32[]}

ENTRY add_constants {
  foo = f32[] constant(3.14)
  ROOT bar = f32[] add(foo, foo)
}

)"
},
{
"Dot",
R"(HloModule dot, entry_computation_layout={(f32[2,10]{1,0},f32[10,2]{1,0})->f32[2]{0}}

ENTRY dot {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[10,2]{1,0} parameter(1)
  ROOT dot = f32[2]{0} dot(a, b), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={1}, rhs_contracting_dims={0}
}

)"
},
{
"gather",
R"(HloModule gather, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0},s64[10,9,8,7,5]{4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}}

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
R"(HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

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
R"(HloModule CRS_Subgroups, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY AllReduceWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT all-reduce = f32[128,32]{0,1} all-reduce(input), replica_groups={{0,1},{2,3}}, to_apply=add
}

)",
/*replica_count=*/4,
},
// all-reduce with constrained layout
{
"AllReduceWithLayout",
R"(HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT crs = f32[8]{0} all-reduce(input), replica_groups={}, constrain_layout=true, to_apply=add
}

)"
},
// all-reduce with channel-id
{
"AllReduceAllReduce",
R"(HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  crs.1 = f32[8]{0} all-reduce(input), channel_id=1, replica_groups={{0}}, to_apply=add
  ROOT crs.0 = f32[8]{0} all-reduce(input), channel_id=1, replica_groups={{0}}, to_apply=add
}

)"
},
// all-reduce start and done
{
"AllReduceStartAndDone",
R"(HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  crs = f32[8]{0} all-reduce-start(input), replica_groups={}, to_apply=add
  ROOT done = f32[8]{0} all-reduce-done(crs)
}

)"
},
// reduce-scatter
{
"ReduceScatter",
R"(HloModule RS, entry_computation_layout={(f32[8]{0})->f32[4]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT ars = f32[4]{0} reduce-scatter(input), replica_groups={{0,1}}, dimensions={0}, to_apply=add
}

)"
},
// all-gather
{
"AllGather",
R"(HloModule AllGather, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}

ENTRY AllGather {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,128]{0,1} all-gather(input), replica_groups={}, dimensions={1}
}

)"
},
// all-gather with constrained layout
{
"AllGatherWithLayout",
R"(HloModule AllGather, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}

ENTRY AllGather {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,128]{0,1} all-gather(input), replica_groups={}, constrain_layout=true, dimensions={1}
}

)"
},
// all-gather with subgroups
{
"AllGatherWithSubgroups",
R"(HloModule AllGatherWithSubgroups, entry_computation_layout={(f32[128,32]{0,1})->f32[128,64]{0,1}}

ENTRY AllGatherWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,64]{0,1} all-gather(input), replica_groups={{0,1},{2,3}}, dimensions={1}
}

)",
/*replica_count=*/4,
},
// all-to-all
{
"AllToAll",
R"(HloModule AllToAll, entry_computation_layout={(f32[128,32]{0,1})->(f32[128,32]{0,1})}

ENTRY AllToAll {
  input = f32[128,32]{0,1} parameter(0)
  ROOT a2a = (f32[128,32]{0,1}) all-to-all(input), replica_groups={}
}

)"
},
// all-to-all with subgroups
{
"AllToAllWithSubgroups",
R"(HloModule AllToAllWithSubgroups, entry_computation_layout={(f32[128,32]{0,1},f32[128,32]{0,1})->(f32[128,32]{0,1}, f32[128,32]{0,1})}

ENTRY AllToAllWithSubgroups {
  p0 = f32[128,32]{0,1} parameter(0)
  p1 = f32[128,32]{0,1} parameter(1)
  ROOT a2a = (f32[128,32]{0,1}, f32[128,32]{0,1}) all-to-all(p0, p1), replica_groups={{1,2},{3,0}}
}

)",
/*replica_count=*/4,
},
// collective-permute
{
"CollectivePermute",
R"(HloModule CollectivePermute, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}

ENTRY CollectivePermute {
  input = f32[128,32]{0,1} parameter(0)
  ROOT root = f32[128,32]{0,1} collective-permute(input), source_target_pairs={{0,1},{1,2},{2,3}}
}

)",
/*replica_count=*/4
},
// collective-permute with in-place updates
{
"CollectivePermuteInPlaceUpdate",
R"(HloModule CollectivePermuteInPlaceUpdate, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}

ENTRY CollectivePermuteInPlaceUpdate {
  input = f32[128,32]{0,1} parameter(0)
  constant = f32[] constant(1)
  output = f32[128,128]{0,1} broadcast(constant), dimensions={}
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[]) tuple(constant.1, constant.2)
  ROOT root = f32[128,128]{0,1} collective-permute(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{128,32}}
}

)",
/*replica_count=*/4
},
// collective-permute with in-place updates with multiple targets per source
{
"CollectivePermuteInPlaceUpdateMultipleReadWrite",
R"(HloModule CollectivePermuteInPlaceUpdateMultipleReadWrite, entry_computation_layout={(f32[8,8,128]{2,1,0})->f32[8,8,128]{2,1,0}}

ENTRY CollectivePermuteInPlaceUpdate {
  constant.3 = s32[] constant(2)
  constant.1 = s32[] constant(0)
  output_offset.3 = (s32[], s32[], s32[]) tuple(constant.3, constant.1, constant.1)
  constant.4 = s32[] constant(3)
  output_offset.4 = (s32[], s32[], s32[]) tuple(constant.4, constant.1, constant.1)
  input = f32[8,8,128]{2,1,0} parameter(0)
  constant = f32[] constant(1)
  output = f32[8,8,128]{2,1,0} broadcast(constant), dimensions={}
  input_offset.1 = (s32[], s32[], s32[]) tuple(constant.1, constant.1, constant.1)
  constant.2 = s32[] constant(1)
  input_offset.2 = (s32[], s32[], s32[]) tuple(constant.2, constant.1, constant.1)
  input_offset = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(input_offset.1, input_offset.2)
  output_offset = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(input_offset.1, input_offset.2)
  ROOT root = f32[8,8,128]{2,1,0} collective-permute(input, output, input_offset, output_offset), source_target_pairs={{0,1},{1,2},{2,3},{0,3},{2,1},{3,2}}, slice_sizes={{1,8,128},{1,8,128}}
}

)",
/*replica_count=*/4
},
{
"CollectivePermuteInPlaceUpdateTupleMultipleReadWrite",
R"(HloModule hlo_runner_test_0.1, entry_computation_layout={()->(u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)})}

ENTRY hlo_runner_test_0.1 {
  replica_id = u32[] replica-id()
  broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(replica_id), dimensions={}
  tuple.input = (u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple(broadcast.0, broadcast.0)
  constant.1 = u32[] constant(1000)
  broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(constant.1), dimensions={}
  broadcast.2 = u32[4,8,128]{2,1,0:T(2,128)} broadcast(constant.1), dimensions={}
  tuple.output = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple(broadcast.1, broadcast.2)
  constant.2 = s32[] constant(0)
  tuple.2 = (s32[], s32[], s32[]) tuple(constant.2, constant.2, constant.2)
  constant.3 = s32[] constant(1)
  tuple.3 = (s32[], s32[], s32[]) tuple(constant.3, constant.2, constant.2)
  tuple.4 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(tuple.2, tuple.3)
  tuple.7 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(tuple.2, tuple.2)
  tuple.8 = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(tuple.4, tuple.7)
  constant.4 = s32[] constant(2)
  tuple.5 = (s32[], s32[], s32[]) tuple(constant.4, constant.2, constant.2)
  tuple.6 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(tuple.2, tuple.5)
  tuple.9 = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(tuple.4, tuple.6)
  ROOT collective-permute.53 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute(tuple.input, tuple.output, tuple.8, tuple.9), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
}

)",
/*replica_count=*/4
},

// collective-permute tuple with in-place updates
{
"CollectivePermuteTupleInPlaceUpdate",
R"(HloModule CollectivePermuteTupleInPlaceUpdate, entry_computation_layout={(f32[128,32]{0,1})->(f32[128,128]{0,1}, f32[128,128]{0,1})}

ENTRY CollectivePermuteInPlaceUpdate {
  input = f32[128,32]{0,1} parameter(0)
  tuple.input = (f32[128,32]{0,1}, f32[128,32]{0,1}) tuple(input, input)
  constant = f32[] constant(1)
  output = f32[128,128]{0,1} broadcast(constant), dimensions={}
  tuple.output = (f32[128,128]{0,1}, f32[128,128]{0,1}) tuple(output, output)
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[]) tuple(constant.2, constant.1)
  tuple.3 = ((s32[], s32[]), (s32[], s32[])) tuple(tuple.1, tuple.2)
  tuple.4 = (s32[], s32[]) tuple(constant.1, constant.1)
  tuple.5 = (s32[], s32[]) tuple(constant.2, constant.2)
  tuple.6 = ((s32[], s32[]), (s32[], s32[])) tuple(tuple.4, tuple.5)
  ROOT root = (f32[128,128]{0,1}, f32[128,128]{0,1}) collective-permute(tuple.input, tuple.output, tuple.3, tuple.6), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{64,32},{64,32}}
}

)",
/*replica_count=*/4
},
// collective-permute-start and -done with inplace update
{
"CollectivePermuteStartAndDone",
R"(HloModule CollectivePermuteStartAndDone, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}

ENTRY CollectivePermuteStartAndDone {
  input = f32[128,32]{0,1} parameter(0)
  collective-permute-start.1 = (f32[128,32]{0,1}, f32[128,32]{0,1}, u32[], u32[]) collective-permute-start(input), source_target_pairs={{0,1},{1,2},{2,3}}
  ROOT collective-permute-done.1 = f32[128,32]{0,1} collective-permute-done(collective-permute-start.1)
}

)",
/*replica_count=*/4
},
// collective-permute-start and -done
{
"CollectivePermuteStartAndDoneInplaceUpdate",
R"(HloModule CollectivePermuteStartAndDoneInplaceUpdate, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}

ENTRY CollectivePermuteStartAndDoneInplaceUpdate {
  input = f32[128,32]{0,1} parameter(0)
  constant = f32[] constant(1)
  output = f32[128,128]{0,1} broadcast(constant), dimensions={}
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[]) tuple(constant.1, constant.2)
  collective-permute-start.1 = (f32[128,32]{0,1}, f32[128,128]{0,1}, u32[], u32[]) collective-permute-start(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{64,32}}
  ROOT collective-permute-done.1 = f32[128,128]{0,1} collective-permute-done(collective-permute-start.1)
}

)",
/*replica_count=*/4
},
// replica-id
{
"ReplicaId",
R"(HloModule replica-id, entry_computation_layout={()->u32[]}

ENTRY Replica-id {
  ROOT replica-id = u32[] replica-id()
}

)"
},
// partition-id
{
"PartitionId",
R"(HloModule partition-id, entry_computation_layout={()->u32[]}

ENTRY PartitionId {
  ROOT id = u32[] partition-id()
}

)"
},
// Iota
{
"Iota",
R"(HloModule iota, entry_computation_layout={()->f32[100]{0}}

ENTRY Iota {
  ROOT iota = f32[100]{0} iota(), iota_dimension=0
}

)"
},
// custom-call with window, dim_labels and feature_group_count
{
"CustomCallWithWindowAndDimLabelsAndFeatureGroupCount",
R"(HloModule CustomCallWithWindowAndDimLabelsAndFeatureGroupCount, entry_computation_layout={()->f32[100]{0}}

ENTRY Computation {
  ROOT r = f32[100]{0} custom-call(), window={size=2x2}, dim_labels=b01f_01io->b01f, feature_group_count=2, custom_call_target="target"
}

)"
    },
// custom-call with unknown dim labels.
{
"CustomCallWithUnknownDimLabels",
R"(HloModule CustomCallWithUnknownDimLabels, entry_computation_layout={()->f32[100]{0}}

ENTRY Computation {
  ROOT r = f32[100]{0} custom-call(), window={size=2x2}, dim_labels=?b01f_0?1io->b01?f, custom_call_target="target"
}

)"
    },
// is_scheduled=true attribute
{
"ScheduledModule",
R"(HloModule scheduled_module, is_scheduled=true, entry_computation_layout={(f32[1024]{0},s32[1024]{0})->(f32[1024]{0}, s32[1024]{0})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lhs = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024]{0} parameter(0)
  values = s32[1024]{0} parameter(1)
  ROOT sorted = (f32[1024]{0}, s32[1024]{0}) sort(keys, values), dimensions={0}, to_apply=compare
}

)"
    },
// AfterAll with multiple operands
{
"AfterAllWithMultipleOperands",
R"(HloModule AfterAllWithMultipleOperands, entry_computation_layout={(f32[])->token[]}

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
R"(HloModule AddDependency, entry_computation_layout={(f32[])->f32[]}

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
R"(HloModule MinMaxValues, entry_computation_layout={()->c128[2]{0}}

ENTRY MinMaxValues {
  x.s8 = s8[2]{0} constant({-128, 127})
  x.s16 = s16[2]{0} constant({-32768, 32767})
  x.s32 = s32[2]{0} constant({-2147483648, 2147483647})
  x.u8 = u8[2]{0} constant({0, 255})
  x.u16 = u16[2]{0} constant({0, 65535})
  x.u32 = u32[2]{0} constant({0, 4294967295})
  x.f16 = f16[2]{0} constant({-65504, 65504})
  x.bf16 = bf16[2]{0} constant({-3.39e+38, 3.39e+38})
  x.f32 = f32[2]{0} constant({-3.40282e+38, 3.40282e+38})
  x.f64 = f64[2]{0} constant({-1.79769e+308, 1.79769e+308})
  x.c64 = c64[2]{0} constant({(-3.40282e+38, 3.40282e+38), (3.40282e+38, -3.40282e+38)})
  ROOT c.c128 = c128[2]{0} constant({(-1.79769e+308, 1.79769e+308), (1.79769e+308, -1.79769e+308)})
}

)"
},

// Bitcast-convert usage
{
"BitcastConvert",
R"(HloModule BitcastConvert, entry_computation_layout={(f32[100]{0})->u32[100]{0}}

ENTRY BitcastConvertUsage {
  p = f32[100]{0} parameter(0)
  ROOT out = u32[100]{0} bitcast-convert(p)
}

)"
},
});
  // clang-format on
}

std::vector<NonRoundtripTestData> CreateNonRoundtripTestCases() {
  // clang-format off
return std::vector<NonRoundtripTestData>({
{
"SimpleNesting",
R"(HloModule test

ENTRY test {
    ROOT root = add(f32[10] parameter(0), multiply(f32[10] parameter(1), f32[10] parameter(2)))
})",
R"(HloModule test, entry_computation_layout={(f32[10]{0},f32[10]{0},f32[10]{0})->f32[10]{0}}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f32[10]{0} parameter(1)
  parameter.anon.2 = f32[10]{0} parameter(2)
  multiply.anon = f32[10]{0} multiply(parameter.anon.1, parameter.anon.2)
  ROOT root = f32[10]{0} add(parameter.anon, multiply.anon)
})"
},

{
"AmbiguousNames",
R"(HloModule test
ENTRY test {
  add = add(f32[10] parameter(0), f32[10] parameter(1))
  ROOT add2 = add(add, add(add, add))
})",
R"(HloModule test, entry_computation_layout={(f32[10]{0},f32[10]{0})->f32[10]{0}}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f32[10]{0} parameter(1)
  add = f32[10]{0} add(parameter.anon, parameter.anon.1)
  add.anon = f32[10]{0} add(add, add)
  ROOT add2 = f32[10]{0} add(add, add.anon)
})"
},

{
"TupleShapeInsideAnonymousInstr",
R"(HloModule test

ENTRY test {
  ROOT root = get-tuple-element(
    (f32[10], f16[10]) tuple(f32[10] parameter(0), f16[10] parameter(1))
  ), index=0
})",
R"(HloModule test, entry_computation_layout={(f32[10]{0},f16[10]{0})->f32[10]{0}}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f16[10]{0} parameter(1)
  tuple.anon = (f32[10]{0}, f16[10]{0}) tuple(parameter.anon, parameter.anon.1)
  ROOT root = f32[10]{0} get-tuple-element(tuple.anon), index=0
})"
},

{
"MixAnonAndNonAnonOperands",
R"(HloModule test

ENTRY test {
  add = add(f32[10] parameter(0), f32[10] parameter(1))
  ROOT root = tuple(add, add(add, add), add)
})",
R"(HloModule test, entry_computation_layout={(f32[10]{0},f32[10]{0})->(f32[10]{0}, f32[10]{0}, f32[10]{0})}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f32[10]{0} parameter(1)
  add = f32[10]{0} add(parameter.anon, parameter.anon.1)
  add.anon = f32[10]{0} add(add, add)
  ROOT root = (f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(add, add.anon, add)
})"
},

{
"BroadcastOfScalarDoesntNeedDimensionsAttr",
R"(HloModule test

ENTRY test {
  ROOT root = sqrt(f32[10,10] broadcast(f32[] parameter(0)))
})",
R"(HloModule test, entry_computation_layout={(f32[])->f32[10,10]{1,0}}

ENTRY test {
  parameter.anon = f32[] parameter(0)
  broadcast.anon = f32[10,10]{1,0} broadcast(parameter.anon), dimensions={}
  ROOT root = f32[10,10]{1,0} sqrt(broadcast.anon)
})"
},

{
"SparseShape",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(D,C)})->f32[10,10]{1,0:D(D,C)}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)} parameter(0)
})",
},

{
"SparseShapeWithIndexPrimitiveType",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)#(u32)} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(D,C)#(u32)})->f32[10,10]{1,0:D(D,C)#(u32)}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)#(u32)} parameter(0)
})",
},

{
"SparseShapeWithPointerPrimitiveType",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)*(u32)} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(D,C)*(u32)})->f32[10,10]{1,0:D(D,C)*(u32)}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)*(u32)} parameter(0)
})",
},

{
"SparseShapeWithPhysicalShape",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(D,C)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))})->f32[10,10]{1,0:D(D,C)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))} parameter(0)
})",
},

{
"SparseShapeFull",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)#(u64)*(u32)S(42)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(D,C)#(u64)*(u32)S(42)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))})->f32[10,10]{1,0:D(D,C)#(u64)*(u32)S(42)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)#(u64)*(u32)S(42)P((s32[10]{0:T(100)}, s32[10]{0:T(100)}, f32[10]{0:T(100)}))} parameter(0)
})",
},

{
"SparseCOO",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(C+,S)} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(C+,S)})->f32[10,10]{1,0:D(C+,S)}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(C+,S)} parameter(0)
})",
},

{
"SparseCOOUnordered",
R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(C+~,S~)} parameter(0)
})",
R"(HloModule test, entry_computation_layout={(f32[10,10]{1,0:D(C+~,S~)})->f32[10,10]{1,0:D(C+~,S~)}}

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(C+~,S~)} parameter(0)
})",
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
  // Expects "ToString(ParseHloModule(std::string)) == string", that is, parses
  // the string, asserts that it succeeded, stringifies the parsed module, and
  // checks that it equals the original string.
  void ExpectEqual() {
    std::unique_ptr<HloModule> module;
    const std::string& original = GetParam().module_string;
    HloModuleConfig config;
    config.set_replica_count(GetParam().replica_count);
    if (GetParam().enable_verification) {
      auto verified_module = std::make_unique<VerifiedHloModule>(
          GetParam().test_name, config,
          /*verifier_layout_sensitive=*/false,
          /*allow_mixed_precision_in_hlo_verifier=*/true,
          ShapeUtil::ByteSizeOfElements);
      TF_ASSERT_OK(verified_module->ParseHloStringAndVerifyModule(original));
      module = std::move(verified_module);
    } else {
      TF_ASSERT_OK_AND_ASSIGN(module,
                              ParseAndReturnUnverifiedModule(original, config));
    }
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

class HloNonRoundtripParserTest
    : public ::testing::TestWithParam<NonRoundtripTestData> {};
TEST_P(HloNonRoundtripParserTest, Run) {
  auto module = std::make_unique<VerifiedHloModule>(
      GetParam().test_name, HloModuleConfig{},
      /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      ShapeUtil::ByteSizeOfElements);
  TF_ASSERT_OK(
      module->ParseHloStringAndVerifyModule(GetParam().input_module_string));
  EXPECT_EQ(absl::StripAsciiWhitespace(GetParam().output_module_string),
            absl::StripAsciiWhitespace(
                module->ToString(HloPrintOptions::ShortParsable())));
}

INSTANTIATE_TEST_SUITE_P(HloParserTestSuccessInstantiation,
                         HloNonRoundtripParserTest,
                         ::testing::ValuesIn(CreateNonRoundtripTestCases()),
                         NonRoundtripTestDataToString);

class HloParserTest : public ::testing::Test {
 protected:
  static void ExpectHasSubstr(string_view s, string_view expected) {
    EXPECT_TRUE(absl::StrContains(s, expected))
        << "'" << s << "' does not contain '" << expected << "'";
  }
  StatusOr<std::unique_ptr<VerifiedHloModule>> ParseAndReturnVerifiedModule(
      absl::string_view hlo_text) {
    auto module = std::make_unique<VerifiedHloModule>(
        ::testing::UnitTest::GetInstance()->current_test_info()->name(),
        HloModuleConfig(),
        /*verifier_layout_sensitive=*/false,
        /*allow_mixed_precision_in_hlo_verifier=*/true,
        ShapeUtil::ByteSizeOfElements);
    TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
    return std::move(module);
  }
};

TEST_F(HloParserTest, Empty) {
  const std::string original = "";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, Garbage) {
  const std::string original =
      "HloModule thi$ str1ng makes# N0 sen$e @all!*&^%$";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, WrongOpcode) {
  const std::string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: f32[], y: f32[]) -> f32[] {
  %x = f32[]{} parameter(0)
  %y = f32[]{} parameter(1)
  %le = pred[]{} le(f32[]{} %x, f32[]{} %y)
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, MetadataWithCholesky) {
  const std::string original = R"(HloModule metadata_with_cholesky
ENTRY %blabla (a: f32[1,291,291]) -> f32[1,291,291] {
  %a = f32[1,291,291] parameter(0)
  %out = f32[1,291,291] cholesky(f32[1,291,291] %a), lower=true, metadata={op_type="Cholesky" op_name="Cholesky" profile_type={1}}
}
)";
  auto result = ParseAndReturnVerifiedModule(original);
  EXPECT_EQ(OkStatus(), result.status());
  EXPECT_EQ("Cholesky", result.value()
                            ->entry_computation()
                            ->root_instruction()
                            ->metadata()
                            .op_name());
  EXPECT_EQ("Cholesky", result.value()
                            ->entry_computation()
                            ->root_instruction()
                            ->metadata()
                            .op_type());
  EXPECT_EQ(WINDOW, *result.value()
                         ->entry_computation()
                         ->root_instruction()
                         ->metadata()
                         .profile_type()
                         .begin());
}

TEST_F(HloParserTest, WrongShape) {
  const std::string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: g32[]) -> g32[] {
  %x = g32[]{} parameter(0)
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, WrongOperandsSize) {
  const std::string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: f32[]) -> pred[] {
  %x = f32[]{} parameter(0)
  %eq = pred[]{} compare(f32[]{} %x), direction=EQ
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, OperandNotFound) {
  const std::string original = R"(HloModule operand_not_found:
ENTRY %blabla (x: f32[]) -> pred[] {
  %x = f32[]{} parameter(0)
  %eq = pred[]{} compare(f32[]{} %x, f32[]{} %y), direction=EQ
}
)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, MoreConstants) {
  const std::string original = R"(HloModule SelectScalarS32True_module

ENTRY %SelectScalarS32True.v4 () -> s32[] {
  %constant.2 = pred[] constant(true)
  %constant.1 = s32[] constant(-42), sharding={replicated}
  %constant = s32[] constant(42)
  %select = s32[] select(pred[] %constant.2, s32[] %constant.1, s32[] %constant)
}

)";
  auto result = ParseAndReturnVerifiedModule(original);
  TF_EXPECT_OK(result.status());
  // Constant instructions have no name. The string will be parsed successfully
  // but the constant names will not be exactly the same.
}

TEST_F(HloParserTest, ConfigurationField) {
  const std::string original = R"(HloModule AModule
ENTRY %configuration_test() -> s32[] {
  %constant = s32[] constant(42), backend_config="foo bar"
})";
  auto result = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(result.status());
  EXPECT_EQ("foo bar", result.value()
                           ->entry_computation()
                           ->root_instruction()
                           ->raw_backend_config_string());
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_1) {
  const std::string original = R"(HloModule some_2_module

ENTRY %some_2 () -> f32[2] {
  ROOT %constant = f32[2]{0} constant({1,{2}})
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects nested array in rank 1, but sees larger");
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_2) {
  const std::string original = R"(HloModule some_2x3_module

ENTRY %some_2x3 () -> f32[2,3] {
  ROOT %constant = f32[2,3]{1,0} constant({1, 2, 3, 4, 5, 6})
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects nested array in rank 2, but sees 1");
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_3) {
  const std::string original = R"(HloModule some_2x3x2_module

ENTRY %some_2x3x2 () -> f32[2,3,2] {
  ROOT %constant = f32[2,3,2]{2,1,0} constant({{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}})
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects 3 elements in the [0]th element");
}

TEST_F(HloParserTest, ConstantF16Overflow) {
  const std::string original =
      R"(HloModule ConstantF16Overflow_module

ENTRY %ConstantF16Overflow.v4 () -> f16[] {
  ROOT %constant = f16[] constant(-65520)
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "is out of range for literal's primitive type F16");
}

TEST_F(HloParserTest, ConstantBf16NoOverflow) {
  // 65505 is in range for bf16.
  const std::string original = R"(
  HloModule test_module
  ENTRY test {
    ROOT c = bf16[] constant(-65505)
  })";
  EXPECT_EQ(OkStatus(), ParseAndReturnVerifiedModule(original).status());
}

TEST_F(HloParserTest, ConstantBf16Overflow) {
  // 1e100 is out of range for bf16.
  const std::string original = R"(
  HloModule test_module
  ENTRY test {
    ROOT c = bf16[] constant(1e100)
  })";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "out of range");
}

TEST_F(HloParserTest, ConstantUnsignedUnderflow) {
  const std::string original = R"(
      HloModule ConstantUnsignedUnderflow_module
      ENTRY %ConstantUnsignedUnderflow () -> u64[] {
        ROOT %constant = u64[] constant(-1)
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_EQ(OkStatus(), result.status());
}

TEST_F(HloParserTest, ConstantUnsignedOverflow) {
  const std::string original = R"(
      HloModule ConstantUnsignedOverflow_module
      ENTRY %ConstantUnsignedOverflow () -> u32[] {
        ROOT %constant = u32[] constant(4294967296)
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "is out of range for literal's primitive type U32");
}

TEST_F(HloParserTest, ConstantUnsignedInt64Overflow) {
  const std::string original = R"(
      HloModule ConstantUnsignedOverflow_module
      ENTRY %ConstantUnsignedOverflow () -> u64[] {
        ROOT %constant = u64[] constant(9223372036854775808)
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_EQ(OkStatus(), result.status());
}

TEST_F(HloParserTest, ConstantC64Overflow) {
  const std::string original = R"(
      HloModule test_module
      ENTRY test () -> c64[] {
        ROOT c = c64[] constant((1e100, 0))
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, ConstantC64Underflow) {
  const std::string original = R"(
      HloModule test_module
      ENTRY test () -> c64[] {
        ROOT c = c64[] constant((0, -1e100))
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, ConstantF64Overflow) {
  const std::string original = R"(
      HloModule test_module
      ENTRY test {
        ROOT c = f64[] constant(1.8e308)
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, ConstantF64Underflow) {
  const std::string original = R"(
      HloModule test_module
      ENTRY test {
        ROOT c = f64[] constant(-1.8e308)
      })";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_NE(OkStatus(), result.status());
}

TEST_F(HloParserTest, ConstantWithExp) {
  const std::string original = R"(HloModule ConstantWithExp_module

ENTRY %ConstantWithExp.v4 () -> f32[] {
  %constant.1 = f32[] constant(3e+2)
}

)";
  auto result = ParseAndReturnVerifiedModule(original);
  TF_EXPECT_OK(result.status());
  // The string will be parsed successfully but the output strings are not
  // exactly the same, because "3e2" is parsed into value 300 and will be
  // printed as "300".
}

TEST_F(HloParserTest, ShortConstant) {
  const std::string original =
      R"(HloModule ShortConstant_module, entry_computation_layout={()->f32[67,89]{1,0}}

ENTRY %ShortConstant.v4 () -> f32[67,89] {
  ROOT %constant.1 = f32[67,89]{1,0} constant({...})
}

)";
  auto result = ParseAndReturnVerifiedModule(original);
  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.value()->ToString(HloPrintOptions()), original);
}

TEST_F(HloParserTest, NegativeNan) {
  const std::string original =
      R"(HloModule NegativeNan_module, entry_computation_layout={()->bf16[2]{0}}

ENTRY %NegativeNan () -> bf16[2] {
  ROOT %constant = bf16[2]{0} constant({-nan, -nan})
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_EQ(OkStatus(), result.status());
  EXPECT_EQ(result.value()->ToString(HloPrintOptions()), original);
}

TEST_F(HloParserTest, NanPayload) {
  const std::string original =
      R"(HloModule NanPayload_module, entry_computation_layout={()->bf16[2]{0}}

ENTRY %NanPayload () -> bf16[2] {
  ROOT %constant = bf16[2]{0} constant({-nan(0x7f), -nan(0x3f)})
}

)";
  auto result = ParseAndReturnUnverifiedModule(original);
  EXPECT_EQ(OkStatus(), result.status());
  EXPECT_EQ(result.value()->ToString(HloPrintOptions()), original);
}

TEST_F(HloParserTest, AttributesAnyOrder) {
  const std::string original = R"(HloModule any_order_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,4,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,4,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), feature_group_count=1, sharding={maximal device=1}, backend_config="foo", dim_labels=b0f_0io->b0f, window={pad=1_1 size=1}
}

)";
  TF_EXPECT_OK(ParseAndReturnVerifiedModule(original).status());
}

TEST_F(HloParserTest, InvalidDimLabels) {
  std::string prefix = R"(HloModule invalid_dim_labels_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1} )";
  std::string suffix = R"(
}

)";

  ExpectHasSubstr(ParseAndReturnUnverifiedModule(
                      absl::StrCat(prefix, ",dim_labels=00_01->10", suffix))
                      .status()
                      .error_message(),
                  "expects unique");

  ExpectHasSubstr(ParseAndReturnUnverifiedModule(
                      absl::StrCat(prefix, ",dim_labels=012_0123->210", suffix))
                      .status()
                      .error_message(),
                  "must have same number of spatial dimensions");

  ExpectHasSubstr(ParseAndReturnUnverifiedModule(
                      absl::StrCat(prefix, ",dim_labels=013_0123->210", suffix))
                      .status()
                      .error_message(),
                  "expects [0-2bf?]");
}

TEST_F(HloParserTest, UnexpectedAttribute) {
  const std::string original = R"(HloModule unexpected_attr_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, calls=%recv
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "unexpected attribute \"calls\"");
}

TEST_F(HloParserTest, MissingAttribute) {
  const std::string original = R"(HloModule missing_attr_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(-2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0)
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "attribute channel_id is expected but not seen");
}

TEST_F(HloParserTest, PredecessorUndefined) {
  const std::string original = R"(HloModule pre_not_found_module

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15
  %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15
  ROOT %constant = f32[] constant(2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, control-predecessors={%done}
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "'done' is not defined");
}

TEST_F(HloParserTest, SliceAllowOmitStride1) {
  const std::string original = R"(HloModule slice_module

ENTRY %slice.v2 (p0: f32[3,3,4,4]) -> f32[3,3,2,4] {
  %p0 = f32[3,3,4,4]{3,2,1,0} parameter(0)
  ROOT %slice = f32[3,3,2,4]{3,2,1,0} slice(f32[3,3,4,4]{3,2,1,0} %p0), slice={[0:3], [0:3], [0:4:2], [0:4]}
}

)";
  TF_EXPECT_OK(ParseAndReturnVerifiedModule(original).status());
}

TEST_F(HloParserTest, PaddingConfigIsNotWindowPad) {
  const std::string original = R"(HloModule window_pad_module

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), dim_labels=b0f_0io->b0f, window={pad=1_1_0 size=1}
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "expects padding_low and padding_high separated by '_'");
}

TEST_F(HloParserTest, CommaBetweenSubAttributes) {
  const std::string original = R"(HloModule test_comma_module

ENTRY %test_comma.v4 () -> f32[] {
  ROOT %constant = f32[] constant(-4.2), metadata={source_line=5, op_type="::const"}
}

)";
  TF_EXPECT_OK(ParseAndReturnVerifiedModule(original).status());
}

TEST_F(HloParserTest, ComputationShapeDoesNotMatchRootShape) {
  const std::string original = R"(HloModule custom_call:

ENTRY %CustomCall () -> f32[1] {
  %constant = f32[1]{0} constant({12345})
  ROOT %foo = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar"
})";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Shape of computation CustomCall, f32[1], is not compatible "
      "with that of its root instruction foo, f32[1,2,3]");
}

TEST_F(HloParserTest, EntryComputationWithLayout) {
  const std::string original = R"(HloModule layout:
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

  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
  auto program_layout = module.value()->entry_computation_layout();
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
  const std::string original = R"(HloModule no_entry:
c1 {
  const1 = f32[1]{0} constant({12345})
}
c2 {
  const2 = f32[1]{0} constant({67890})
})";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
  EXPECT_EQ(module.value()->entry_computation()->name(), "c2");
}

TEST_F(HloParserTest, NoRoot) {
  const std::string original = R"(HloModule no_root:
ENTRY consts {
  first = f32[1]{0} constant({12345})
  last = f32[1]{0} constant({67890})
})";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
  EXPECT_EQ(module.value()->entry_computation()->root_instruction()->name(),
            "last");
}

TEST_F(HloParserTest, Comments) {
  const std::string original = R"(/* module description. */
HloModule comments:

ENTRY /*comment*/ c1 {
  /* blah */
  ROOT const1 = /*foo*/f32[1]{0} constant({12345 /*bar*/})
  /* comment */
}

/* something else */

)";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, MultilineComments) {
  const std::string original = R"(HloModule multiline_comment:
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
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, UnterminatedComment) {
  const std::string original = R"(HloModule unterminated_comment:
ENTRY c1 {
/* unterminated
  ROOT const1 = f32[1]{0} constant({12345})
})";
  // Verify that the error message points to the beginning of the unterminated
  // comment.
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "/* unterminated\n^");
}

TEST_F(HloParserTest, SlashSlashComments) {
  const std::string original = R"(HloModule slash_slash_comment:
// Garbage
ENTRY c1 {
  // Foo bar
  ROOT const1 = f32[1]{0} constant({12345}) // Something else
})";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, SlashSlashCommentMsDosEolFormat) {
  const std::string original =
      "HloModule slash_slash_comment:\r\n// Garbage\r\nENTRY c1 {\r\n// Foo "
      "bar\r\nROOT const1 = f32[1]{0} constant({12345}) // Something else\r\n}";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, SlashSlashCommentMacEolFormat) {
  const std::string original =
      "HloModule slash_slash_comment:\r// Garbage\rENTRY c1 {\r// Foo "
      "bar\rROOT const1 = f32[1]{0} constant({12345}) // Something else\r}";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
}

TEST_F(HloParserTest, MultipleEntries) {
  const std::string original = R"(HloModule multiple_entries:
ENTRY c1 {
  const1 = f32[1]{0} constant({12345})
}
ENTRY c2 {
  const2 = f32[1]{0} constant({67890})
})";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "expects only one ENTRY");
}

TEST_F(HloParserTest, SimpleAliasing) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0}: (0, {0}, must-alias), {1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
  )";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
  std::unique_ptr<HloModule> parsed_module = std::move(module).value();
  EXPECT_EQ(parsed_module->input_output_alias_config().GetAliasedOutput(0, {0}),
            ShapeIndex{0});

  EXPECT_TRUE(
      parsed_module->input_output_alias_config().ParameterMustAlias(0, {0}));
  EXPECT_EQ(parsed_module->input_output_alias_config().GetAliasedOutput(0, {1}),
            ShapeIndex{1});
  EXPECT_FALSE(
      parsed_module->input_output_alias_config().ParameterMustAlias(0, {1}));
}

TEST_F(HloParserTest, NestedAliasing) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0, 0}: (0, {0}), {1, 1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  %t0 = (f32[], f32[]) tuple(%p0, %p1)
  %t1 = (f32[], f32[]) tuple(%p0, %p1)
  ROOT %out = ((f32[], f32[]), (f32[], f32[])) tuple(%t0, %t1)
}
  )";
  auto module = ParseAndReturnVerifiedModule(original);
  TF_ASSERT_OK(module.status());
  std::unique_ptr<HloModule> parsed_module = std::move(module).value();
  EXPECT_EQ(parsed_module->input_output_alias_config().GetAliasedOutput(0, {0}),
            ShapeIndex({0, 0}));
  EXPECT_EQ(parsed_module->input_output_alias_config().GetAliasedOutput(0, {1}),
            ShapeIndex({1, 1}));
}

TEST_F(HloParserTest, AliasingWrongIndex) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0 : (0, {0}), {1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
  )";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Expects '}' at the end of ShapeIndex");
}

TEST_F(HloParserTest, AliasingShapeIndexNotNumerical) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0, a}: (0, {0}), {1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
  )";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "expects integer");
}

TEST_F(HloParserTest, AliasingWrongFormatNoColon) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0, 0}: (0, {0}), (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
  )";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Expects '{' at the start of ShapeIndex");
}

TEST_F(HloParserTest, AliasingWrongFormatTwoColons) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0}: (0, {0}): {0, 1}, {1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
  )";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Expects '}' at the end of aliasing description");
}

TEST_F(HloParserTest, AliasingWrongFormatAlphaParam) {
  const std::string original = R"(
HloModule Module, input_output_alias={ {0, a}: (zero, {0}), {1}: (0, {1}) }

ENTRY entry {
  %p = (f32[], f32[]) parameter(0)
  %p0 = f32[] get-tuple-element((f32[], f32[]) %p), index=0
  %p1 = f32[] get-tuple-element((f32[], f32[]) %p), index=1
  ROOT %out = (f32[], f32[]) tuple(%p0, %p1)
}
  )";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "expects integer");
}

TEST_F(HloParserTest, MultipleRoots) {
  const std::string original = R"(HloModule multiple_roots:
ENTRY consts {
  ROOT const1 = f32[1]{0} constant({12345})
  ROOT const2 = f32[1]{0} constant({12345})
})";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "one computation should have only one ROOT");
}

TEST_F(HloParserTest, ComputationExists) {
  const std::string original = R"(HloModule comp_exists
comp {
  const1 = f32[1]{0} constant({12345})
}
comp {
  const2 = f32[1]{0} constant({67890})
})";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      R"(was parsing 2:1: error: computation previously defined here
comp {
^)");
}

TEST_F(HloParserTest, CrossComputationLookup) {
  const std::string original = R"(HloModule cross_computation_lookup:
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
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "was parsing 8:39: error: instruction does not exist: aparam");
}

TEST_F(HloParserTest, SameNameDiffComputations) {
  const std::string original = R"(HloModule same_names:
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(original));
  ASSERT_NE(module->entry_computation(), nullptr);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Reduce()));
}

TEST_F(HloParserTest, ParseSharding) {
  const std::string original = "{maximal device=42}";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, ParseSharding(original));
  EXPECT_EQ(sharding.ToString(), original);
}

TEST_F(HloParserTest, ParseShardingPartialReplication) {
  const std::string original = "{devices=[2,2]0,1,2,3 last_tile_dim_replicate}";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, ParseSharding(original));
  EXPECT_EQ(sharding.ToString(), original);
  Array<int64_t> group_tiling({2});
  group_tiling(0) = 0;
  group_tiling(1) = 1;
  std::vector<int64_t> group0_members({0, 1});
  std::vector<int64_t> group1_members({2, 3});
  EXPECT_EQ(
      HloSharding::PartialTile(group_tiling, {group0_members, group1_members})
          .ToString(),
      original);
}

TEST_F(HloParserTest, ParseShardingSubGroup) {
  const std::string original =
      "{devices=[2,2,2,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
      "last_tile_dims={manual, replicated}}";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, ParseSharding(original));
  EXPECT_EQ(sharding.ToString(), original);
  Array<int64_t> tile_assignment({2, 2, 2, 2});
  tile_assignment.FillIota(0);
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::MANUAL,
                                                  OpSharding::REPLICATED};
  EXPECT_EQ(HloSharding::Subgroup(tile_assignment, subgroup_types).ToString(),
            original);
}

TEST_F(HloParserTest, ParseFrontendAttributes) {
  const std::string original =
      R"({attr_a="test_a",attr_b="b",attr_c="s64",attr_d="a/b"})";
  TF_ASSERT_OK_AND_ASSIGN(FrontendAttributes frontend_attributes,
                          ParseFrontendAttributes(original));
  EXPECT_EQ(FrontendAttributesToString(frontend_attributes), original);
}

TEST_F(HloParserTest, ParseWindow) {
  Window original = window_util::MakeWindow({1, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(Window parsed,
                          ParseWindow(window_util::ToString(original)))
  EXPECT_EQ(window_util::ToString(original), window_util::ToString(parsed));
}

TEST_F(HloParserTest, ParseConvolutionDimensionNumbers) {
  const std::string original = "b0f_0io->b0f";
  TF_ASSERT_OK_AND_ASSIGN(ConvolutionDimensionNumbers dnums,
                          ParseConvolutionDimensionNumbers(original));
  EXPECT_EQ(original, ConvolutionDimensionNumbersToString(dnums));
}

TEST_F(HloParserTest, ParseConvolutionDimensionNumbersWithUnknownDims) {
  const std::string original = "b0?f_?0?io->?b?0?f";
  TF_ASSERT_OK_AND_ASSIGN(ConvolutionDimensionNumbers dnums,
                          ParseConvolutionDimensionNumbers(original));
  EXPECT_EQ(original, ConvolutionDimensionNumbersToString(dnums));
}

TEST_F(HloParserTest, ParseReplicaGroups) {
  const std::string original = "{{0,1},{2,3}}";
  TF_ASSERT_OK_AND_ASSIGN(std::vector<ReplicaGroup> replica_groups,
                          ParseReplicaGroupsOnly(original));
  EXPECT_EQ(original, ReplicaGroupsToString(replica_groups));
}

TEST_F(HloParserTest, ParsePaddingConfigNoInteriorPadding) {
  const std::string original = "0_1x2_3";
  TF_ASSERT_OK_AND_ASSIGN(PaddingConfig dnums, ParsePaddingConfig(original));
  EXPECT_EQ(original, PaddingConfigToString(dnums));
}

TEST_F(HloParserTest, ParsePaddingConfigInteriorPadding) {
  const std::string original = "0_1_0x2_3_4";
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
  const std::string original = R"(HloModule nontuple_infeed:
ENTRY nontuple_infeed {
  token0 = token[] after-all()
  ROOT infeed = pred[] infeed(token0)
})";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "infeed must have a non-empty tuple shape");
}

TEST(HloParserSingleOpTest, SingleOp) {
  const std::string text =
      "%multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, "
      "f32[2,4]{1,0} %x)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
}

TEST(HloParserSingleOpTest, SingleOpNoShapeProducesError) {
  const std::string text =
      "multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)";
  StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnUnverifiedModule(text);
  ASSERT_TRUE(!module.status().ok());
  LOG(INFO) << "Status: " << module.status();
  EXPECT_THAT(module.status().ToString(),
              HasSubstr("expects '=' in instruction"));
}

TEST(HloParserSingleOpTest, SingleOpNoOperandShapesProducesError) {
  const std::string text = "%multiply = f32[2,4]{1,0} multiply(%broadcast, %x)";
  StatusOr<std::unique_ptr<HloModule>> module =
      ParseAndReturnUnverifiedModule(text);
  ASSERT_TRUE(!module.status().ok());
  LOG(INFO) << "Status: " << module.status();
  EXPECT_THAT(module.status().ToString(),
              HasSubstr("Operand had no shape in HLO text"));
}

TEST(HloParserSingleOpTest, SingleOpNoNames) {
  const std::string text =
      "%multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0}, f32[2,4]{1,0})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
}

TEST(HloParserSingleOpTest, CanonicalOp) {
  const std::string text =
      "f32[2,4]{1,0} multiply(f32[2,4]{1,0}, f32[2,4]{1,0})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
  EXPECT_EQ(
      computation->root_instruction()->ToString(HloPrintOptions::Canonical()),
      text);
}

TEST(HloParserSingleOpTest, CanonicalOpWithNested) {
  const std::string text =
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

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_EQ(
      computation->root_instruction()->ToString(HloPrintOptions::Canonical()),
      text);
}

TEST(HloParserSingleOpTest, CanonicalOpIndexedConditionalInlinedBranches) {
  const std::string text =
      R"(f32[5,10]{1,0} conditional(s32[], f32[5,10]{1,0}, f32[5,10]{1,0}, f32[5,10]{1,0}), branch_computations={
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  ROOT tmp_1 = f32[5,10]{1,0} ceil(f32[5,10]{1,0} tmp_0)
},
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  ROOT tmp_1 = f32[5,10]{1,0} floor(f32[5,10]{1,0} tmp_0)
},
{
  tmp_0 = f32[5,10]{1,0} parameter(0)
  ROOT tmp_1 = f32[5,10]{1,0} copy(f32[5,10]{1,0} tmp_0)
}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_EQ(
      computation->root_instruction()->ToString(HloPrintOptions::Canonical()),
      text);
}

TEST(HloParserSingleOpTest, SingleOpWithNested) {
  const std::string text =
      R"(%fusion = f32[3,2,1,1]{3,2,1,0} fusion(f32[3,2,1,1]{3,2,1,0} %p0, f32[2]{0} %p1), kind=kLoop, calls=
{
  %param_0 = f32[3,2,1,1]{3,2,1,0} parameter(0)
  %param_1 = f32[2]{0} parameter(1)
  %broadcast = f32[3,2,1,1]{3,2,1,0} broadcast(f32[2]{0} %param_1), dimensions={1}
  ROOT %subtract = f32[3,2,1,1]{3,2,1,0} subtract(f32[3,2,1,1]{3,2,1,0} %param_0, f32[3,2,1,1]{3,2,1,0} %broadcast)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
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
  const std::string text =
      R"(reduce = f32[] reduce(f32[10], f32[]), dimensions={1}, to_apply=
{
  result = f32[] add(f32[] x, f32[] y)
})";
  auto status = ParseAndReturnUnverifiedModule(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("does not exist: x"));
}

TEST(HloParserSingleOpTest, SingleOpWithNested_NoLhs) {
  const std::string text =
      R"(reduce = f32[] reduce(f32[10], f32[]), dimensions={1}, to_apply=
{
  f32[] add(f32[] x, f32[] y)
})";
  auto status = ParseAndReturnUnverifiedModule(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("expects name"));
}

TEST(HloParserSingleOpTest, SingleOpWithNested_NoOperandName) {
  const std::string text =
      R"(reduce = f32[] reduce(f32[10], f32[]), dimensions={1}, to_apply=
{
  result = f32[] add(f32[], f32[])
})";
  auto status = ParseAndReturnUnverifiedModule(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("expects name"));
}

TEST(HloParserSingleOpTest, ConvolutionTrivialFeatureGroupCount) {
  const std::string text =
      R"(%convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(text));
  const HloComputation* computation = module->entry_computation();
  ASSERT_NE(computation, nullptr);
  EXPECT_THAT(computation->root_instruction(),
              GmockMatch(m::Convolution(m::Parameter(0), m::Parameter(1))));
  auto* convolution =
      Cast<HloConvolutionInstruction>(computation->root_instruction());
  EXPECT_EQ(convolution->feature_group_count(), 1);
}

TEST(HloParserSingleOpTest, MultipleOpsProducesError) {
  const std::string text = R"(
    param = f32[2,5,1,3] parameter(0)
    transpose = f32[1,5,2,3] transpose(param), dimensions={2,1,0,3}
  )";
  auto status = ParseAndReturnUnverifiedModule(text).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("Expected eof"));
}

TEST_F(HloParserTest, IsScheduledIsFalse) {
  const std::string text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  ASSERT_FALSE(module->has_schedule());
}

TEST_F(HloParserTest, IsScheduledNotPresent) {
  const std::string text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  ASSERT_FALSE(module->has_schedule());
}

TEST_F(HloParserTest, IsScheduledIsTrue) {
  const std::string text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());
  EXPECT_EQ(module->schedule().sequences().size(), 1);
  ASSERT_TRUE(
      module->schedule().is_computation_scheduled(module->entry_computation()));
  EXPECT_THAT(
      module->schedule().sequence(module->entry_computation()).instructions(),
      ElementsAre(GmockMatch(m::Parameter()), GmockMatch(m::Broadcast()),
                  GmockMatch(m::Parameter()), GmockMatch(m::Multiply()),
                  GmockMatch(m::Parameter()), GmockMatch(m::Add())));
}

TEST_F(HloParserTest, IsScheduledIsTrueDifferentOrder) {
  // As above but in with a different schedule order.
  const std::string text = R"(
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());
  EXPECT_EQ(module->schedule().sequences().size(), 1);
  ASSERT_TRUE(
      module->schedule().is_computation_scheduled(module->entry_computation()));
  EXPECT_THAT(
      module->schedule().sequence(module->entry_computation()).instructions(),
      ElementsAre(GmockMatch(m::Parameter()), GmockMatch(m::Parameter()),
                  GmockMatch(m::Parameter()), GmockMatch(m::Broadcast()),
                  GmockMatch(m::Multiply()), GmockMatch(m::Add())));
}

TEST_F(HloParserTest, CustomCallWrongNumberofOperandConstraints) {
  const std::string original =
      R"(HloModule CustomCallWrongNumberofOperandConstraints

ENTRY %CustomCallWrongNumberofOperandConstraints (p0: f32[42,2,3], p1: f32[123,4]) -> f32[1,2,3] {
  %p0 = f32[42,2,3]{0,1,2} parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = f32[1,2,3]{0,1,2} custom-call(f32[42,2,3]{0,1,2} %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={f32[42,2,3]{0,1,2}}
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Expected 2 operand layout constraints, 1 given");
}

TEST_F(HloParserTest, CustomCallIncompatibleOperandConstraints) {
  const std::string original =
      R"(HloModule CustomCallIncompatibleOperandConstraints

ENTRY %CustomCallIncompatibleOperandConstraints (p0: f32[42,2,3], p1: f32[123,4]) -> f32[1,2,3] {
  %p0 = f32[42,2,3]{0,1,2} parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = f32[1,2,3]{0,1,2} custom-call(f32[42,2,3]{0,1,2} %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={f32[42,2,3]{0,1,2}, f32[555,5]{1,0}}
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "operand 1 is not compatible with operand shape");
}

TEST_F(HloParserTest, CustomCallWithNonexistentVersion) {
  const std::string original = R"(HloModule custom_call

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call.1 = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo", api_version=API_VERSION_THAT_DOESNT_EXIST
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Unknown API version");
}

TEST_F(HloParserTest, CustomCallWithUnspecifiedVersion) {
  const std::string original = R"(HloModule custom_call

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call.1 = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo", api_version=API_VERSION_UNSPECIFIED
}

)";
  ExpectHasSubstr(
      ParseAndReturnUnverifiedModule(original).status().error_message(),
      "Invalid API version");
}

TEST_F(HloParserTest, AllowShapeWhitespace) {
  const std::string text = R"(
HloModule module

ENTRY entry {
  ROOT root = f32[ 1, 2,3, 4, 5]{0, 1, 2,3, 4 } parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
}

TEST_F(HloParserTest, ShapeMismatchInOperand) {
  const std::string text = R"(
HloModule foobar

ENTRY %entrycomp (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2] parameter(0)
  %constant.1 = f32[2,2] constant({{1, 2}, {3, 4}})
  ROOT %add.1 = f32[2,2] add(f32[2,2] %p, f32[2,5] %constant.1)
}
)";

  ExpectHasSubstr(ParseAndReturnUnverifiedModule(text).status().error_message(),
                  "The declared operand shape f32[2,5]{1,0} is not compatible"
                  " with the shape of the operand instruction f32[2,2]{1,0}.");
}

TEST_F(HloParserTest, ParseShapeStringR2F32) {
  std::string shape_string = "f32[123,456]";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShape(F32, {123, 456});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringTupleOfArrays) {
  std::string shape_string = "(f32[1572864],s8[5120,1024])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1572864}),
                                 ShapeUtil::MakeShape(S8, {5120, 1024})});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringNestedTuple) {
  std::string shape_string = "(f32[1],(f32[2], token[]), opaque[], f32[3])";
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
  std::string shape_string = "f32[123,456]{0,1}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithDenseLayout(F32, {123, 456}, {0, 1});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseShapeStringWithTilingLayout) {
  // One tile.
  std::string shape_string = "f32[123,456]{0,1:T(2,128)}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithDenseLayout(F32, {123, 456}, {0, 1},
                                                       {Tile({2, 128})});
  EXPECT_EQ(expected, actual)
      << "expected: " << ShapeUtil::HumanStringWithLayout(expected)
      << "actual:   " << ShapeUtil::HumanStringWithLayout(actual);

  // Tile with negative dimension size for combining dimensions.
  shape_string = "f32[123,456,789]{0,1,2:T(2, * , 128)}";
  TF_ASSERT_OK_AND_ASSIGN(actual, ParseShape(shape_string));
  expected = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {123, 456, 789}, {0, 1, 2},
      {Tile({2, Tile::kCombineDimension, 128})});
  EXPECT_EQ(expected, actual)
      << "expected: " << ShapeUtil::HumanStringWithLayout(expected)
      << "actual:   " << ShapeUtil::HumanStringWithLayout(actual);

  // Two tiles.
  shape_string = "bf16[123,456,789]{2,1,0:T(2,*,128)(2,1)}";
  TF_ASSERT_OK_AND_ASSIGN(actual, ParseShape(shape_string));
  expected = ShapeUtil::MakeShapeWithDenseLayout(
      BF16, {123, 456, 789}, {2, 1, 0},
      {Tile({2, Tile::kCombineDimension, 128}), Tile({2, 1})});
  EXPECT_EQ(expected, actual)
      << "expected: " << ShapeUtil::HumanStringWithLayout(expected)
      << "actual:   " << ShapeUtil::HumanStringWithLayout(actual);

  // Wrong minor_to_major.
  shape_string = "f32[123,456,789]{1:T(2, * , 128)}";
  auto result = ParseShape(shape_string);
  ExpectHasSubstr(result.status().error_message(),
                  "Dimensions size is 3, but minor to major size is 1.");
}

TEST_F(HloParserTest, ParseShapeStringWithMemorySpaceLayout) {
  // Tile, element size, and memory space.
  std::string shape_string = "pred[123,456]{1,0:T(2,128)S(3)}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithDenseLayout(PRED, {123, 456}, {1, 0},
                                                       {Tile({2, 128})}, 3);
  EXPECT_EQ(expected, actual)
      << "expected: " << ShapeUtil::HumanStringWithLayout(expected)
      << "actual:   " << ShapeUtil::HumanStringWithLayout(actual);

  // Element size and memory space.
  shape_string = "pred[123,456]{1,0:S(3)}";
  TF_ASSERT_OK_AND_ASSIGN(actual, ParseShape(shape_string));
  expected =
      ShapeUtil::MakeShapeWithDenseLayout(PRED, {123, 456}, {1, 0}, {}, 3);
  EXPECT_EQ(expected, actual)
      << "expected: " << ShapeUtil::HumanStringWithLayout(expected)
      << "actual:   " << ShapeUtil::HumanStringWithLayout(actual);

  // Memory space only.
  shape_string = "pred[123,456]{1,0:S(3)}";
  TF_ASSERT_OK_AND_ASSIGN(actual, ParseShape(shape_string));
  expected =
      ShapeUtil::MakeShapeWithDenseLayout(PRED, {123, 456}, {1, 0}, {}, 3);
  EXPECT_EQ(expected, actual)
      << "expected: " << ShapeUtil::HumanStringWithLayout(expected)
      << "actual:   " << ShapeUtil::HumanStringWithLayout(actual);
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
  std::string shape_strings[] = {"f32[123,456]foobar{0,1}", "f32[123,456]{foo}",
                                 "f32[123,456]dense{foo}"};
  for (const std::string& shape_string : shape_strings) {
    StatusOr<Shape> result = ParseShape(shape_string);
    ASSERT_FALSE(result.ok()) << "shape: " << shape_string;
  }
}

TEST_F(HloParserTest, ParseDynamicArray) {
  std::string shape_string = "f32[123,<=456]";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeShape(F32, {123, 456}, {false, true});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseDynamicTuple) {
  std::string shape_string = "(f32[42], u32[<=123,<=456])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ParseShape(shape_string));
  Shape expected = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeShape(U32, {123, 456}, {true, true})});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST_F(HloParserTest, ParseInvalidDimLevel) {
  constexpr std::string_view shape_string = "f32[123]{0:D(D+~)}";
  StatusOr<Shape> result = ParseShape(shape_string);
  ASSERT_THAT(
      result.status(),
      tsl::testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          testing::HasSubstr(
              "invalid DimLevelType/unique/ordered combination in shape")));
}

TEST_F(HloParserTest, NegativeParameterNumber) {
  const std::string hlo_string = "par0 = f32[3,5] parameter(-1)";
  auto result = ParseAndReturnUnverifiedModule(hlo_string);
  ASSERT_FALSE(result.status().ok());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("parameter number must be >= 0"));
}

TEST_F(HloParserTest, WrongNumberOfParameterLeafBuffersInReplication) {
  const std::string hlo_string =
      "par0 = (f32[3,5], f32[]) parameter(0), "
      "parameter_replication={true,false,true}";
  auto result = ParseAndReturnUnverifiedModule(hlo_string);
  ASSERT_FALSE(result.status().ok());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("parameter has 2 leaf buffers, but "
                        "parameter_replication has 3 elements"));
}

TEST_F(HloParserTest, CheckIndexedConditionalDimension) {
  const char* const hlo_string = R"(
  HloModule Module

  branch0 {
    tparam = f32[4] parameter(0)
    ROOT tgte1 = f32[4] ceil(tparam)
  }

  branch1 {
    fparam = f32[4] parameter(0)
    ROOT fgte1 = f32[4] floor(fparam)
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    b0 = s32[2] parameter(1)
    ROOT conditional = f32[4] conditional(b0, p0, p0),
      branch_computations={branch0, branch1}
  }
  )";
  auto result = ParseAndReturnUnverifiedModule(hlo_string);
  EXPECT_NE(OkStatus(), result.status());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("The first operand must be a scalar"));
}

TEST_F(HloParserTest, CheckIndexedConditionalElementType) {
  const char* const hlo_string = R"(
  HloModule Module

  branch0 {
    tparam = f32[4] parameter(0)
    ROOT tgte1 = f32[4] ceil(tparam)
  }

  branch1 {
    fparam = f32[4] parameter(0)
    ROOT fgte1 = f32[4] floor(fparam)
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    b0 = f32[] parameter(1)
    ROOT conditional = f32[4] conditional(b0, p0, p0),
      branch_computations={branch0, branch1}
  }
  )";
  auto result = ParseAndReturnUnverifiedModule(hlo_string);
  EXPECT_NE(OkStatus(), result.status());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("The first operand must be a scalar of PRED or S32"));
}

TEST_F(HloParserTest,
       CheckPredicatedConditionalRequiresTrueAndFalseComputation) {
  const char* const hlo_string = R"(
  HloModule Module

  branch0 {
    tparam = f32[4] parameter(0)
    ROOT tgte1 = f32[4] ceil(tparam)
  }

  branch1 {
    fparam = f32[4] parameter(0)
    ROOT fgte1 = f32[4] floor(fparam)
  }

  ENTRY entry {
    p0 = f32[4] parameter(0)
    b0 = pred[] parameter(1)
    ROOT conditional = f32[4] conditional(b0, p0, p0),
      branch_computations={branch0, branch1}
  }
  )";
  auto result = ParseAndReturnUnverifiedModule(hlo_string);
  EXPECT_NE(OkStatus(), result.status());
  EXPECT_THAT(result.status().error_message(),
              HasSubstr("unexpected attribute \"branch_computations\""));
}

// Result shape inference tests cases.
TEST_F(HloParserTest, InferUnaryShape) {
  constexpr char text[] = R"(HloModule InferUnaryShapeTest
ENTRY InferUnaryShape {
  a = f32[2,10]{1,0} parameter(0)
  ROOT v = abs(a)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
}

TEST_F(HloParserTest, InferBinaryShape) {
  constexpr char text[] = R"(HloModule InferBinaryShapeTest
ENTRY InferBinaryShape {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[2,10]{1,0} parameter(1)
  ROOT sum = add(a, b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_TRUE(ShapeUtil::Equal(
      module->entry_computation()->ComputeProgramShape().result(),
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 10}, {1, 0})));
}

TEST_F(HloParserTest, InferTernaryShape) {
  constexpr char text[] = R"(HloModule InferTernaryShapeTest
ENTRY InferTernaryShape {
  p = pred[] constant(true)
  f = s32[] constant(-42)
  t = s32[] constant(42)
  ROOT select = select(p, f, t)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_TRUE(ShapeUtil::Equal(
      module->entry_computation()->ComputeProgramShape().result(),
      ShapeUtil::MakeScalarShape(S32)));
}

TEST_F(HloParserTest, InferDotShape) {
  constexpr char text[] = R"(HloModule InferDotShapeTest
ENTRY InferDotShape {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[10,2]{1,0} parameter(1)
  ROOT dot = dot(a, b), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_TRUE(ShapeUtil::Equal(
      module->entry_computation()->ComputeProgramShape().result(),
      ShapeUtil::MakeShape(F32, {2}, {0})));
}

TEST_F(HloParserTest, InferTupleShape) {
  constexpr char text[] = R"(HloModule InferTupleShapeTest
ENTRY InferTupleShape () -> s32[2,3] {
  c0 = f32[3]{0} constant({1, 2, 3})
  c1 = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 } })
  tuple = tuple(c0, c1)
  ROOT get = get-tuple-element(tuple), index=1, sharding={maximal device=0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_TRUE(ShapeUtil::Equal(
      module->entry_computation()->ComputeProgramShape().result(),
      ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 3}, {1, 0})));
}

TEST_F(HloParserTest, InferShapeMixedExplicitShape) {
  constexpr char text[] = R"(HloModule InferUnaryShapeTest
Negate {
  x = f32[] parameter(0)
  ROOT negate = negate(x)
}

Identity {
  y = f32[] parameter(0)
  ROOT copy = copy(y)
}

ENTRY InferUnaryShape {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  p = pred[] parameter(2)
  c = f32[] add(a, b)
  ROOT conditional = conditional(p, a, c), true_computation=Negate, false_computation=Identity
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(text));
  EXPECT_TRUE(ShapeUtil::Equal(
      module->entry_computation()->ComputeProgramShape().result(),
      ShapeUtil::MakeScalarShape(F32)));
}

TEST_F(HloParserTest, CheckAliasPassthroughParams) {
  const char* const hlo_string = R"(
HloModule TestModule, alias_passthrough_params=true

ENTRY TestComputation {
    p0 = f16[2048,1024] parameter(0)
    p1 = f16[2048,1024] parameter(1)
    ROOT root = (f16[2048,1024], f16[2048,1024]) tuple(p0, p1)
}
)";
  auto result = ParseAndReturnVerifiedModule(hlo_string);
  TF_EXPECT_OK(result.status());
  EXPECT_TRUE(result.value()->config().alias_passthrough_params());
}

TEST_F(HloParserTest, NestedBroadcastWithoutDimensionsAttribute) {
  const char* const hlo_string = R"(
HloModule test
ENTRY test {
    ROOT root = sqrt(f32[10,10] broadcast(f32[10] parameter(0)))
}
)";
  auto result = ParseAndReturnVerifiedModule(hlo_string);
  EXPECT_NE(OkStatus(), result.status());
  EXPECT_THAT(result.status().error_message(), HasSubstr("dimensions"));
}

TEST_F(HloParserTest, InvalidDimLevelType) {
  const std::string original = R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(X,C)} parameter(0)
})";
  EXPECT_THAT(ParseAndReturnUnverifiedModule(original).status(),
              tsl::testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("expected a DimLevelType abbreviation")));
}

TEST_F(HloParserTest, InvalidDimLevelTypeCount) {
  const std::string original = R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(C)} parameter(0)
})";
  EXPECT_THAT(
      ParseAndReturnUnverifiedModule(original).status(),
      tsl::testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("Dimensions size is 2, but dim level types size is 1")));
}

TEST_F(HloParserTest, RejectSparseTiles) {
  const std::string original = R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:D(D,C)T(128,8)} parameter(0)
})";
  EXPECT_THAT(ParseAndReturnUnverifiedModule(original).status(),
              tsl::testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Layout has tiles, but is for a sparse array")));
}

TEST_F(HloParserTest, RejectDensePhysicalShape) {
  const std::string original = R"(HloModule test

ENTRY test {
  ROOT root = f32[10,10]{1,0:T(128,8)P(f32[10,10])} parameter(0)
})";
  EXPECT_THAT(
      ParseAndReturnUnverifiedModule(original).status(),
      tsl::testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr(
              "Layout has physical shape, but is not for a sparse array")));
}

}  // namespace
}  // namespace xla
