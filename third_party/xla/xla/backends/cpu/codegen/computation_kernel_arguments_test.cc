/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/computation_kernel_arguments.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"

namespace xla::cpu {
namespace {

using ::testing::Pointwise;

MATCHER(SameShape, "") {
  return std::get<0>(arg).shape == std::get<1>(arg).shape;
}

// Two structurally identical computations, call_a and call_b, each running a
// counter loop. $MUL_A/$MUL_B define the loop multiplier `m` and $ARGS_B the
// operands of the call to call_b. The copy of the loop counter init mirrors
// copy insertion; a constant cannot share the loop state buffer.
constexpr absl::string_view kHloTemplate = R"(
HloModule m

cond_a {
  p = (s32[], f32[16]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(5)
  ROOT lt = pred[] compare(i, n), direction=LT
}

body_a {
  p = (s32[], f32[16]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  v = f32[16] get-tuple-element(p), index=1
  one = s32[] constant(1)
  i1 = s32[] add(i, one)
  $MUL_A
  v1 = f32[16] multiply(v, m)
  ROOT t = (s32[], f32[16]) tuple(i1, v1)
}

call_a {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  s = f32[16] add(p0, p1)
  zero = s32[] constant(0)
  zero_copy = s32[] copy(zero)
  init = (s32[], f32[16]) tuple(zero_copy, s)
  w = (s32[], f32[16]) while(init), condition=cond_a, body=body_a
  ROOT r = f32[16] get-tuple-element(w), index=1
}

cond_b {
  p = (s32[], f32[16]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  n = s32[] constant(5)
  ROOT lt = pred[] compare(i, n), direction=LT
}

body_b {
  p = (s32[], f32[16]) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  v = f32[16] get-tuple-element(p), index=1
  one = s32[] constant(1)
  i1 = s32[] add(i, one)
  $MUL_B
  v1 = f32[16] multiply(v, m)
  ROOT t = (s32[], f32[16]) tuple(i1, v1)
}

call_b {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  s = f32[16] add(p0, p1)
  zero = s32[] constant(0)
  zero_copy = s32[] copy(zero)
  init = (s32[], f32[16]) tuple(zero_copy, s)
  w = (s32[], f32[16]) while(init), condition=cond_b, body=body_b
  ROOT r = f32[16] get-tuple-element(w), index=1
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  p2 = f32[16] parameter(2)
  p3 = f32[16] parameter(3)
  k0 = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
  k1 = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
  c0 = f32[16] call($ARGS_A), to_apply=call_a
  c1 = f32[16] call($ARGS_B), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(c0, c1)
}
)";

constexpr absl::string_view kScalarMul = R"(
  c = f32[] constant(2.5)
  m = f32[16] broadcast(c), dimensions={})";

constexpr absl::string_view kLargeMul =
    "m = f32[16] constant({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
    "16})";

struct Calls {
  ComputationKernelArguments a;
  ComputationKernelArguments b;
};

class ComputationKernelArgumentsTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<Calls> CreateForCalls(absl::string_view mul_a,
                                       absl::string_view mul_b,
                                       absl::string_view args_a = "p0, p1",
                                       absl::string_view args_b = "p2, p3") {
    return CreateForModule(
        absl::StrReplaceAll(kHloTemplate, {{"$MUL_A", mul_a},
                                           {"$MUL_B", mul_b},
                                           {"$ARGS_A", args_a},
                                           {"$ARGS_B", args_b}}));
  }

  // Assigns buffers for `hlo` and creates arguments for call_a and call_b.
  absl::StatusOr<Calls> CreateForModule(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(module_, ParseAndReturnVerifiedModule(hlo));
    // As in CpuCompiler::CreateBufferAssignment.
    BufferAssigner::Options options;
    options.allocate_buffers_for_constants = true;
    ABSL_ASSIGN_OR_RETURN(
        assignment_,
        BufferAssigner::Run(
            module_.get(),
            std::make_unique<DependencyHloOrdering>(module_.get()),
            [](const BufferValue& buffer) {
              return CpuExecutable::ShapeSizeBytes(buffer.shape());
            },
            &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; },
            std::move(options)));
    ABSL_ASSIGN_OR_RETURN(ComputationKernelArguments a, Create("call_a"));
    ABSL_ASSIGN_OR_RETURN(ComputationKernelArguments b, Create("call_b"));
    return Calls{std::move(a), std::move(b)};
  }

  absl::StatusOr<ComputationKernelArguments> Create(absl::string_view name) {
    return ComputationKernelArguments::Create(
        *module_->GetComputationWithName(name), *assignment_);
  }

 private:
  AliasInfo alias_info_;
  std::unique_ptr<HloModule> module_;
  std::unique_ptr<BufferAssignment> assignment_;
};

TEST_F(ComputationKernelArgumentsTest, IdenticalCallsShareFingerprint) {
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForCalls(kScalarMul, kScalarMul));

  ASSERT_TRUE(calls.a.fingerprint().has_value());
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
  EXPECT_NE(calls.a.arguments(), calls.b.arguments());
  EXPECT_THAT(calls.a.arguments(), Pointwise(SameShape(), calls.b.arguments()));
  EXPECT_THAT(calls.a.results(), Pointwise(SameShape(), calls.b.results()));

  ASSERT_OK_AND_ASSIGN(ComputationKernelArguments again, Create("call_a"));
  EXPECT_EQ(again.arguments(), calls.a.arguments());
  EXPECT_EQ(again.results(), calls.a.results());
  EXPECT_EQ(again.fingerprint(), calls.a.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, DifferentCallsNoShare) {
  const struct {
    absl::string_view name;
    std::string mul_a;
    std::string mul_b;
    absl::string_view args_b = "p2, p3";
  } cases[] = {
      {"float constant", std::string(kScalarMul),
       absl::StrReplaceAll(kScalarMul, {{"2.5", "3.5"}})},
      {"large constant", std::string(kLargeMul),
       absl::StrReplaceAll(kLargeMul, {{"16}", "17}"}})},
      {"operand sharing", std::string(kScalarMul), std::string(kScalarMul),
       "p0, p0"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    ASSERT_OK_AND_ASSIGN(Calls calls,
                         CreateForCalls(c.mul_a, c.mul_b, "p0, p1", c.args_b));
    ASSERT_TRUE(calls.a.fingerprint().has_value());
    ASSERT_TRUE(calls.b.fingerprint().has_value());
    EXPECT_NE(calls.a.fingerprint(), calls.b.fingerprint());
  }
}

// k0 and k1 are equal but distinct constant allocations, whose literals the
// emitter may bake into the code.
TEST_F(ComputationKernelArgumentsTest, ExternalConstantOperand) {
  ASSERT_OK_AND_ASSIGN(
      Calls same, CreateForCalls(kScalarMul, kScalarMul, "k0, p1", "k0, p3"));
  ASSERT_TRUE(same.a.fingerprint().has_value());
  EXPECT_EQ(same.a.fingerprint(), same.b.fingerprint());

  ASSERT_OK_AND_ASSIGN(Calls different, CreateForCalls(kScalarMul, kScalarMul,
                                                       "k0, p1", "k1, p3"));
  ASSERT_TRUE(different.a.fingerprint().has_value());
  ASSERT_TRUE(different.b.fingerprint().has_value());
  EXPECT_NE(different.a.fingerprint(), different.b.fingerprint());
}

// The sum may reuse the buffer of the temporary operand of call_b, but not the
// read-only entry parameter of call_a.
TEST_F(ComputationKernelArgumentsTest, OperandReusePatternNoShare) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  ROOT s = f32[16] add(p0, p1)
}

call_b {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  ROOT s = f32[16] add(p0, p1)
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  p0_copy = f32[16] copy(p0)
  c0 = f32[16] call(p0, p1), to_apply=call_a
  c1 = f32[16] call(p0_copy, p1), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));

  // The first argument is p0 and the only result is s.
  ASSERT_EQ(calls.a.results().size(), 1);
  ASSERT_EQ(calls.b.results().size(), 1);
  EXPECT_NE(calls.a.arguments()[0].slice, calls.a.results()[0].slice);
  EXPECT_EQ(calls.b.arguments()[0].slice, calls.b.results()[0].slice);
  ASSERT_TRUE(calls.a.fingerprint().has_value());
  ASSERT_TRUE(calls.b.fingerprint().has_value());
  EXPECT_NE(calls.a.fingerprint(), calls.b.fingerprint());
}

// Values of the reducers are thread-local, outside the kernel table.
TEST_F(ComputationKernelArgumentsTest, ThreadLocalBuffersShareFingerprint) {
  constexpr absl::string_view kHlo = R"(
HloModule m

add_a {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT s = f32[] add(x, y)
}

call_a {
  p0 = f32[16] parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p0, zero), dimensions={0}, to_apply=add_a
}

add_b {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT s = f32[] add(x, y)
}

call_b {
  p0 = f32[16] parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p0, zero), dimensions={0}, to_apply=add_b
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  c0 = f32[] call(p0), to_apply=call_a
  c1 = f32[] call(p1), to_apply=call_b
  ROOT t = (f32[], f32[]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  ASSERT_TRUE(calls.a.fingerprint().has_value());
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

// The emitted kernel owns a private RNG state, which callers must not share.
TEST_F(ComputationKernelArgumentsTest, RngStateNotReusable) {
  constexpr absl::string_view kHlo = R"(
HloModule m

rng_a {
  ROOT s = u64[2] rng-get-and-update-state(), delta=16
}

call_a {
  ROOT c = u64[2] call(), to_apply=rng_a
}

rng_b {
  ROOT s = u64[2] rng-get-and-update-state(), delta=16
}

call_b {
  ROOT c = u64[2] call(), to_apply=rng_b
}

ENTRY e {
  c0 = u64[2] call(), to_apply=call_a
  c1 = u64[2] call(), to_apply=call_b
  ROOT t = (u64[2], u64[2]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.fingerprint(), std::nullopt);
  EXPECT_EQ(calls.b.fingerprint(), std::nullopt);
}

// The RNG state op is in a reducer of a fusion, which the walk does not enter.
TEST_F(ComputationKernelArgumentsTest, RngStateInFusionNotReusable) {
  constexpr absl::string_view kHlo = R"(
HloModule m

add_a {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  s = u64[2] rng-get-and-update-state(), delta=16
  ROOT r = f32[] add(x, y)
}

fused_a {
  p0 = f32[16] parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p0, zero), dimensions={0}, to_apply=add_a
}

call_a {
  p0 = f32[16] parameter(0)
  ROOT f = f32[] fusion(p0), kind=kLoop, calls=fused_a
}

add_b {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  s = u64[2] rng-get-and-update-state(), delta=16
  ROOT r = f32[] add(x, y)
}

fused_b {
  p0 = f32[16] parameter(0)
  zero = f32[] constant(0)
  ROOT r = f32[] reduce(p0, zero), dimensions={0}, to_apply=add_b
}

call_b {
  p0 = f32[16] parameter(0)
  ROOT f = f32[] fusion(p0), kind=kLoop, calls=fused_b
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  c0 = f32[] call(p0), to_apply=call_a
  c1 = f32[] call(p1), to_apply=call_b
  ROOT t = (f32[], f32[]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.fingerprint(), std::nullopt);
  EXPECT_EQ(calls.b.fingerprint(), std::nullopt);
}

// The inner tuple {0} of p0 is not the top-level buffer of any instruction.
TEST_F(ComputationKernelArgumentsTest, NestedTupleShareFingerprint) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p0 = ((s32[], f32[16]), f32[16]) parameter(0)
  ROOT g = f32[16] get-tuple-element(p0), index=1
}

call_b {
  p0 = ((s32[], f32[16]), f32[16]) parameter(0)
  ROOT g = f32[16] get-tuple-element(p0), index=1
}

ENTRY e {
  p0 = ((s32[], f32[16]), f32[16]) parameter(0)
  p1 = ((s32[], f32[16]), f32[16]) parameter(1)
  c0 = f32[16] call(p0), to_apply=call_a
  c1 = f32[16] call(p1), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  ASSERT_TRUE(calls.a.fingerprint().has_value());
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, UnusedParameterSharesFingerprint) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  ROOT n = f32[16] negate(p0)
}

call_b {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  ROOT n = f32[16] negate(p0)
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  c0 = f32[16] call(p0, p1), to_apply=call_a
  c1 = f32[16] call($ARGS_B), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(absl::StrReplaceAll(
                                        kHlo, {{"$ARGS_B", "p1, p0"}})));
  EXPECT_EQ(calls.a.arguments().size(), 2);
  ASSERT_TRUE(calls.a.fingerprint().has_value());
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());

  ASSERT_OK_AND_ASSIGN(Calls aliased, CreateForModule(absl::StrReplaceAll(
                                          kHlo, {{"$ARGS_B", "p0, p0"}})));
  ASSERT_TRUE(aliased.b.fingerprint().has_value());
  EXPECT_NE(aliased.a.fingerprint(), aliased.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, RootParameterSharesFingerprint) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  ROOT p0 = f32[16] parameter(0)
}

call_b {
  ROOT p0 = f32[16] parameter(0)
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  c0 = f32[16] call(p0), to_apply=call_a
  c1 = f32[16] call(p1), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.arguments().size(), 1);
  ASSERT_TRUE(calls.a.fingerprint().has_value());
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

}  // namespace
}  // namespace xla::cpu
