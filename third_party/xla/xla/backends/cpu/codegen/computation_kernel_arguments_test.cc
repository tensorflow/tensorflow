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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
#include "xla/service/shaped_slice.h"

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
    EXPECT_NE(calls.a.fingerprint(), calls.b.fingerprint());
  }
}

// k0 and k1 are equal but distinct constant allocations, whose literals the
// emitter may bake into the code.
TEST_F(ComputationKernelArgumentsTest, ExternalConstantOperand) {
  ASSERT_OK_AND_ASSIGN(
      Calls same, CreateForCalls(kScalarMul, kScalarMul, "k0, p1", "k0, p3"));
  EXPECT_EQ(same.a.fingerprint(), same.b.fingerprint());

  ASSERT_OK_AND_ASSIGN(Calls different, CreateForCalls(kScalarMul, kScalarMul,
                                                       "k0, p1", "k1, p3"));
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

  // In call_a, p0 and p1 are read only entry parameters (2 arguments, 1
  // result). In call_b, the sum reuses the temporary operand p0_copy, so
  // p0_copy and s share one written result region (1 argument, 1 result).
  EXPECT_EQ(calls.a.arguments().size(), 2);
  EXPECT_EQ(calls.a.results().size(), 1);
  EXPECT_EQ(calls.b.arguments().size(), 1);
  EXPECT_EQ(calls.b.results().size(), 1);
  EXPECT_NE(calls.a.arguments()[0].slice, calls.a.results()[0].slice);
  EXPECT_NE(calls.a.fingerprint(), calls.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, MergeOverlapping) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/64, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/64, /*color=*/0);
  auto s0 = [&](int64_t off, int64_t size) {
    return BufferAllocation::Slice(&alloc0, off, size);
  };
  auto s1 = [&](int64_t off, int64_t size) {
    return BufferAllocation::Slice(&alloc1, off, size);
  };

  EXPECT_THAT(ComputationKernelArguments::MergeOverlapping(
                  {s0(0, 16), s0(0, 1), s0(4, 4), s0(8, 0)}),
              ::testing::ElementsAre(s0(0, 16)));
  EXPECT_THAT(
      ComputationKernelArguments::MergeOverlapping({s0(0, 4), s0(4, 4)}),
      ::testing::ElementsAre(s0(0, 4), s0(4, 4)));
  EXPECT_THAT(ComputationKernelArguments::MergeOverlapping(
                  {s0(0, 8), s0(6, 4), s0(9, 3), s1(0, 4)}),
              ::testing::ElementsAre(s0(0, 12), s1(0, 4)));
}

TEST_F(ComputationKernelArgumentsTest, InPlaceUpdateIsOneResultRegion) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p0 = f32[16] parameter(0)
  u = f32[2] constant({1, 2})
  i = s32[] constant(4)
  ROOT d = f32[16] dynamic-update-slice(p0, u, i)
}

call_b {
  p0 = f32[16] parameter(0)
  u = f32[2] constant({1, 2})
  i = s32[] constant(4)
  ROOT d = f32[16] dynamic-update-slice(p0, u, i)
}

ENTRY e {
  p0 = f32[16] parameter(0)
  c0 = f32[16] copy(p0)
  r0 = f32[16] call(c0), to_apply=call_a
  c1 = f32[16] copy(p0)
  r1 = f32[16] call(c1), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(r0, r1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  ASSERT_EQ(calls.a.results().size(), 1);
  for (const ShapedSlice& arg : calls.a.arguments()) {
    EXPECT_FALSE(arg.slice.OverlapsWith(calls.a.results()[0].slice));
  }
}

// The callee only reads the caller's temporaries through `get-tuple-element`,
// so their regions are arguments.
TEST_F(ComputationKernelArgumentsTest, ReadOnlyBuffersAreArguments) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p = (f32[16], f32[16]) parameter(0)
  a = f32[16] get-tuple-element(p), index=0
  b = f32[16] get-tuple-element(p), index=1
  ROOT s = f32[16] add(a, b)
}

call_b {
  p = (f32[16], f32[16]) parameter(0)
  a = f32[16] get-tuple-element(p), index=0
  b = f32[16] get-tuple-element(p), index=1
  ROOT s = f32[16] add(a, b)
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  t0 = f32[16] negate(p0)
  t1 = f32[16] negate(p1)
  tup = (f32[16], f32[16]) tuple(t0, t1)
  c0 = f32[16] call(tup), to_apply=call_a
  c1 = f32[16] call(tup), to_apply=call_b
  ROOT t = (f32[16], f32[16], f32[16], f32[16]) tuple(c0, c1, t0, t1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.arguments().size(), 2);
  EXPECT_EQ(calls.a.results().size(), 1);
}

// call_a updates donated p0 in place. Non donated p1 is read only, so call_b
// writes a temporary and must not reuse call_a's kernel.
TEST_F(ComputationKernelArgumentsTest,
       DonatedAndNonDonatedEntryParametersDiffer) {
  constexpr absl::string_view kHlo = R"(
HloModule m, input_output_alias={ {0}: (0, {}, may-alias) }

call_a {
  p0 = f32[16] parameter(0)
  u = f32[2] constant({1, 2})
  i = s32[] constant(4)
  ROOT d = f32[16] dynamic-update-slice(p0, u, i)
}

call_b {
  p0 = f32[16] parameter(0)
  u = f32[2] constant({1, 2})
  i = s32[] constant(4)
  ROOT d = f32[16] dynamic-update-slice(p0, u, i)
}

ENTRY e {
  p0 = f32[16] parameter(0)
  p1 = f32[16] parameter(1)
  r0 = f32[16] call(p0), to_apply=call_a
  r1 = f32[16] call(p1), to_apply=call_b
  ROOT t = (f32[16], f32[16]) tuple(r0, r1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_NE(calls.a.fingerprint(), calls.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, ParametersAreDisjoint) {
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForCalls(kScalarMul, kScalarMul));
  std::vector<BufferAllocation::Slice> all;
  for (const ShapedSlice& s : calls.a.arguments()) {
    all.push_back(s.slice);
  }
  for (const ShapedSlice& s : calls.a.results()) {
    all.push_back(s.slice);
  }
  for (size_t i = 0; i < all.size(); ++i) {
    for (size_t j = i + 1; j < all.size(); ++j) {
      EXPECT_FALSE(all[i].OverlapsWith(all[j]));
    }
  }
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
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
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
  EXPECT_EQ(calls.a.arguments().size(), 3);
  EXPECT_EQ(calls.a.results().size(), 0);
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, TupleConstantSharesFingerprint) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p0 = f32[4] parameter(0)
  c = (f32[4], f32[4]) constant(({1, 2, 3, 4}, {5, 6, 7, 8}))
  g = f32[4] get-tuple-element(c), index=1
  ROOT s = f32[4] add(p0, g)
}

call_b {
  p0 = f32[4] parameter(0)
  c = (f32[4], f32[4]) constant(({1, 2, 3, 4}, {5, 6, 7, 8}))
  g = f32[4] get-tuple-element(c), index=1
  ROOT s = f32[4] add(p0, g)
}

ENTRY e {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  c0 = f32[4] call(p0), to_apply=call_a
  c1 = f32[4] call(p1), to_apply=call_b
  ROOT t = (f32[4], f32[4]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, OpaqueAndEmptySubshapesSkipped) {
  constexpr absl::string_view kHlo = R"(
HloModule m

call_a {
  p0 = (opaque[], f32[0], f32[4]) parameter(0)
  x = f32[4] get-tuple-element(p0), index=2
  ROOT y = f32[4] negate(x)
}

call_b {
  p0 = (opaque[], f32[0], f32[4]) parameter(0)
  x = f32[4] get-tuple-element(p0), index=2
  ROOT y = f32[4] negate(x)
}

ENTRY e {
  p0 = (opaque[], f32[0], f32[4]) parameter(0)
  p1 = (opaque[], f32[0], f32[4]) parameter(1)
  c0 = f32[4] call(p0), to_apply=call_a
  c1 = f32[4] call(p1), to_apply=call_b
  ROOT t = (f32[4], f32[4]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.arguments().size(), 1);
  EXPECT_EQ(calls.a.results().size(), 1);
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

TEST_F(ComputationKernelArgumentsTest, UnwrittenWhileConditionStaysArgument) {
  constexpr absl::string_view kHlo = R"(
HloModule m

cond_a {
  s = (pred[]) parameter(0)
  ROOT c = pred[] get-tuple-element(s), index=0
}

body_a {
  s = (pred[]) parameter(0)
  c = pred[] get-tuple-element(s), index=0
  ROOT r = (pred[]) tuple(c)
}

call_a {
  flag = pred[] parameter(0)
  init = (pred[]) tuple(flag)
  w = (pred[]) while(init), condition=cond_a, body=body_a
  ROOT out = pred[] get-tuple-element(w), index=0
}

cond_b {
  s = (pred[]) parameter(0)
  ROOT c = pred[] get-tuple-element(s), index=0
}

body_b {
  s = (pred[]) parameter(0)
  c = pred[] get-tuple-element(s), index=0
  ROOT r = (pred[]) tuple(c)
}

call_b {
  flag = pred[] parameter(0)
  init = (pred[]) tuple(flag)
  w = (pred[]) while(init), condition=cond_b, body=body_b
  ROOT out = pred[] get-tuple-element(w), index=0
}

ENTRY e {
  f0 = pred[] parameter(0)
  f1 = pred[] parameter(1)
  n0 = pred[] not(f0)
  n1 = pred[] not(f1)
  c0 = pred[] call(n0), to_apply=call_a
  c1 = pred[] call(n1), to_apply=call_b
  ROOT t = (pred[], pred[]) tuple(c0, c1)
}
)";
  ASSERT_OK_AND_ASSIGN(Calls calls, CreateForModule(kHlo));
  EXPECT_EQ(calls.a.arguments().size(), 1);
  EXPECT_EQ(calls.a.results().size(), 0);
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
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());

  ASSERT_OK_AND_ASSIGN(Calls aliased, CreateForModule(absl::StrReplaceAll(
                                          kHlo, {{"$ARGS_B", "p0, p0"}})));
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
  EXPECT_EQ(calls.a.fingerprint(), calls.b.fingerprint());
}

}  // namespace
}  // namespace xla::cpu
