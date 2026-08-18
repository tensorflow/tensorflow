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

#include "xla/hlo/tools/comparison/original_tensor_summary_propagator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::tsl::testing::IsOk;

struct PropagatedSummary {
  AbsoluteScopedTensorKey original_tensor_key;
  std::shared_ptr<const tensor_transformation::TensorTransformation>
      pending_transformation;
  OriginalTensorSummary original_tensor_summary;
};

// A matcher for PropagatedSummary.
MATCHER_P(TensorKeyInstructionName, name, "") {
  return arg.original_tensor_key.tensor_key.instruction_name == name;
}

class OriginalTensorSummaryPropagatorTest
    : public ::xla::HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    propagated_summaries_.clear();
    recovered_tensors_.clear();
  }

  OriginalTensorSummaryCallback on_propagated_tensor_summary_ =
      [&](const AbsoluteScopedTensorKey& key,
          std::shared_ptr<const tensor_transformation::TensorTransformation>
              pending,
          const OriginalTensorSummary& summary) {
        if (!summary.summaries.empty()) {
          propagated_summaries_.push_back({key, std::move(pending), summary});
        }
        return absl::OkStatus();
      };

  IsOriginalTensorAlreadyRecoveredCallback
      is_original_tensor_already_recovered_ =
          [&](const AbsoluteScopedTensorKey& key) {
            return recovered_tensors_.contains(key);
          };

  std::vector<PropagatedSummary> propagated_summaries_;
  absl::flat_hash_set<AbsoluteScopedTensorKey> recovered_tensors_;
};

OriginalTensorSummary CreateSimpleSummary() {
  OriginalTensorSummary root_summary;
  root_summary.dimensions = {2, 2};
  ::xla::comparison::FloatSummary float_summary;
  ::xla::comparison::FloatBlockSummary block_summary;
  block_summary.min = 1.0f;
  block_summary.max = 4.0f;
  block_summary.mean = 2.5f;
  block_summary.stddev = 1.118f;
  block_summary.count = 4.0f;
  float_summary.block_summaries.push_back(block_summary);
  root_summary.summaries.push_back(float_summary);
  return root_summary;
}

OriginalTensorSummary CreateSummaryForShape(absl::Span<const int64_t> dims) {
  OriginalTensorSummary summary;
  summary.dimensions.assign(dims.begin(), dims.end());
  ::xla::comparison::FloatSummary float_summary;
  int64_t count = 1;
  for (int64_t dim : dims) {
    count *= dim;
  }
  ::xla::comparison::FloatBlockSummary block_summary;
  block_summary.min = 1.0f;
  block_summary.max = 4.0f;
  block_summary.mean = 2.5f;
  block_summary.stddev = 1.118f;
  block_summary.count = static_cast<float>(count);
  float_summary.block_summaries.push_back(block_summary);
  summary.summaries.push_back(float_summary);
  return summary;
}

TEST_F(OriginalTensorSummaryPropagatorTest, ConstantPropagation) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule constant_propagation_test, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

ENTRY %main (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2]{1,0} parameter(0)
  %c = f32[2,2]{1,0} constant({{1,2},{3,4}})
  ROOT %add = f32[2,2]{1,0} add(%p, %c)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  ASSERT_THAT(propagated_summaries_,
              ElementsAre(TensorKeyInstructionName("c")));
  const auto& summary = propagated_summaries_[0];
  EXPECT_TRUE(summary.original_tensor_key.scope_instructions.empty());
  EXPECT_EQ(summary.pending_transformation, nullptr);
  EXPECT_THAT(summary.original_tensor_summary.dimensions, ElementsAre(2, 2));

  const auto& block_summary =
      summary.original_tensor_summary.summaries[0].block_summaries[0];
  EXPECT_EQ(block_summary.min, 1.0f);
  EXPECT_EQ(block_summary.max, 4.0f);
  EXPECT_NEAR(block_summary.mean, 2.5f, 1e-6);
  EXPECT_NEAR(block_summary.stddev, 1.118034f, 1e-6);
  EXPECT_EQ(block_summary.count, 4);
}

TEST_F(OriginalTensorSummaryPropagatorTest, IotaPropagation) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule iota_propagation_test, entry_computation_layout={(f32[2,3]{1,0})->f32[2,3]{1,0}}

ENTRY %main (p: f32[2,3]) -> f32[2,3] {
  %p = f32[2,3]{1,0} parameter(0)
  %iota = f32[2,3]{1,0} iota(), iota_dimension=1
  ROOT %add = f32[2,3]{1,0} add(%p, %iota)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  ASSERT_THAT(propagated_summaries_,
              ElementsAre(TensorKeyInstructionName("iota")));
  const auto& summary = propagated_summaries_[0];
  EXPECT_TRUE(summary.original_tensor_key.scope_instructions.empty());
  EXPECT_EQ(summary.pending_transformation, nullptr);
  EXPECT_THAT(summary.original_tensor_summary.dimensions, ElementsAre(2, 3));

  // values are {{0,1,2},{0,1,2}}
  const auto& block_summary =
      summary.original_tensor_summary.summaries[0].block_summaries[0];
  EXPECT_EQ(block_summary.min, 0.0f);
  EXPECT_EQ(block_summary.max, 2.0f);
  EXPECT_NEAR(block_summary.mean, 1.0f, 1e-6);
  EXPECT_NEAR(block_summary.stddev, 0.816496f, 1e-6);
  EXPECT_EQ(block_summary.count, 6);
}

TEST_F(OriginalTensorSummaryPropagatorTest, ParameterPropagation) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule parameter_propagation_test, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

%called (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2]{1,0} parameter(0)
  ROOT %neg = f32[2,2]{1,0} negate(%p)
}

ENTRY %main (p_main: f32[2,2]) -> f32[2,2] {
  %p_main = f32[2,2]{1,0} parameter(0)
  %c = f32[2,2]{1,0} constant({{1,2},{3,4}})
  ROOT %call = f32[2,2]{1,0} call(%c), to_apply=%called
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());
  propagated_summaries_.clear();

  AbsoluteScopedTensorKey key_in_called_computation =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("neg"),
                                      {ScopeInstruction::Create("call")});

  ASSERT_THAT(propagator.Process(key_in_called_computation, nullptr,
                                 CreateSimpleSummary()),
              IsOk());

  auto it = std::find_if(
      propagated_summaries_.begin(), propagated_summaries_.end(),
      [](const PropagatedSummary& s) {
        return s.original_tensor_key.tensor_key.instruction_name == "p";
      });
  ASSERT_NE(it, propagated_summaries_.end());
  EXPECT_THAT(it->original_tensor_key.scope_instructions,
              ElementsAre(ScopeInstruction::Create("call")));
  EXPECT_EQ(it->pending_transformation, nullptr);
  const auto& block_summary =
      it->original_tensor_summary.summaries[0].block_summaries[0];
  EXPECT_EQ(block_summary.min, 1.0f);
  EXPECT_EQ(block_summary.max, 4.0f);
}

TEST_F(OriginalTensorSummaryPropagatorTest, NoParameterPropagationForMap) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule map_parameter_propagation_test, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

%map_computation (a: f32[], b: f32[]) -> f32[] {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = f32[] add(%p0, %p1)
}

ENTRY %main (p_main: f32[2,2]) -> f32[2,2] {
  %p_main = f32[2,2]{1,0} parameter(0)
  %c = f32[2,2]{1,0} constant({{1,2},{3,4}})
  ROOT %map = f32[2,2]{1,0} map(%p_main, %c), to_apply=%map_computation
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());
  propagated_summaries_.clear();

  AbsoluteScopedTensorKey key_in_map_computation =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add"),
                                      {ScopeInstruction::Create("map")});

  ASSERT_THAT(propagator.Process(key_in_map_computation, nullptr,
                                 CreateSummaryForShape({})),
              IsOk());

  auto it = std::find_if(
      propagated_summaries_.begin(), propagated_summaries_.end(),
      [](const PropagatedSummary& s) {
        return s.original_tensor_key.tensor_key.instruction_name == "p0" ||
               s.original_tensor_key.tensor_key.instruction_name == "p1";
      });
  EXPECT_EQ(it, propagated_summaries_.end());
}

TEST_F(OriginalTensorSummaryPropagatorTest, ForwardPropagation) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule forward_prop, entry_computation_layout={(f32[2,2]{1,0})->f32[2,4]{1,0}}

ENTRY %main (p: f32[2,2]) -> f32[2,4] {
  %p = f32[2,2]{1,0} parameter(0)
  %copy = f32[2,2]{1,0} copy(%p)
  %reshape = f32[4]{0} reshape(%copy)
  %bcast = f32[2,4]{1,0} broadcast(%reshape), dimensions={1}
  %transpose = f32[4,2]{1,0} transpose(%bcast), dimensions={1,0}
  %tuple = (f32[2,2]{1,0}, f32[4,2]{1,0}) tuple(%p, %transpose)
  %gte = f32[4,2]{1,0} get-tuple-element(%tuple), index=1
  ROOT %root_op = f32[2,4]{1,0} copy(%bcast)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  AbsoluteScopedTensorKey p_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p"));
  ASSERT_THAT(propagator.Process(p_key, nullptr, CreateSimpleSummary()),
              IsOk());

  // There are 9 propagated summaries in total: p, copy, reshape, bcast,
  // transpose, tuple member 0 (from p), tuple member 1 (from transpose), gte
  // (from tuple member 1) and root_op.
  EXPECT_THAT(propagated_summaries_, SizeIs(9));

  // Check transformations.
  auto find_summary = [&](absl::string_view name,
                          ShapeIndexView shape_index = {}) {
    auto it = std::find_if(
        propagated_summaries_.begin(), propagated_summaries_.end(),
        [&](const PropagatedSummary& s) {
          return s.original_tensor_key.tensor_key.instruction_name == name &&
                 s.original_tensor_key.tensor_key.shape_index ==
                     ShapeIndex(shape_index);
        });
    return it;
  };
  EXPECT_EQ(find_summary("p", {})->pending_transformation, nullptr);
  EXPECT_EQ(find_summary("copy")->pending_transformation, nullptr);
  EXPECT_EQ(tensor_transformation::ToString(
                find_summary("reshape")->pending_transformation.get()),
            "Reshape{dimensions=[4], continuation=nullptr}");
  EXPECT_EQ(tensor_transformation::ToString(
                find_summary("bcast")->pending_transformation.get()),
            "Reshape{dimensions=[4], continuation=Broadcast{dimensions=[2, 4], "
            "broadcast_dimensions=[1], continuation=nullptr}}");
  EXPECT_EQ(tensor_transformation::ToString(
                find_summary("transpose")->pending_transformation.get()),
            "Reshape{dimensions=[4], continuation=Broadcast{dimensions=[2, 4], "
            "broadcast_dimensions=[1], continuation=Broadcast{dimensions=[4, "
            "2], broadcast_dimensions=[1, 0], continuation=nullptr}}}");

  auto tuple_it = std::find_if(
      propagated_summaries_.begin(), propagated_summaries_.end(),
      [&](const PropagatedSummary& s) {
        return s.original_tensor_key.tensor_key.instruction_name == "tuple" &&
               s.original_tensor_key.tensor_key.shape_index == ShapeIndex{1};
      });
  EXPECT_NE(tuple_it, propagated_summaries_.end());
  EXPECT_EQ(
      tensor_transformation::ToString(tuple_it->pending_transformation.get()),
      tensor_transformation::ToString(
          find_summary("transpose")->pending_transformation.get()));

  EXPECT_EQ(tensor_transformation::ToString(
                find_summary("gte")->pending_transformation.get()),
            tensor_transformation::ToString(
                find_summary("transpose")->pending_transformation.get()));
}

TEST_F(OriginalTensorSummaryPropagatorTest, BackwardPropagation) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule backward_prop, entry_computation_layout={(f32[4]{0})->f32[2,4]{1,0}}

ENTRY %main (p: f32[4]) -> f32[2,4] {
  %p = f32[4]{0} parameter(0)
  %bcast = f32[2,4]{1,0} broadcast(%p), dimensions={1}
  %transpose = f32[4,2]{1,0} transpose(%bcast), dimensions={1,0}
  %reshape = f32[8]{0} reshape(%transpose)
  %bitcast = f32[2,4]{1,0} bitcast(%reshape)
  ROOT %copy = f32[2,4]{1,0} copy(%bitcast)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  AbsoluteScopedTensorKey copy_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("copy"));
  ASSERT_THAT(
      propagator.Process(copy_key, nullptr, CreateSummaryForShape({2, 4})),
      IsOk());
  // Process calls on_propagated_tensor_summary_ for the root, then propagates.
  // Propagates backward: copy -> bitcast -> reshape -> transpose -> bcast -> p
  EXPECT_THAT(propagated_summaries_, SizeIs(6));

  auto find_summary = [&](absl::string_view name) {
    auto it = std::find_if(
        propagated_summaries_.begin(), propagated_summaries_.end(),
        [&](const PropagatedSummary& s) {
          return s.original_tensor_key.tensor_key.instruction_name == name;
        });
    return it;
  };

  EXPECT_EQ(find_summary("copy")->pending_transformation, nullptr);
  EXPECT_EQ(find_summary("bitcast")->pending_transformation, nullptr);
  EXPECT_EQ(tensor_transformation::ToString(
                find_summary("reshape")->pending_transformation.get()),
            "Reshape{dimensions=[8], continuation=nullptr}");
  EXPECT_EQ(tensor_transformation::ToString(
                find_summary("transpose")->pending_transformation.get()),
            "Reshape{dimensions=[8], continuation=Reshape{dimensions=[4, 2], "
            "continuation=nullptr}}");
  EXPECT_EQ(
      tensor_transformation::ToString(
          find_summary("bcast")->pending_transformation.get()),
      "Reshape{dimensions=[8], continuation=Reshape{dimensions=[4, 2], "
      "continuation=Broadcast{dimensions=[2, 4], broadcast_dimensions=[1, 0], "
      "continuation=nullptr}}}");
  EXPECT_EQ(
      tensor_transformation::ToString(
          find_summary("p")->pending_transformation.get()),
      "Reshape{dimensions=[8], continuation=Reshape{dimensions=[4, 2], "
      "continuation=Broadcast{dimensions=[2, 4], broadcast_dimensions=[1, 0], "
      "continuation=Broadcast{dimensions=[4], broadcast_dimensions=[-1, 0], "
      "continuation=nullptr}}}}");
}

TEST_F(OriginalTensorSummaryPropagatorTest,
       DoesNotPropagateToAlreadyPropagated) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule no_repropagate, entry_computation_layout={(f32[2,2]{1,0})->(f32[2,2]{1,0}, f32[2,2]{1,0})}
ENTRY %main (p: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p = f32[2,2]{1,0} parameter(0)
  %copy = f32[2,2]{1,0} copy(%p)
  ROOT %tuple = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(%copy, %copy)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  AbsoluteScopedTensorKey tuple0_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("tuple", ShapeIndex{0}));
  ASSERT_THAT(propagator.Process(tuple0_key, nullptr, CreateSimpleSummary()),
              IsOk());
  // tuple{0} -> copy -> p
  EXPECT_THAT(propagated_summaries_,
              SizeIs(3));  // tuple{0}, copy, p
  propagated_summaries_.clear();

  AbsoluteScopedTensorKey tuple1_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("tuple", ShapeIndex{1}));
  ASSERT_THAT(propagator.Process(tuple1_key, nullptr, CreateSimpleSummary()),
              IsOk());
  // only tuple{1} is new. copy and p are already propagated.
  EXPECT_THAT(propagated_summaries_, SizeIs(1));
  EXPECT_EQ(
      propagated_summaries_[0].original_tensor_key.tensor_key.instruction_name,
      "tuple");
  EXPECT_EQ(propagated_summaries_[0].original_tensor_key.tensor_key.shape_index,
            ShapeIndex{1});
}

TEST_F(OriginalTensorSummaryPropagatorTest,
       DoesNotPropagateToAlreadyRecovered) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule no_repropagate_recovered, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}
ENTRY %main (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2]{1,0} parameter(0)
  ROOT %copy = f32[2,2]{1,0} copy(%p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  AbsoluteScopedTensorKey p_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("p"));
  recovered_tensors_.insert(p_key);

  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  AbsoluteScopedTensorKey copy_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("copy"));
  ASSERT_THAT(propagator.Process(copy_key, nullptr, CreateSimpleSummary()),
              IsOk());
  // Propagates from copy, but stops at p.
  EXPECT_THAT(propagated_summaries_, SizeIs(1));
  EXPECT_EQ(
      propagated_summaries_[0].original_tensor_key.tensor_key.instruction_name,
      "copy");
}

TEST_F(OriginalTensorSummaryPropagatorTest, NestedComputationDrillDown) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule nested_computation, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

%inner (p.inner: f32[2,2]) -> f32[2,2] {
  %p.inner = f32[2,2]{1,0} parameter(0)
  %c.inner = f32[2,2]{1,0} constant({{5,6},{7,8}})
  ROOT %add.inner = f32[2,2]{1,0} add(%p.inner, %c.inner)
}

%outer (p.outer: f32[2,2]) -> f32[2,2] {
  %p.outer = f32[2,2]{1,0} parameter(0)
  %c.outer = f32[2,2]{1,0} constant({{1,2},{3,4}})
  ROOT %call.inner = f32[2,2]{1,0} call(%c.outer), to_apply=%inner
}

ENTRY %main (p.main: f32[2,2]) -> f32[2,2] {
  %p.main = f32[2,2]{1,0} parameter(0)
  ROOT %call.outer = f32[2,2]{1,0} call(%p.main), to_apply=%outer
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  AbsoluteScopedTensorKey key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("add.inner"), {ScopeInstruction::Create("call.outer"),
                                       ScopeInstruction::Create("call.inner")});

  ASSERT_THAT(propagator.Process(key, nullptr, CreateSimpleSummary()), IsOk());

  // Expected propagated summaries:
  // 1. In `outer` computation: `c.outer`
  // 2. In `inner` computation: `p.inner` (from `c.outer`), `c.inner`
  // 3. The root tensor `add.inner` passed to process.
  // ... and any forward/backward propagation from `add.inner`.
  // Here we just check the constants and parameter were propagated.

  auto find_summary = [&](absl::string_view name,
                          const std::vector<ScopeInstruction>& scope) {
    auto it = std::find_if(
        propagated_summaries_.begin(), propagated_summaries_.end(),
        [&](const PropagatedSummary& s) {
          return s.original_tensor_key.tensor_key.instruction_name == name &&
                 s.original_tensor_key.scope_instructions == scope;
        });
    return it;
  };

  auto c_outer_it =
      find_summary("c.outer", {ScopeInstruction::Create("call.outer")});
  ASSERT_NE(c_outer_it, propagated_summaries_.end());
  EXPECT_EQ(
      c_outer_it->original_tensor_summary.summaries[0].block_summaries[0].max,
      4.0f);

  auto p_inner_it =
      find_summary("p.inner", {ScopeInstruction::Create("call.outer"),
                               ScopeInstruction::Create("call.inner")});
  ASSERT_NE(p_inner_it, propagated_summaries_.end());
  EXPECT_EQ(
      p_inner_it->original_tensor_summary.summaries[0].block_summaries[0].max,
      4.0f);

  auto c_inner_it =
      find_summary("c.inner", {ScopeInstruction::Create("call.outer"),
                               ScopeInstruction::Create("call.inner")});
  ASSERT_NE(c_inner_it, propagated_summaries_.end());
  EXPECT_EQ(
      c_inner_it->original_tensor_summary.summaries[0].block_summaries[0].max,
      8.0f);
}

TEST_F(OriginalTensorSummaryPropagatorTest, ConsecutiveCalls) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule consecutive_calls, entry_computation_layout={(f32[2,2]{1,0})->(f32[2,2]{1,0}, f32[2,2]{1,0})}

%computation1 (p1: f32[2,2]) -> f32[2,2] {
  %p1 = f32[2,2]{1,0} parameter(0)
  %c1 = f32[2,2]{1,0} constant({{1,1},{1,1}})
  ROOT %add1 = f32[2,2]{1,0} add(%p1, %c1)
}

%computation2 (p2: f32[2,2]) -> f32[2,2] {
  %p2 = f32[2,2]{1,0} parameter(0)
  %c2 = f32[2,2]{1,0} constant({{2,2},{2,2}})
  ROOT %add2 = f32[2,2]{1,0} add(%p2, %c2)
}

ENTRY %main (p_main: f32[2,2]) -> (f32[2,2], f32[2,2]) {
  %p_main = f32[2,2]{1,0} parameter(0)
  %c_main_1 = f32[2,2]{1,0} constant({{10,10},{10,10}})
  %call1 = f32[2,2]{1,0} call(%c_main_1), to_apply=%computation1
  %c_main_2 = f32[2,2]{1,0} constant({{20,20},{20,20}})
  %call2 = f32[2,2]{1,0} call(%c_main_2), to_apply=%computation2
  ROOT %tuple = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(%call1, %call2)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());

  // Initialization should propagate constants from the main computation.
  // There are forward propagations as well.
  propagated_summaries_.clear();

  // Process a tensor in the first called computation.
  AbsoluteScopedTensorKey key1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("add1"), {ScopeInstruction::Create("call1")});
  ASSERT_THAT(propagator.Process(key1, nullptr, CreateSimpleSummary()), IsOk());

  auto find_summary_with_scope =
      [&](absl::string_view name, const std::vector<ScopeInstruction>& scope) {
        return std::find_if(
            propagated_summaries_.begin(), propagated_summaries_.end(),
            [&](const PropagatedSummary& s) {
              return s.original_tensor_key.tensor_key.instruction_name ==
                         name &&
                     s.original_tensor_key.scope_instructions == scope;
            });
      };

  // Check that parameters and constants in computation1 are propagated.
  auto p1_it =
      find_summary_with_scope("p1", {ScopeInstruction::Create("call1")});
  ASSERT_NE(p1_it, propagated_summaries_.end());
  EXPECT_EQ(p1_it->original_tensor_summary.summaries[0].block_summaries[0].min,
            10.0f);

  auto c1_it =
      find_summary_with_scope("c1", {ScopeInstruction::Create("call1")});
  ASSERT_NE(c1_it, propagated_summaries_.end());
  EXPECT_EQ(c1_it->original_tensor_summary.summaries[0].block_summaries[0].min,
            1.0f);
  propagated_summaries_.clear();

  // Process a tensor in the second called computation.
  AbsoluteScopedTensorKey key2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("add2"), {ScopeInstruction::Create("call2")});
  ASSERT_THAT(propagator.Process(key2, nullptr, CreateSimpleSummary()), IsOk());

  // Check that parameters and constants in computation2 are propagated.
  auto p2_it =
      find_summary_with_scope("p2", {ScopeInstruction::Create("call2")});
  ASSERT_NE(p2_it, propagated_summaries_.end());
  EXPECT_EQ(p2_it->original_tensor_summary.summaries[0].block_summaries[0].min,
            20.0f);

  auto c2_it =
      find_summary_with_scope("c2", {ScopeInstruction::Create("call2")});
  ASSERT_NE(c2_it, propagated_summaries_.end());
  EXPECT_EQ(c2_it->original_tensor_summary.summaries[0].block_summaries[0].min,
            2.0f);

  // Check that no summaries from computation1 are propagated in this step.
  EXPECT_EQ(find_summary_with_scope("p1", {ScopeInstruction::Create("call1")}),
            propagated_summaries_.end());
  EXPECT_EQ(find_summary_with_scope("c1", {ScopeInstruction::Create("call1")}),
            propagated_summaries_.end());
}

TEST_F(OriginalTensorSummaryPropagatorTest, SkipsGuessedScopeInstruction) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule skips_guessed_scope, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}
ENTRY %main (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2]{1,0} parameter(0)
  ROOT %copy = f32[2,2]{1,0} copy(%p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());
  propagated_summaries_.clear();

  AbsoluteScopedTensorKey guessed_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("copy"), {ScopeInstruction::Create("call?")});

  ASSERT_THAT(propagator.Process(guessed_key, nullptr, CreateSimpleSummary()),
              IsOk());
  EXPECT_TRUE(propagated_summaries_.empty());
}

TEST_F(OriginalTensorSummaryPropagatorTest, WildcardPropagation) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule wildcard_propagation, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

%loop_body (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2]{1,0} parameter(0)
  ROOT %copy = f32[2,2]{1,0} copy(%p)
}

%loop_cond (p: f32[2,2]) -> pred[] {
  %p = f32[2,2]{1,0} parameter(0)
  ROOT %pred_ = pred[] constant(true)
}

ENTRY %main (p: f32[2,2]) -> f32[2,2] {
  %p = f32[2,2]{1,0} parameter(0)
  ROOT %loop = f32[2,2]{1,0} while(%p), condition=%loop_cond, body=%loop_body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));

  OriginalTensorSummaryPropagator propagator(
      module.get(), std::move(on_propagated_tensor_summary_),
      std::move(is_original_tensor_already_recovered_));
  ASSERT_THAT(propagator.Initialize(), IsOk());
  propagated_summaries_.clear();

  // Provide summary for a key inside loop body but with a wildcard iteration.
  AbsoluteScopedTensorKey wildcard_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("copy"), {ScopeInstruction::FromString("loop#*")});
  ASSERT_THAT(propagator.Process(wildcard_key, nullptr, CreateSimpleSummary()),
              IsOk());
  // It should be delayed, so no propagations yet.
  EXPECT_TRUE(propagated_summaries_.empty());

  // Trigger entering the loop iteration #1 by processing another key.
  AbsoluteScopedTensorKey concrete_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("p"), {ScopeInstruction::FromString("loop#1")});
  ASSERT_THAT(propagator.Process(concrete_key, nullptr, CreateSimpleSummary()),
              IsOk());

  // Both the concrete key summary and the instantiated wildcard summary should
  // now be propagated.
  auto find_summary = [&](absl::string_view name,
                          const std::vector<ScopeInstruction>& scope) {
    return std::find_if(
        propagated_summaries_.begin(), propagated_summaries_.end(),
        [&](const PropagatedSummary& s) {
          return s.original_tensor_key.tensor_key.instruction_name == name &&
                 s.original_tensor_key.scope_instructions == scope;
        });
  };

  EXPECT_NE(find_summary("copy", {ScopeInstruction::FromString("loop#1")}),
            propagated_summaries_.end());
  EXPECT_NE(find_summary("p", {ScopeInstruction::FromString("loop#1")}),
            propagated_summaries_.end());
}

}  // namespace
}  // namespace xla::numerics::comparison
