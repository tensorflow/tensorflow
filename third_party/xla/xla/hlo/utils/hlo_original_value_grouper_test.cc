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

#include "xla/hlo/utils/hlo_original_value_grouper.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"
#include "xla/literal.h"
#include "xla/literal_util.h"

namespace xla {
namespace {

class HloOriginalValueGrouperTest : public HloHardwareIndependentTestBase {};

TEST_F(HloOriginalValueGrouperTest, SharedHloIdBaseCase) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"A"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"B"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

ENTRY %e () -> s32[1,3] {
  %c1 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"A"}}
  ROOT %c2 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"B"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  int64_t reported_callback_id = -1;
  std::vector<std::optional<Literal>> reported_literals;

  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    reported_callback_id = callback_id;
    for (const auto& lit : literals) {
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  auto lit_a = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(1));
  std::vector<HloModule::DebugAttributes> attrs_a;
  attrs_a.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("A"), {},
                                lit_a, attrs_a, 0);
  EXPECT_EQ(call_count, 0);

  auto lit_b = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(2));
  std::vector<HloModule::DebugAttributes> attrs_b;
  attrs_b.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/1});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("B"), {},
                                lit_b, attrs_b, 0);

  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(reported_callback_id, 1);
  ASSERT_EQ(reported_literals.size(), 2);
  EXPECT_EQ(reported_literals[0]->Get<int32_t>({}), 1);
  EXPECT_EQ(reported_literals[1]->Get<int32_t>({}), 2);
}

TEST_F(HloOriginalValueGrouperTest, WildcardDifferentHloIds) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}, origin_recovery_table={
  {"c2"} : {"c2"},
  "
    ENTRY %identity1 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
  {"r"} : {"r"},
  "
    ENTRY %identity2 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
},
debug_attributes={
  {"c2"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"r"}:({log_mode=default,callback_id=2,op_id=0,partitioned=false})
}

body {
  p = s32[1,3]{1,0} parameter(0)
  c2 = s32[1,3]{1,0} constant({{1,2,3}}), origin={{"c2"}}
  ROOT r = s32[1,3]{1,0} add(p, c2), origin={{"r"}}
}

condition {
  p = s32[1,3]{1,0} parameter(0)
  ROOT c = pred[] constant(true)
}

ENTRY %e () -> s32[1,3] {
  p0 = s32[1,3]{1,0} constant({{0,1,2}})
  ROOT %while.1 = s32[1,3]{1,0} while(p0), condition=condition, body=body
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::vector<int64_t> reported_callback_ids;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    reported_callback_ids.push_back(callback_id);
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  auto lit_c2 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(10));
  std::vector<HloModule::DebugAttributes> attrs_c2;
  attrs_c2.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#*/c2"), {}, lit_c2, attrs_c2,
      0);
  EXPECT_TRUE(reported_callback_ids.empty());

  auto lit_r = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(20));
  std::vector<HloModule::DebugAttributes> attrs_r;
  attrs_r.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/2,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#0/r"), {}, lit_r, attrs_r,
      0);

  ASSERT_EQ(reported_callback_ids.size(), 2);
  EXPECT_EQ(reported_callback_ids[0], 1);
  EXPECT_EQ(reported_callback_ids[1], 2);
}

TEST_F(HloOriginalValueGrouperTest, WildcardSameHloId) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}, origin_recovery_table={
  {"c2"} : {"c2"},
  "
    ENTRY %identity1 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
  {"r"} : {"r"},
  "
    ENTRY %identity2 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
},
debug_attributes={
  {"c2"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"r"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

body {
  p = s32[1,3]{1,0} parameter(0)
  c2 = s32[1,3]{1,0} constant({{1,2,3}}), origin={{"c2"}}
  ROOT r = s32[1,3]{1,0} add(p, c2), origin={{"r"}}
}

condition {
  p = s32[1,3]{1,0} parameter(0)
  ROOT c = pred[] constant(true)
}

ENTRY %e () -> s32[1,3] {
  p0 = s32[1,3]{1,0} constant({{0,1,2}})
  ROOT %while.1 = s32[1,3]{1,0} while(p0), condition=condition, body=body
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  std::vector<std::optional<Literal>> reported_literals;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    for (const auto& lit : literals) {
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  auto lit_c2 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(10));
  std::vector<HloModule::DebugAttributes> attrs_c2;
  attrs_c2.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#*/c2"), {}, lit_c2, attrs_c2,
      0);
  EXPECT_EQ(call_count, 0);

  auto lit_r = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(20));
  std::vector<HloModule::DebugAttributes> attrs_r;
  attrs_r.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/1});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#0/r"), {}, lit_r, attrs_r,
      0);

  EXPECT_EQ(call_count, 1);
  ASSERT_EQ(reported_literals.size(), 2);
  EXPECT_EQ(reported_literals[0]->Get<int32_t>({}), 10);
  EXPECT_EQ(reported_literals[1]->Get<int32_t>({}), 20);
}

TEST_F(HloOriginalValueGrouperTest, OutOfOrderSkipMissing) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}, origin_recovery_table={
  {"c2"} : {"c2"},
  "
    ENTRY %identity1 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
  {"r"} : {"r"},
  "
    ENTRY %identity2 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
},
debug_attributes={
  {"c2"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"r"}:({log_mode=default,callback_id=2,op_id=0,partitioned=false})
}

body {
  p = s32[1,3]{1,0} parameter(0)
  c2 = s32[1,3]{1,0} constant({{1,2,3}}), origin={{"c2"}}
  ROOT r = s32[1,3]{1,0} add(p, c2), origin={{"r"}}
}

condition {
  p = s32[1,3]{1,0} parameter(0)
  ROOT c = pred[] constant(true)
}

ENTRY %e () -> s32[1,3] {
  p0 = s32[1,3]{1,0} constant({{0,1,2}})
  ROOT %while.1 = s32[1,3]{1,0} while(p0), condition=condition, body=body
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  std::vector<int64_t> reported_callback_ids;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    reported_callback_ids.push_back(callback_id);
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  // r arrives for iteration 0, c2 is missing
  auto lit_r0 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(20));
  std::vector<HloModule::DebugAttributes> attrs_r;
  attrs_r.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/2,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#0/r"), {}, lit_r0, attrs_r,
      0);

  ASSERT_EQ(reported_callback_ids.size(), 1);
  EXPECT_EQ(reported_callback_ids[0], 2);
  reported_callback_ids.clear();

  // r arrives for iteration 1, c2 is missing
  auto lit_r1 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(21));
  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#1/r"), {}, lit_r1, attrs_r,
      0);

  ASSERT_EQ(reported_callback_ids.size(), 1);
  EXPECT_EQ(reported_callback_ids[0], 2);
  reported_callback_ids.clear();

  // Now c2 arrives
  auto lit_c2 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(10));
  std::vector<HloModule::DebugAttributes> attrs_c2;
  attrs_c2.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#*/c2"), {}, lit_c2, attrs_c2,
      0);
  EXPECT_TRUE(reported_callback_ids.empty());

  // r arrives for iteration 2
  auto lit_r2 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(22));
  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#2/r"), {}, lit_r2, attrs_r,
      0);

  ASSERT_EQ(reported_callback_ids.size(), 2);
  EXPECT_EQ(reported_callback_ids[0], 1);
  EXPECT_EQ(reported_callback_ids[1], 2);
}

TEST_F(HloOriginalValueGrouperTest, OutOfOrderWaitSameHloId) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}, origin_recovery_table={
  {"c2"} : {"c2"},
  "
    ENTRY %identity1 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
  {"r"} : {"r"},
  "
    ENTRY %identity2 (p: s32[1,3]) -> s32[1,3] {
      ROOT %p = s32[1,3]{1,0} parameter(0)
    }
  "
},
debug_attributes={
  {"c2"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"r"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

body {
  p = s32[1,3]{1,0} parameter(0)
  c2 = s32[1,3]{1,0} constant({{1,2,3}}), origin={{"c2"}}
  ROOT r = s32[1,3]{1,0} add(p, c2), origin={{"r"}}
}

condition {
  p = s32[1,3]{1,0} parameter(0)
  ROOT c = pred[] constant(true)
}

ENTRY %e () -> s32[1,3] {
  p0 = s32[1,3]{1,0} constant({{0,1,2}})
  ROOT %while.1 = s32[1,3]{1,0} while(p0), condition=condition, body=body
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  std::vector<std::optional<Literal>> reported_literals;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    for (const auto& lit : literals) {
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  auto lit_r = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(20));
  std::vector<HloModule::DebugAttributes> attrs_r;
  attrs_r.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/1});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#0/r"), {}, lit_r, attrs_r,
      0);
  EXPECT_EQ(call_count, 0);  // waiting for c2

  auto lit_c2 = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(10));
  std::vector<HloModule::DebugAttributes> attrs_c2;
  attrs_c2.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("while.1#*/c2"), {}, lit_c2, attrs_c2,
      0);

  EXPECT_EQ(call_count, 1);
  ASSERT_EQ(reported_literals.size(), 2);
  EXPECT_EQ(reported_literals[0]->Get<int32_t>({}), 10);
  EXPECT_EQ(reported_literals[1]->Get<int32_t>({}), 20);
}

TEST_F(HloOriginalValueGrouperTest, DuplicateArrivals) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"A"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"B"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

ENTRY %e () -> s32[1,3] {
  %c1 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"A"}}
  ROOT %c2 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"B"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  std::vector<std::optional<Literal>> reported_literals;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    for (const auto& lit : literals) {
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  auto lit_a = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(1));
  std::vector<HloModule::DebugAttributes> attrs_a;
  attrs_a.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("A"), {},
                                lit_a, attrs_a, 0);
  // Duplicate arrival
  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("A"), {},
                                lit_a, attrs_a, 0);
  EXPECT_EQ(call_count, 0);

  auto lit_b = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(2));
  std::vector<HloModule::DebugAttributes> attrs_b;
  attrs_b.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/1});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("B"), {},
                                lit_b, attrs_b, 0);

  EXPECT_EQ(call_count, 1);
  ASSERT_EQ(reported_literals.size(), 2);
  EXPECT_EQ(reported_literals[0]->Get<int32_t>({}), 1);
  EXPECT_EQ(reported_literals[1]->Get<int32_t>({}), 2);
}

TEST_F(HloOriginalValueGrouperTest, NullLiterals) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"A"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"B"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

ENTRY %e () -> s32[1,3] {
  %c1 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"A"}}
  ROOT %c2 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"B"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  std::vector<bool> received_null;
  std::vector<std::optional<Literal>> reported_literals;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    for (const auto& lit : literals) {
      received_null.push_back(lit == nullptr);
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  std::vector<HloModule::DebugAttributes> attrs_a;
  attrs_a.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("A"), {},
                                nullptr, attrs_a, 0);
  EXPECT_EQ(call_count, 0);

  auto lit_b = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(2));
  std::vector<HloModule::DebugAttributes> attrs_b;
  attrs_b.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/1});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("B"), {},
                                lit_b, attrs_b, 0);

  EXPECT_EQ(call_count, 1);
  ASSERT_EQ(received_null.size(), 2);
  EXPECT_TRUE(received_null[0]);
  EXPECT_FALSE(received_null[1]);
  EXPECT_EQ(reported_literals[1]->Get<int32_t>({}), 2);
}

TEST_F(HloOriginalValueGrouperTest, HandlesUnaddressableDevicePartialAbort) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[2]{0}},
origin_recovery_table={
  {"A"} : {"__ovp_1"},
  "
    ENTRY %recovery_1 (p: s32[2]) -> s32[2] {
      ROOT %p = s32[2]{0} parameter(0), sharding={replicated}
    }
  "
  {"B"} : {"__ovp_2"},
  "
    ENTRY %recovery_2 (p: s32[2]) -> s32[2] {
      ROOT %p = s32[2]{0} parameter(0), sharding={devices=[2]<=[2]}
    }
  "
},
debug_attributes={
  {"A"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"B"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

ENTRY %e () -> s32[2] {
  %c1 = s32[2]{0} constant({0, 1}), origin={{"__ovp_1"}}, sharding={replicated}
  ROOT %c2 = s32[2]{0} constant({0, 1}), origin={{"__ovp_2"}}, sharding={devices=[2]<=[2]}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  int64_t reported_callback_id = -1;
  std::vector<std::optional<Literal>> reported_literals;

  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    reported_callback_id = callback_id;
    for (const auto& lit : literals) {
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  auto logical_device_is_addressable = [&](int64_t logical_device_id) -> bool {
    return logical_device_id == 0;
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true,
                                  logical_device_is_addressable);

  auto lit_a = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(42));
  std::vector<HloModule::DebugAttributes> attrs_a;
  attrs_a.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("A"), {},
                                lit_a, attrs_a, 0);

  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(reported_callback_id, 1);
  ASSERT_EQ(reported_literals.size(), 2);
  EXPECT_NE(reported_literals[0], std::nullopt);
  EXPECT_EQ(reported_literals[0]->Get<int32_t>({}), 42);
  EXPECT_EQ(reported_literals[1], std::nullopt);
}

TEST_F(HloOriginalValueGrouperTest, NonDebugLogModeIgnored) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}

ENTRY %e () -> s32[1,3] {
  ROOT %c1 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"A"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  OriginalArray oa = {"A", {}};
  module->AddDebugAttributes(
      oa, {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kNone,
           /*callback_id=*/1,
           /*partitioned=*/false,
           /*op_id=*/0});

  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/true);

  auto lit_a = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(1));
  std::vector<HloModule::DebugAttributes> attrs_a;
  attrs_a.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kNone,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(AbsoluteScopedTensorKey::FromString("A"), {},
                                lit_a, attrs_a, 0);
  EXPECT_EQ(call_count, 0);
}

TEST_F(HloOriginalValueGrouperTest, RespectsRecoverabilityCheck) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"Recoverable"}:({log_mode=default,callback_id=1,op_id=0,partitioned=false}),
  {"Unrecoverable"}:({log_mode=default,callback_id=1,op_id=1,partitioned=false})
}

ENTRY %e () -> s32[1,3] {
  ROOT %c1 = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"Recoverable"}}
}
)hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto analysis,
                       HloOriginalValueAnalysis::Create(module.get()));
  auto analysis_shared =
      std::shared_ptr<const HloOriginalValueAnalysis>(std::move(analysis));

  int call_count = 0;
  std::vector<std::optional<Literal>> reported_literals;
  auto callback = [&](int64_t callback_id, int64_t replica_id,
                      int64_t partition_id,
                      absl::Span<std::shared_ptr<Literal> const> literals) {
    call_count++;
    for (const auto& lit : literals) {
      reported_literals.push_back(
          lit != nullptr ? std::make_optional(lit->Clone()) : std::nullopt);
    }
  };

  HloOriginalValueGrouper grouper(module.get(), analysis_shared, callback,
                                  /*skip_recoverability_check=*/false);

  auto lit_rec = std::make_shared<Literal>(LiteralUtil::CreateR0<int32_t>(99));
  std::vector<HloModule::DebugAttributes> attrs_rec;
  attrs_rec.push_back(
      {/*log_mode=*/HloModule::DebugAttributes::DebugLogMode::kDefault,
       /*callback_id=*/1,
       /*partitioned=*/false,
       /*op_id=*/0});

  grouper.OnOriginalTensorReady(
      AbsoluteScopedTensorKey::FromString("Recoverable"), {}, lit_rec,
      attrs_rec, 0);

  EXPECT_EQ(call_count, 1);
  ASSERT_EQ(reported_literals.size(), 2);
  EXPECT_NE(reported_literals[0], std::nullopt);
  EXPECT_EQ(reported_literals[0]->Get<int32_t>({}), 99);
  EXPECT_EQ(reported_literals[1], std::nullopt);
}

}  // namespace
}  // namespace xla
