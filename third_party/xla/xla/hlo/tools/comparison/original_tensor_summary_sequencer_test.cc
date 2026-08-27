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

#include "xla/hlo/tools/comparison/original_tensor_summary_sequencer.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::ElementsAre;
using ::tsl::proto_testing::EqualsProto;

RecoveredTensorSummaryProto CreateSummaryProto(
    const AbsoluteScopedTensorKey& key) {
  RecoveredTensorSummaryProto proto;
  *proto.mutable_tensor_key() = key.ToProto();
  return proto;
}

void WriteSummaries(absl::string_view path,
                    absl::Span<const RecoveredTensorSummaryProto> summaries) {
  riegeli::RecordWriter writer(
      riegeli::FdWriter(path),
      riegeli::RecordWriterBase::Options().set_transpose(true));
  for (const auto& summary : summaries) {
    writer.WriteRecord(summary);
  }
  ASSERT_TRUE(writer.Close()) << writer.status();
}

std::vector<RecoveredTensorSummaryProto> ReadSummaries(absl::string_view path) {
  riegeli::RecordReader reader{riegeli::FdReader(path)};
  RecoveredTensorSummaryProto summary;
  std::vector<RecoveredTensorSummaryProto> summaries;
  while (reader.ReadRecord(summary)) {
    summaries.push_back(summary);
  }
  CHECK(reader.Close()) << reader.status();
  return summaries;
}

TEST(OriginalTensorSummarySequencerTest, SortsByTopoRank) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks = {
      {"instr1", 0}, {"instr2", 1}, {"instr3", 2}};
  OriginalTensorSummarySequencer sequencer(std::move(topo_ranks));
  auto k1 = AbsoluteScopedTensorKey::Create(TensorKey::Create("instr1"));
  auto k2 = AbsoluteScopedTensorKey::Create(TensorKey::Create("instr2"));
  auto k3 = AbsoluteScopedTensorKey::Create(TensorKey::Create("instr3"));
  std::vector<RecoveredTensorSummaryProto> summaries = {
      CreateSummaryProto(k2), CreateSummaryProto(k3), CreateSummaryProto(k1)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK_AND_ASSIGN(auto callback,
                       sequencer.Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k1)),
                          EqualsProto(CreateSummaryProto(k2)),
                          EqualsProto(CreateSummaryProto(k3))));
  EXPECT_TRUE((*callback)(k1));
  EXPECT_TRUE((*callback)(k2));
  EXPECT_TRUE((*callback)(k3));
  EXPECT_FALSE(
      (*callback)(AbsoluteScopedTensorKey::Create(TensorKey::Create("other"))));
}

TEST(OriginalTensorSummarySequencerTest, SortsByShapeIndex) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks = {{"instr1", 0}};
  OriginalTensorSummarySequencer sequencer(std::move(topo_ranks));
  auto k1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr1", ShapeIndex{0}));
  auto k2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr1", ShapeIndex{1}));
  std::vector<RecoveredTensorSummaryProto> summaries = {CreateSummaryProto(k2),
                                                        CreateSummaryProto(k1)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK(sequencer.Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k1)),
                          EqualsProto(CreateSummaryProto(k2))));
}

TEST(OriginalTensorSummarySequencerTest, SortsByScopeInstructionRank) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks = {
      {"scope1", 0}, {"scope2", 1}, {"instr1", 2}};
  OriginalTensorSummarySequencer sequencer(std::move(topo_ranks));
  auto k1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr1"), {ScopeInstruction::Create("scope1")});
  auto k2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr1"), {ScopeInstruction::Create("scope2")});
  std::vector<RecoveredTensorSummaryProto> summaries = {CreateSummaryProto(k2),
                                                        CreateSummaryProto(k1)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK(sequencer.Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k1)),
                          EqualsProto(CreateSummaryProto(k2))));
}

TEST(OriginalTensorSummarySequencerTest, SortsByIterationIndex) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks = {{"scope1", 0},
                                                          {"instr1", 1}};
  OriginalTensorSummarySequencer sequencer(std::move(topo_ranks));
  auto k1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr1"), {ScopeInstruction::Create("scope1", 1)});
  auto k2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr1"), {ScopeInstruction::Create("scope1", 2)});
  std::vector<RecoveredTensorSummaryProto> summaries = {CreateSummaryProto(k2),
                                                        CreateSummaryProto(k1)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK(sequencer.Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k1)),
                          EqualsProto(CreateSummaryProto(k2))));
}

TEST(OriginalTensorSummarySequencerTest, SortsWithCallLikeInstructions) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks = {{"call", 0},
                                                          {"instr_in_call", 1}};
  OriginalTensorSummarySequencer sequencer(std::move(topo_ranks));
  auto k_in_call = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("instr_in_call"), {ScopeInstruction::Create("call")});
  auto k_call = AbsoluteScopedTensorKey::Create(TensorKey::Create("call"));
  std::vector<RecoveredTensorSummaryProto> summaries = {
      CreateSummaryProto(k_call), CreateSummaryProto(k_in_call)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK(sequencer.Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k_in_call)),
                          EqualsProto(CreateSummaryProto(k_call))));
}

TEST(OriginalTensorSummarySequencerTest, HandlesUnknownInstruction) {
  absl::flat_hash_map<std::string, int64_t> topo_ranks = {{"instr1", 0},
                                                          {"instr2", 1}};
  OriginalTensorSummarySequencer sequencer(std::move(topo_ranks));
  auto k1 = AbsoluteScopedTensorKey::Create(TensorKey::Create("instr1"));
  auto k2 = AbsoluteScopedTensorKey::Create(TensorKey::Create("instr2"));
  auto k_unknown =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("unknown"));
  std::vector<RecoveredTensorSummaryProto> summaries = {
      CreateSummaryProto(k2), CreateSummaryProto(k_unknown),
      CreateSummaryProto(k1)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "input_unknown.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "output_unknown.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK_AND_ASSIGN(auto callback,
                       sequencer.Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k1)),
                          EqualsProto(CreateSummaryProto(k2)),
                          EqualsProto(CreateSummaryProto(k_unknown))));
  EXPECT_TRUE((*callback)(k1));
  EXPECT_TRUE((*callback)(k2));
  EXPECT_TRUE((*callback)(k_unknown));
  EXPECT_FALSE(
      (*callback)(AbsoluteScopedTensorKey::Create(TensorKey::Create("other"))));
}

class OriginalTensorSummarySequencerCreateTest
    : public HloHardwareIndependentTestBase {};

TEST_F(OriginalTensorSummarySequencerCreateTest, CreateAndSequence) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule test_module_for_create

%computation1 (p: s32[]) -> s32[] {
  %p = s32[] parameter(0)
  ROOT %a = s32[] add(%p, %p)
}

ENTRY %entry_computation (p0: s32[]) -> s32[] {
  %p0 = s32[] parameter(0)
  %c = s32[] call(%p0), to_apply=%computation1
  ROOT %r = s32[] add(%c, %c)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto sequencer,
                       OriginalTensorSummarySequencer::Create(module.get()));

  auto k_p0 = AbsoluteScopedTensorKey::Create(TensorKey::Create("p0"));
  auto k_p = AbsoluteScopedTensorKey::Create(TensorKey::Create("p"),
                                             {ScopeInstruction::Create("c")});
  auto k_a = AbsoluteScopedTensorKey::Create(TensorKey::Create("a"),
                                             {ScopeInstruction::Create("c")});
  auto k_c = AbsoluteScopedTensorKey::Create(TensorKey::Create("c"));
  auto k_r = AbsoluteScopedTensorKey::Create(TensorKey::Create("r"));

  std::vector<RecoveredTensorSummaryProto> summaries = {
      CreateSummaryProto(k_r), CreateSummaryProto(k_c), CreateSummaryProto(k_a),
      CreateSummaryProto(k_p), CreateSummaryProto(k_p0)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "create_input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "create_output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK(sequencer->Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k_p0)),
                          EqualsProto(CreateSummaryProto(k_p)),
                          EqualsProto(CreateSummaryProto(k_a)),
                          EqualsProto(CreateSummaryProto(k_c)),
                          EqualsProto(CreateSummaryProto(k_r))));
}

TEST_F(OriginalTensorSummarySequencerCreateTest,
       SortsWhileConditionBeforeBody) {
  constexpr absl::string_view hlo_string = R"hlo(
HloModule while_test

%while_body(body_param: (s32[], s32[])) -> (s32[], s32[]) {
  %body_param = (s32[], s32[]) parameter(0)
  %body_gte0 = s32[] get-tuple-element(%body_param), index=0
  %c1 = s32[] constant(1)
  %add = s32[] add(%body_gte0, %c1)
  %body_gte1 = s32[] get-tuple-element(%body_param), index=1
  ROOT %body_root = (s32[], s32[]) tuple(%add, %body_gte1)
}

%while_condition(cond_param: (s32[], s32[])) -> pred[] {
  %cond_param = (s32[], s32[]) parameter(0)
  %cond_gte = s32[] get-tuple-element(%cond_param), index=0
  %c10 = s32[] constant(10)
  ROOT %compare = pred[] compare(%cond_gte, %c10), direction=LT
}

ENTRY main {
  %init_val = (s32[], s32[]) parameter(0)
  ROOT %while_op = (s32[], s32[]) while(%init_val), condition=%while_condition, body=%while_body
}
)hlo";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(auto sequencer,
                       OriginalTensorSummarySequencer::Create(module.get()));

  auto k_cond = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("compare"), {ScopeInstruction::Create("while_op", 1)});
  auto k_body = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("add"), {ScopeInstruction::Create("while_op", 1)});

  std::vector<RecoveredTensorSummaryProto> summaries = {
      CreateSummaryProto(k_body), CreateSummaryProto(k_cond)};
  const std::string input_path =
      tsl::io::JoinPath(::testing::TempDir(), "while_input.riegeli");
  const std::string output_path =
      tsl::io::JoinPath(::testing::TempDir(), "while_output.riegeli");
  WriteSummaries(input_path, summaries);
  ASSERT_OK(sequencer->Sequence(input_path, output_path));
  EXPECT_THAT(ReadSummaries(output_path),
              ElementsAre(EqualsProto(CreateSummaryProto(k_cond)),
                          EqualsProto(CreateSummaryProto(k_body))));
}

}  // namespace
}  // namespace xla::numerics::comparison
