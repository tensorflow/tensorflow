/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_domain_isolator.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_domain_remover.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloDomainTest : public HloVerifiedTestBase {
 protected:
  bool FindUserViaDomainPath(HloInstruction* instruction,
                             HloInstruction* operand) const {
    for (HloInstruction* user : operand->users()) {
      if (user == instruction) {
        return true;
      }
      if (user->opcode() == HloOpcode::kDomain &&
          FindUserViaDomainPath(instruction, user)) {
        return true;
      }
    }
    return false;
  }

  // Checks whether there is a kDomain instruction in the edge between the
  // instruction and the operand.
  bool HasDomainEdge(HloModule* module,
                     tensorflow::StringPiece instruction_name,
                     tensorflow::StringPiece operand_name) {
    HloInstruction* instruction = FindInstruction(module, instruction_name);
    HloInstruction* operand = FindInstruction(module, operand_name);
    CHECK_NE(instruction, nullptr);
    CHECK_NE(operand, nullptr);
    if (!instruction->IsUserOf(operand)) {
      // If instruction is not an immediate user, we must find a path from
      // operand to instruction anyway, otherwise there is a corruption.
      if (FindUserViaDomainPath(instruction, operand)) {
        return true;
      }
      LOG(FATAL) << "Bad HLO module generated across the '" << instruction_name
                 << "' and '" << operand_name << "' instructions:\n"
                 << module->ToString();
    }
    return false;
  }

  StatusOr<HloModule*> ParseModule(tensorflow::StringPiece hlo_string) {
    HloModuleConfig config;
    config.set_debug_options(legacy_flags::GetDebugOptionsFromFlags());
    ParseAndVerifyModule(hlo_string, config);
    return &module();
  }
};

// Dummy DomainMetadata implementation which create kDomain boundaries around
// HLO instructions with the same metadata().op_name() values.
class OpNameMetadata : public DomainMetadata {
 public:
  explicit OpNameMetadata(string opname) : opname_(std::move(opname)) {}

  std::unique_ptr<DomainMetadata> Clone() const override {
    return MakeUnique<OpNameMetadata>(opname_);
  }

  tensorflow::StringPiece Kind() const override { return KindName(); }

  bool Matches(const DomainMetadata& other) const override {
    const OpNameMetadata* other_ptr =
        dynamic_cast<const OpNameMetadata*>(&other);
    if (other_ptr == nullptr) {
      // If other is not a OpNameMetadata, then it is clearly a no match.
      return false;
    }
    return opname_ == other_ptr->opname_;
  }

  string ToString() const override { return opname_; }

  Status NormalizeInstructions(
      const DomainMetadata::Domain& domain) const override {
    // For the purposes of this test, nothing to do.
    return Status::OK();
  }

  static tensorflow::StringPiece KindName() { return "opname"; }

 private:
  string opname_;
};

// Creator function for OpNameMetadata domains.
std::unique_ptr<HloInstruction> OpNameDomainCreator(HloInstruction* instruction,
                                                    HloInstruction* operand) {
  if (instruction->metadata().op_name() == operand->metadata().op_name()) {
    return nullptr;
  }
  std::unique_ptr<DomainMetadata> operand_side_metadata =
      MakeUnique<OpNameMetadata>(operand->metadata().op_name());
  std::unique_ptr<DomainMetadata> user_side_metadata =
      MakeUnique<OpNameMetadata>(instruction->metadata().op_name());
  return HloInstruction::CreateDomain(operand->shape(), operand,
                                      std::move(operand_side_metadata),
                                      std::move(user_side_metadata));
}

Status OpNameDomainNormalizer(const DomainMetadata::Domain& domain) {
  // Nothing to do for the particular use this test make of the OpName domains.
  return Status::OK();
}

TEST_F(HloDomainTest, CheckDomainLinks) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[4], f32[4]) parameter(0)
  a = f32[4] get-tuple-element(p0), index=0
  b = f32[4] get-tuple-element(p0), index=1
  c = f32[4] add(f32[4] a, f32[4] b), sharding={maximal device=1}
  d = f32[4] subtract(a, b), sharding={maximal device=1}
  e = f32[4] multiply(c, d), sharding={maximal device=1}
  ROOT f = (f32[4], f32[4], f32[4]) tuple(c, d, e)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(CreateShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "c", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "c", "b"));
  EXPECT_TRUE(HasDomainEdge(module, "d", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "d", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "c"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "d"));

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  EXPECT_FALSE(HasDomainEdge(module, "c", "a"));
  EXPECT_FALSE(HasDomainEdge(module, "c", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "d", "a"));
  EXPECT_FALSE(HasDomainEdge(module, "d", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "c"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "d"));
}

TEST_F(HloDomainTest, CheckNoDomainAddedIfNoSharding) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[4], f32[4]) parameter(0)
  a = f32[4] get-tuple-element(p0), index=0
  b = f32[4] get-tuple-element(p0), index=1
  c = f32[4] add(f32[4] a, f32[4] b)
  d = f32[4] subtract(a, b)
  e = f32[4] multiply(c, d)
  ROOT f = (f32[4], f32[4], f32[4]) tuple(c, d, e)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(CreateShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(!isolator_changed);
}

TEST_F(HloDomainTest, CheckDomainAroundIO) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[4]) parameter(0)
  a = f32[4] get-tuple-element(p0), index=0
  b = (f32[4], u32[]) send(a), channel_id=1, sharding={maximal device=0}
  c = () send-done(b), channel_id=1, sharding={maximal device=0}
  d = (f32[4], u32[]) recv(), channel_id=2, sharding={maximal device=0}
  e = f32[4] recv-done(d), channel_id=2, sharding={maximal device=0}
  f = f32[4] add(a, e)
  g = f32[4] subtract(a, e)
  ROOT h = (f32[4], f32[4]) tuple(f, g)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(CreateShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "b", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "f", "e"));
  EXPECT_FALSE(HasDomainEdge(module, "a", "p0"));
  EXPECT_FALSE(HasDomainEdge(module, "c", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "d"));

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  EXPECT_FALSE(HasDomainEdge(module, "b", "a"));
  EXPECT_FALSE(HasDomainEdge(module, "f", "e"));
}

TEST_F(HloDomainTest, CheckNoDomainAddedOnPureIOComputation) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  a = (f32[4], u32[]) recv(), channel_id=1, sharding={maximal device=-1}
  b = f32[4] recv-done(a), channel_id=1, sharding={maximal device=-1}
  c = f32[4] add(b, b), sharding={maximal device=-1}
  d = (f32[4], u32[]) send(c), channel_id=2, sharding={maximal device=-1}
  ROOT e = () send-done(d), channel_id=2, sharding={maximal device=-1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(CreateShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_FALSE(isolator_changed);
}

TEST_F(HloDomainTest, CheckNormalizationOnPureIOComputation) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  a = (f32[4], u32[]) recv(), channel_id=1, sharding={maximal device=0}
  b = f32[4] recv-done(a), channel_id=1, sharding={maximal device=0}
  c = f32[4] add(b, b)
  d = (f32[4], u32[]) send(c), channel_id=2, sharding={maximal device=0}
  ROOT e = () send-done(d), channel_id=2, sharding={maximal device=0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_FALSE(remover_changed);

  HloInstruction* add = FindInstruction(module, "c");
  ASSERT_NE(add, nullptr);
  auto device = add->sharding_unique_device();
  EXPECT_TRUE(device.has_value());
  EXPECT_EQ(*device, 0);
}

TEST_F(HloDomainTest, CheckMultiDomainLinks) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[4], f32[4]) parameter(0)
  a = f32[4] get-tuple-element(p0), index=0
  b = f32[4] get-tuple-element(p0), index=1
  c = f32[4] add(a, b), sharding={maximal device=1}
  d = f32[4] subtract(a, c), sharding={maximal device=1}, metadata={op_name="D"}
  e = f32[4] multiply(c, d), sharding={maximal device=1}, metadata={op_name="D"}
  f = f32[4] add(e, c), sharding={maximal device=1}
  ROOT g = (f32[4], f32[4], f32[4]) tuple(c, d, f)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator sharding_isolator(CreateShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool sharding_isolator_changed,
                          sharding_isolator.Run(module));
  EXPECT_TRUE(sharding_isolator_changed);

  HloDomainIsolator opname_isolator(OpNameDomainCreator);
  TF_ASSERT_OK_AND_ASSIGN(bool opname_isolator_changed,
                          opname_isolator.Run(module));
  EXPECT_TRUE(opname_isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "c", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "c", "b"));
  EXPECT_TRUE(HasDomainEdge(module, "d", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "d", "c"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "d"));

  HloDomainRemover sharding_remover(ShardingMetadata::KindName(),
                                    NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool sharding_remover_changed,
                          sharding_remover.Run(module));
  EXPECT_TRUE(sharding_remover_changed);

  HloDomainRemover opname_remover(OpNameMetadata::KindName(),
                                  OpNameDomainNormalizer);
  TF_ASSERT_OK_AND_ASSIGN(bool opname_remover_changed,
                          opname_remover.Run(module));
  EXPECT_TRUE(opname_remover_changed);

  EXPECT_FALSE(HasDomainEdge(module, "c", "a"));
  EXPECT_FALSE(HasDomainEdge(module, "c", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "d", "a"));
  EXPECT_FALSE(HasDomainEdge(module, "d", "c"));
}

TEST_F(HloDomainTest, CheckNormalizationOnInfeedTuple) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  token = token[] generate-token()
  infeed = ((f32[4], f32[4]), token[]) infeed(token),
    sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  infeed.data = (f32[4], f32[4]) get-tuple-element(infeed), index=0
  gte0 = f32[4] get-tuple-element(infeed.data), index=0
  gte1 = f32[4] get-tuple-element(infeed.data), index=1
  copy0 = f32[4] copy(gte0)
  copy1 = f32[4] copy(gte1)
  ROOT add = f32[4] add(copy0, copy1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(CreateShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "infeed.data", "infeed"));
  EXPECT_FALSE(HasDomainEdge(module, "copy0", "gte0"));
  EXPECT_FALSE(HasDomainEdge(module, "copy1", "gte1"));

  // Inject unassigned tuple/gte within the infeed domain, to simulate the
  // HLO passes adding unexpected instructions.
  //
  //            infeed
  //              |
  //          infeed.data (tuple element 0 of infeed)
  //           /      \
  //         GTE0    GTE1
  //         /          \
  //       COPY0       COPY1
  //          \         /
  //           \       /
  //             TUPLE
  //               |
  HloInstruction* infeed = FindInstruction(module, "infeed");
  ASSERT_NE(infeed, nullptr);
  HloInstruction* infeed_data =
      infeed->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetTupleElementShape(infeed->shape(), 0), infeed, 0));

  auto infeed_data_users = infeed_data->users();
  HloInstruction* new_gte0 = infeed_data->parent()->AddInstruction(
      HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetTupleElementShape(infeed_data->shape(), 0), infeed_data,
          0));
  HloInstruction* new_copy0 =
      infeed_data->parent()->AddInstruction(HloInstruction::CreateUnary(
          new_gte0->shape(), HloOpcode::kCopy, new_gte0));
  HloInstruction* new_gte1 = infeed_data->parent()->AddInstruction(
      HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetTupleElementShape(infeed_data->shape(), 1), infeed_data,
          1));
  HloInstruction* new_copy1 =
      infeed_data->parent()->AddInstruction(HloInstruction::CreateUnary(
          new_gte1->shape(), HloOpcode::kCopy, new_gte1));
  HloInstruction* new_tuple = infeed_data->parent()->AddInstruction(
      HloInstruction::CreateTuple({new_copy0, new_copy1}));
  for (HloInstruction* user : infeed_data_users) {
    TF_EXPECT_OK(infeed_data->ReplaceUseWith(user, new_tuple));
  }

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  struct Assignment {
    HloInstruction* instruction;
    int64 device;
  } assignments[] = {
      {new_gte0, 1},
      {new_copy0, 1},
      {new_gte1, 0},
      {new_copy1, 0},
  };
  for (auto& assignment : assignments) {
    auto device = assignment.instruction->sharding_unique_device();
    ASSERT_TRUE(device.has_value());
    EXPECT_EQ(*device, assignment.device);
  }
  EXPECT_TRUE(new_tuple->has_sharding());
  EXPECT_EQ(
      new_tuple->sharding(),
      HloSharding::Tuple(new_tuple->shape(), {HloSharding::AssignDevice(1),
                                              HloSharding::AssignDevice(0)}));
}

}  // namespace
}  // namespace xla
