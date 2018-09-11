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

#include "absl/memory/memory.h"
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
  bool HasDomainEdge(HloModule* module, absl::string_view instruction_name,
                     absl::string_view operand_name) {
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

  StatusOr<HloModule*> ParseModule(absl::string_view hlo_string) {
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
    return absl::make_unique<OpNameMetadata>(opname_);
  }

  absl::string_view Kind() const override { return KindName(); }

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

  static absl::string_view KindName() { return "opname"; }

 private:
  string opname_;
};

// Creator function for OpNameMetadata domains.
HloInstruction* OpNameDomainCreator(HloInstruction* instruction,
                                    HloInstruction* root,
                                    HloInstruction* operand) {
  if (instruction->metadata().op_name() == root->metadata().op_name()) {
    return nullptr;
  }
  std::unique_ptr<DomainMetadata> operand_side_metadata =
      absl::make_unique<OpNameMetadata>(root->metadata().op_name());
  std::unique_ptr<DomainMetadata> user_side_metadata =
      absl::make_unique<OpNameMetadata>(instruction->metadata().op_name());
  return operand->parent()->AddInstruction(HloInstruction::CreateDomain(
      operand->shape(), operand, std::move(operand_side_metadata),
      std::move(user_side_metadata)));
}

Status OpNameDomainNormalizer(const DomainMetadata::Domain& domain,
                              const DomainMetadata* metadata) {
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

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "c", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "c", "b"));
  EXPECT_TRUE(HasDomainEdge(module, "d", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "d", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "c"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "d"));

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           ShardingMetadata::NormalizeShardingDomain);
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

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(!isolator_changed);
}

TEST_F(HloDomainTest, CheckDomainAroundIO) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = (f32[4]) parameter(0)
  a = f32[4] get-tuple-element(p0), index=0
  token = token[] after-all()
  b = (f32[4], u32[], token[]) send(a, token), channel_id=1, sharding={maximal device=0}
  c = token[] send-done(b), channel_id=1, sharding={maximal device=0}
  d = (f32[4], u32[], token[]) recv(token), channel_id=2, sharding={maximal device=0}
  e = (f32[4], token[]) recv-done(d), channel_id=2, sharding={maximal device=0}
  e_element = f32[4] get-tuple-element(e), index=0, sharding={maximal device=0}
  f = f32[4] add(a, e_element)
  g = f32[4] subtract(a, e_element)
  ROOT h = (f32[4], f32[4]) tuple(f, g)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "b", "a"));
  EXPECT_TRUE(HasDomainEdge(module, "f", "e_element"));
  EXPECT_FALSE(HasDomainEdge(module, "a", "p0"));
  EXPECT_FALSE(HasDomainEdge(module, "c", "b"));
  EXPECT_FALSE(HasDomainEdge(module, "e", "d"));

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           ShardingMetadata::NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  EXPECT_FALSE(HasDomainEdge(module, "b", "a"));
  EXPECT_FALSE(HasDomainEdge(module, "f", "e_element"));
}

TEST_F(HloDomainTest, CheckNoDomainAddedOnPureIOComputation) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  token = token[] after-all(), sharding={maximal device=-1}
  a = (f32[4], u32[], token[]) recv(token), channel_id=1, sharding={maximal device=-1}
  b = (f32[4], token[]) recv-done(a), channel_id=1, sharding={maximal device=-1}
  b_element = f32[4] get-tuple-element(b), index=0, sharding={maximal device=-1}
  c = f32[4] add(b_element, b_element), sharding={maximal device=-1}
  d = (f32[4], u32[], token[]) send(c, token), channel_id=2, sharding={maximal device=-1}
  ROOT e = token[] send-done(d), channel_id=2, sharding={maximal device=-1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_FALSE(isolator_changed);
}

TEST_F(HloDomainTest, CheckNormalizationOnPureIOComputation) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  token = token[] after-all(), sharding={maximal device=0}
  a = (f32[4], u32[], token[]) recv(token), channel_id=1, sharding={maximal device=0}
  b = (f32[4], token[]) recv-done(a), channel_id=1, sharding={maximal device=0}
  b_element = f32[4] get-tuple-element(b), index=0, sharding={maximal device=0}
  c = f32[4] add(b_element, b_element)
  d = (f32[4], u32[], token[]) send(c, token), channel_id=2, sharding={maximal device=0}
  ROOT e = token[] send-done(d), channel_id=2, sharding={maximal device=0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           ShardingMetadata::NormalizeShardingDomain);
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

  HloDomainIsolator sharding_isolator(ShardingDomainCreator{});
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
                                    ShardingMetadata::NormalizeShardingDomain);
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
  token = token[] after-all()
  infeed = ((f32[4], f32[4]), token[]) infeed(token),
    sharding={{maximal device=1}, {maximal device=0}, {maximal device=0}}
  infeed.data = (f32[4], f32[4]) get-tuple-element(infeed), index=0,
    sharding={{maximal device=1}, {maximal device=0}}
  gte0 = f32[4] get-tuple-element(infeed.data), index=0
  gte1 = f32[4] get-tuple-element(infeed.data), index=1
  copy0 = f32[4] copy(gte0)
  copy1 = f32[4] copy(gte1)
  ROOT add = f32[4] add(copy0, copy1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

  HloDomainIsolator isolator(ShardingDomainCreator{});
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
  HloInstruction* infeed_data = FindInstruction(module, "infeed.data");
  ASSERT_NE(infeed_data, nullptr);

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
                           ShardingMetadata::NormalizeShardingDomain);
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

TEST_F(HloDomainTest, EmptyRootDomain) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  %param = f32[1] parameter(0), sharding={maximal device=0}
  %tuple = (f32[1]) tuple(%param),
    sharding={maximal device=1}
  ROOT %gte = f32[1] get-tuple-element(%tuple), index=0,
    sharding={maximal device=1}
})";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  EXPECT_TRUE(HasDomainEdge(module, "tuple", "param"));
  EXPECT_FALSE(HasDomainEdge(module, "gte", "tuple"));

  // Remove %tuple and %gte (tuple simplification)
  HloInstruction* gte = FindInstruction(module, "gte");
  HloInstruction* tuple = FindInstruction(module, "tuple");
  module->entry_computation()->set_root_instruction(tuple->mutable_operand(0));
  TF_EXPECT_OK(module->entry_computation()->RemoveInstruction(gte));
  TF_EXPECT_OK(module->entry_computation()->RemoveInstruction(tuple));

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           ShardingMetadata::NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->has_sharding());
  EXPECT_EQ(root->sharding(), HloSharding::AssignDevice(1));
}

// Tests that text dumps of domain instructions can be parsed back, in the
// specific case of null shardings.
TEST_F(HloDomainTest, DumpParseNullSharding) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {});
  auto sharding_md_0 = absl::make_unique<ShardingMetadata>(nullptr);
  auto sharding_md_1 = absl::make_unique<ShardingMetadata>(nullptr);
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p"));
  HloInstruction* domain = builder.AddInstruction(HloInstruction::CreateDomain(
      shape, param, std::move(sharding_md_0), std::move(sharding_md_1)));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, domain, domain));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  auto hlo_string = module->ToString();
  ASSERT_TRUE(ParseModule(hlo_string).status().ok());
}

// Tuple inputs are domain instructions.
TEST_F(HloDomainTest, DomainTuple) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = f32[4] parameter(0), sharding={maximal device=0}
  cst = u32[] constant(0), sharding={maximal device=1}
  tpl = (u32[], f32[4]) tuple(cst, p0),
    sharding={{maximal device=1}, {maximal device=0}}
  ROOT gte = f32[4] get-tuple-element(tpl), index=1, sharding={maximal device=0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  // Clear sharding of tpl instruction, in order to test domain sharding
  // application.
  auto tpl = FindInstruction(module, "tpl");
  tpl->clear_sharding();

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           ShardingMetadata::NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  EXPECT_EQ(HloSharding::Tuple(tpl->shape(), {HloSharding::AssignDevice(1),
                                              HloSharding::AssignDevice(0)}),
            tpl->sharding());
}

TEST_F(HloDomainTest, MultiDomainMultiUser) {
  const char* const hlo_string = R"(
  HloModule Module

ENTRY %entry (p0: (f32[4], f32[4])) -> (f32[4], f32[4], f32[4]) {
  %p0 = (f32[4], f32[4]) parameter(0)
  %a = f32[4]{0} get-tuple-element(%p0), index=0
  %domain = f32[4] domain(%a),
    domain={kind="sharding", entry={maximal device=1}, exit={maximal device=0}}
  %b = f32[4] get-tuple-element(%p0), index=1
  %domain.1 = f32[4] domain(%b),
    domain={kind="sharding", entry={maximal device=1}, exit={maximal device=0}}
  %c = f32[4] add(%domain, %domain.1), sharding={maximal device=1}
  %domain.2 = f32[4] domain(%c),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
  %d = f32[4] subtract(%domain, %c),
    sharding={maximal device=1}, metadata={op_name="D"}
  %domain.3 = f32[4] domain(%d),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
  %e = f32[4] multiply(%c, %d),
    sharding={maximal device=1}, metadata={op_name="D"}
  %f = f32[4] add(f32[4]{0} %e, f32[4]{0} %c), sharding={maximal device=1}
  %domain.4 = f32[4]{0} domain(%f),
    domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
  ROOT %g = (f32[4], f32[4], f32[4]) tuple(%domain.2, %domain.3, %domain.4)
})";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));
  LOG(INFO) << "Original module:\n" << module->ToString();

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
                                    ShardingMetadata::NormalizeShardingDomain);
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

// Emulate instructions inserted at top and bottom within nested tuple domain.
TEST_F(HloDomainTest, DomainTupleTopBottomInsert) {
  const char* const hlo_string = R"(
HloModule Module

ENTRY entry {
  p0 = f32[4] parameter(0), sharding={maximal device=1}
  p1 = (f32[5], f32[6]) parameter(1),
    sharding={{maximal device=1}, {maximal device=0}}
  tuple.0 = (f32[4], (f32[5], f32[6])) tuple(p0, p1),
    sharding={{maximal device=1}, {maximal device=1}, {maximal device=0}}
  ROOT res = (f32[5], f32[6]) get-tuple-element(tuple.0), index=1,
    sharding={{maximal device=1}, {maximal device=0}}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(HloModule * module, ParseModule(hlo_string));

  HloDomainIsolator isolator(ShardingDomainCreator{});
  TF_ASSERT_OK_AND_ASSIGN(bool isolator_changed, isolator.Run(module));
  EXPECT_TRUE(isolator_changed);

  // Clear sharding of tuple.0 instruction, in order to test domain sharding
  // application.
  auto tuple0 = FindInstruction(module, "tuple.0");
  tuple0->clear_sharding();

  // Insert the following instructons above and below tuple.0, to emulate other
  // passes effects:
  //                 COPY.0
  //             \    /
  //            TUPLE.0
  //              /    \
  //           COPY.1   \
  //            /        \
  //         GTE.0      GTE.1
  //           |          |
  //           |        COPY.2
  //            \       /
  //             \     /
  //             TUPLE.1
  //                |
  auto tuple0_users = tuple0->users();
  auto computation = tuple0->parent();
  HloInstruction* copy0 = computation->AddInstruction(
      HloInstruction::CreateUnary(tuple0->operand(1)->shape(), HloOpcode::kCopy,
                                  tuple0->mutable_operand(1)));
  TF_EXPECT_OK(tuple0->ReplaceOperandWith(1, copy0));

  HloInstruction* copy1 = computation->AddInstruction(
      HloInstruction::CreateUnary(tuple0->shape(), HloOpcode::kCopy, tuple0));
  HloInstruction* gte0 =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetTupleElementShape(copy1->shape(), 0), copy1, 0));
  HloInstruction* gte1 =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          ShapeUtil::GetTupleElementShape(tuple0->shape(), 1), tuple0, 1));
  HloInstruction* copy2 = computation->AddInstruction(
      HloInstruction::CreateUnary(gte1->shape(), HloOpcode::kCopy, gte1));
  HloInstruction* tuple1 =
      computation->AddInstruction(HloInstruction::CreateTuple({gte0, copy2}));

  for (HloInstruction* user : tuple0_users) {
    TF_EXPECT_OK(tuple0->ReplaceUseWith(user, tuple1));
  }

  HloDomainRemover remover(ShardingMetadata::KindName(),
                           ShardingMetadata::NormalizeShardingDomain);
  TF_ASSERT_OK_AND_ASSIGN(bool remover_changed, remover.Run(module));
  EXPECT_TRUE(remover_changed);

  EXPECT_TRUE(tuple0->has_sharding());
  EXPECT_EQ(HloSharding::Tuple(tuple0->shape(), {HloSharding::AssignDevice(1),
                                                 HloSharding::AssignDevice(1),
                                                 HloSharding::AssignDevice(0)}),
            tuple0->sharding());

  EXPECT_TRUE(copy0->has_sharding());
  EXPECT_EQ(HloSharding::Tuple(copy0->shape(), {HloSharding::AssignDevice(1),
                                                HloSharding::AssignDevice(0)}),
            copy0->sharding());

  // copy1 has partial information only from gte.0, so in the end it gets no
  // sharding at all. During propagation it does propagate the information from
  // gte.0 though, enabling Tuple.0 to be fully sharded.
  EXPECT_FALSE(copy1->has_sharding());

  EXPECT_TRUE(gte0->has_sharding());
  EXPECT_EQ(HloSharding::AssignDevice(1), gte0->sharding());

  EXPECT_TRUE(gte1->has_sharding());
  EXPECT_EQ(HloSharding::Tuple(gte1->shape(), {HloSharding::AssignDevice(1),
                                               HloSharding::AssignDevice(0)}),
            gte1->sharding());

  EXPECT_TRUE(copy2->has_sharding());
  EXPECT_EQ(HloSharding::Tuple(copy2->shape(), {HloSharding::AssignDevice(1),
                                                HloSharding::AssignDevice(0)}),
            copy2->sharding());

  EXPECT_TRUE(tuple1->has_sharding());
  EXPECT_EQ(tuple0->sharding(), tuple1->sharding());
}

}  // namespace
}  // namespace xla
