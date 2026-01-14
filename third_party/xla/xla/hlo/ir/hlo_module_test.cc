/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_module.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Pointwise;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

// Adapts the internal equals proto to work with PointWise
MATCHER(EqualsProto, "") {
  const auto& a = ::testing::get<0>(arg);
  const auto& b = ::testing::get<1>(arg);
  return ::testing::Matches(tsl::proto_testing::EqualsProto(b))(a);
}

TEST(HloModuleTest, AbslHashValue) {
  HloModule module1("temp_module", HloModuleConfig());
  HloModule module2("temp_module3", HloModuleConfig());
  EXPECT_EQ(absl::HashOf(module1), absl::HashOf(module2));

  absl::string_view hlo = R"(
      HloModule m1
        ENTRY main {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
        ROOT res = f32[] multiply(a, b)
      })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module3,
                          ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module4,
                          ParseAndReturnUnverifiedModule(hlo));
  EXPECT_EQ(absl::HashOf(*module3), absl::HashOf(*module4));
  EXPECT_NE(absl::HashOf(module1), absl::HashOf(*module4));
}

TEST(HloModuleTest, ToFingerprint) {
  auto fp = [](const HloModule& module,
               std::optional<absl::btree_map<std::string, NumericOrString>>
                   custom_fields) {
    return custom_fields.has_value()
               ? module.ToFingerprint(HloPrintOptions::ModuleFingerprint(),
                                      *custom_fields)
               : module.ToFingerprint(HloPrintOptions::ModuleFingerprint());
  };
  HloModule module1("m1", HloModuleConfig());
  HloModule module2("m2", HloModuleConfig());
  EXPECT_EQ(fp(module1, std::nullopt), fp(module2, std::nullopt));

  absl::string_view hlo = R"(
      HloModule m3
        ENTRY main {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
        ROOT res = f32[] multiply(a, b)
      })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module3,
                          ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module4,
                          ParseAndReturnUnverifiedModule(hlo));
  EXPECT_EQ(fp(*module3, std::nullopt), fp(*module4, std::nullopt));
  EXPECT_NE(fp(module1, std::nullopt), fp(*module4, std::nullopt));

  const absl::btree_map<std::string, NumericOrString> custom_fields = {
      {"parameter_0", int64_t{50}},
      {"parameter_1", "string_value"},
      {"parameter_2", 10.594},
  };

  EXPECT_NE(fp(*module3, custom_fields), fp(*module3, std::nullopt));
  EXPECT_NE(fp(module1, custom_fields), fp(*module4, custom_fields));

  EXPECT_EQ(fp(*module3, custom_fields), fp(*module4, custom_fields));
  EXPECT_EQ(fp(*module4, custom_fields), fp(*module4, custom_fields));

  const absl::btree_map<std::string, NumericOrString> custom_fields2 = {
      {"parameter_0", int64_t{1}},
  };
  EXPECT_NE(fp(*module3, custom_fields), fp(*module3, custom_fields2));
}

TEST(HloModuleTest, MutableAndReadOnlyConfigEquals) {
  HloModuleConfig config1;
  config1.set_device_type("GPU");
  HloModule m1("-", config1);
  EXPECT_EQ(m1.config().device_type(), "GPU");
  EXPECT_EQ(&m1.mutable_config(), &m1.config());
  EXPECT_EQ(m1.shared_config().get(), &m1.config());

  m1.mutable_config().set_device_type("TPU");

  EXPECT_EQ(m1.config().device_type(), "TPU");
  EXPECT_EQ(&m1.mutable_config(), &m1.config());
}

TEST(HloModuleTest, SharedConfig) {
  HloModuleConfig config1;
  config1.set_device_type("first");
  config1.set_device_memory_size(7);
  HloModule m1("-", config1);
  HloModule m2("-", m1.shared_config(),
               std::make_unique<CompilationEnvironments>());
  EXPECT_EQ(&m1.config(), &m2.config())
      << "Shared config referres to the same object.";
  EXPECT_EQ(m1.shared_config().use_count(), 3);
  m1.mutable_config().set_device_type("second");
  EXPECT_NE(&m1.config(), &m2.config()) << "Config is copied on modification.";
  EXPECT_EQ(m1.config().device_type(), "second");
  EXPECT_EQ(m2.config().device_type(), "first");
  EXPECT_EQ(m1.config().device_memory_size(), m2.config().device_memory_size());
  EXPECT_EQ(m1.shared_config().use_count(), 2);
  EXPECT_EQ(m2.shared_config().use_count(), 2);
}

// Common patter across XLA. Besides possibility of a dangling pointer issue
// this pattern creates 2 copies. A better way is to use
// HloModule::shared_config()
TEST(HloModuleTest, GetModifySetConfig) {
  HloModuleConfig config1;
  config1.set_device_type("GPU");
  config1.set_device_memory_size(7);
  HloModule m1("-", config1);
  HloModuleConfig temp = m1.config();  // copy
  EXPECT_NE(&temp, &m1.config());
  temp.set_device_type("TPU");
  m1.set_config(temp);  // copy
  EXPECT_EQ(m1.config().device_type(), "TPU");
  EXPECT_EQ(m1.config().device_memory_size(), 7);
  EXPECT_EQ(&m1.config(), &m1.mutable_config());
}

void CreateComputation(HloModule& module, absl::string_view name, bool is_entry,
                       HloSchedule& schedule) {
  HloComputation::Builder builder(name);
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});

  builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));

  HloComputation* c =
      module.AddComputationAndUnifyNamesAndIds(builder.Build(), is_entry);
  schedule.set_sequence(c, {c->root_instruction()});

  if (!is_entry) {
    HloInstruction* call = module.entry_computation()->AddInstruction(
        HloInstruction::CreateCall(shape, {}, c));
    HloInstructionSequence& sequence =
        schedule.GetOrCreateSequence(module.entry_computation());
    sequence.push_back(call);
  }
}

const char* kCloneSuffix = "clone";

std::string GetCloneName(absl::string_view name) {
  return absl::StrCat(name, ".", kCloneSuffix);
}

TEST(HloModuleTest, CloneGeneral) {
  HloModule m1("temp_module", HloModuleConfig());
  HloSchedule schedule(&m1);
  CreateComputation(m1, "TestComputation1", true, schedule);
  CreateComputation(m1, "TestComputation3", false, schedule);
  CreateComputation(m1, "TestComputation2", false, schedule);
  m1.metadata()->set_module_group_name("test");
  CHECK_OK(m1.set_schedule(schedule));
  m1.AddCrossProgramPrefetch(7, ShapeIndex({8}), 100);

  std::unique_ptr<HloModule> m2 = m1.Clone(kCloneSuffix);

  EXPECT_EQ(&m1.config(), &m2->config());
  EXPECT_EQ(GetCloneName(m1.entry_computation()->name()),
            m2->entry_computation()->name());

  EXPECT_EQ(m1.schedule()
                .sequence(m1.entry_computation())
                .instructions()
                .front()
                ->name(),
            m2->schedule()
                .sequence(m2->entry_computation())
                .instructions()
                .front()
                ->name());
  EXPECT_EQ(m1.metadata()->proto().module_group_name(), "test");
  EXPECT_EQ(m2->metadata()->proto().module_group_name(), "test");
  EXPECT_EQ(m1.metadata()->proto().canonical_module_id(), m1.unique_id());
  EXPECT_EQ(m2->metadata()->proto().canonical_module_id(), m2->unique_id());

  EXPECT_EQ(m1.CrossProgramPrefetches().front().alt_memory_offset,
            m2->CrossProgramPrefetches().front().alt_memory_offset);

  EXPECT_EQ(m1.computation_count(), m2->computation_count());
  size_t i = 0;
  auto m1_computations = m1.computations();
  auto m2_computations = m2->computations();
  for (auto it1 = m1_computations.begin(), it2 = m2_computations.begin();
       it1 != m1_computations.end() && it2 != m2_computations.end();
       ++it1, ++it2) {
    const HloComputation *c1 = *it1, *c2 = *it2;
    EXPECT_EQ(GetCloneName(c1->name()), c2->name())
        << "Computation sequence mismatch at " << i;
  }
}

TEST(HloModuleTest, CloneWithContextGeneral) {
  HloModule m1("temp_module", HloModuleConfig());
  HloSchedule schedule(&m1);
  CreateComputation(m1, "TestComputation1", true, schedule);
  CreateComputation(m1, "TestComputation3", false, schedule);
  CreateComputation(m1, "TestComputation2", false, schedule);
  CHECK_OK(m1.set_schedule(schedule));
  m1.AddCrossProgramPrefetch(7, ShapeIndex({8}), 100);

  auto [m2, clone_context] = m1.CloneWithContext(kCloneSuffix);

  EXPECT_EQ(&m1.config(), &m2->config());
  EXPECT_EQ(GetCloneName(m1.entry_computation()->name()),
            m2->entry_computation()->name());

  EXPECT_EQ(m1.schedule()
                .sequence(m1.entry_computation())
                .instructions()
                .front()
                ->name(),
            m2->schedule()
                .sequence(m2->entry_computation())
                .instructions()
                .front()
                ->name());

  EXPECT_EQ(m1.CrossProgramPrefetches().front().alt_memory_offset,
            m2->CrossProgramPrefetches().front().alt_memory_offset);

  EXPECT_EQ(m1.computation_count(), m2->computation_count());
  size_t i = 0;
  auto m1_computations = m1.computations();
  auto m2_computations = m2->computations();
  for (auto it1 = m1_computations.begin(), it2 = m2_computations.begin();
       it1 != m1_computations.end() && it2 != m2_computations.end();
       ++it1, ++it2) {
    const HloComputation *c1 = *it1, *c2 = *it2;
    EXPECT_EQ(GetCloneName(c1->name()), c2->name())
        << "Computation sequence mismatch at " << i;
    EXPECT_EQ(clone_context->FindComputation(c1), c2);
  }
}

TEST(HloModuleTest, CloneAndShareConfig) {
  HloModule m1("-", HloModuleConfig());
  std::unique_ptr<HloModule> pm2 = m1.Clone(kCloneSuffix);
  EXPECT_EQ(&m1.config(), &pm2->config());
  EXPECT_EQ(m1.shared_config().use_count(), 3);
}

TEST(HloModuleTest, CloneWithNewConfig) {
  HloModuleConfig config1;
  config1.set_device_type("GPU");
  config1.set_device_memory_size(7);
  HloModule m1("-", HloModuleConfig());

  HloModuleConfig temp = m1.config();  // copy
  temp.set_device_memory_size(10);

  std::unique_ptr<HloModule> pm2 = m1.Clone("clone", temp);

  EXPECT_NE(&m1.config(), &pm2->config());
  EXPECT_EQ(m1.shared_config().use_count(), 2);
  EXPECT_EQ(pm2->shared_config().use_count(), 2);
  EXPECT_EQ(pm2->config().device_type(), m1.config().device_type());
  EXPECT_NE(pm2->config().device_memory_size(),
            m1.config().device_memory_size());
}

TEST(HloModuleTest, ClonePreservesStackFrameIndex) {
  HloModule m1("temp_module", HloModuleConfig());
  HloSchedule schedule(&m1);
  CreateComputation(m1, "TestComputation1", true, schedule);
  CHECK_OK(m1.set_schedule(schedule));

  StackFrameIndexProto stack_frame_index;
  stack_frame_index.add_file_names("file1.cc");
  stack_frame_index.add_function_names("func1");
  auto* file_location = stack_frame_index.add_file_locations();
  file_location->set_file_name_id(1);
  file_location->set_function_name_id(1);
  file_location->set_line(10);
  file_location->set_column(5);
  auto* stack_frame = stack_frame_index.add_stack_frames();
  stack_frame->set_file_location_id(1);
  stack_frame->set_parent_frame_id(0);
  m1.set_stack_frame_index(stack_frame_index);

  std::unique_ptr<HloModule> m2 = m1.Clone(kCloneSuffix);

  EXPECT_TRUE(m2->stack_frame_index().has_value());
  EXPECT_THAT(m2->stack_frame_index().value(),
              tsl::proto_testing::EqualsProto(stack_frame_index));
}

TEST(HloModuleTest, UniqueIdProvidesComputationPrefix) {
  HloModule m1("temp_module", HloModuleConfig());
  HloSchedule schedule(&m1);
  CreateComputation(m1, "TestComputation1", true, schedule);
  CreateComputation(m1, "TestComputation2", false, schedule);
  CreateComputation(m1, "TestComputation3", false, schedule);
  TF_EXPECT_OK(m1.set_schedule(schedule));

  EXPECT_EQ(m1.GetComputationWithName("TestComputation1")
                ->GetInstructionWithName("p0")
                ->unique_id(),
            0);
  EXPECT_EQ(m1.GetComputationWithName("TestComputation2")
                ->GetInstructionWithName("p0.1")
                ->unique_id(),
            (static_cast<int64_t>(1) << 32) + 0);
  EXPECT_EQ(m1.GetComputationWithName("TestComputation1")
                ->GetInstructionWithName("call")
                ->unique_id(),
            1);
  EXPECT_EQ(m1.GetComputationWithName("TestComputation3")
                ->GetInstructionWithName("p0.2")
                ->unique_id(),
            (static_cast<int64_t>(2) << 32) + 0);
  EXPECT_EQ(m1.GetComputationWithName("TestComputation1")
                ->GetInstructionWithName("call.1")
                ->unique_id(),
            2);
}

TEST(HloModuleTest, ClonePreservesUniqueId) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule m

    add {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    // HloModule::Clone() deletes dead code.
    dead_code {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
      p0 = f16[10000000]{0} parameter(0)
      p1 = f16[10000000]{0} parameter(1)
      ar0 = f16[10000000]{0} all-reduce(p0), replica_groups={}, to_apply=add
      ar1 = f16[10000000]{0} all-reduce(p1), replica_groups={}, to_apply=add
      ROOT result = tuple(ar0, ar1)
    }
  )"));

  // Annotate all instructions with a unique id. Frontend attributes are
  // preserved when cloning.
  static constexpr char kUniqueIdAttr[] = "collective_id";
  hlo_query::ForEachInstructionWithPred(
      *module, HloPredicateTrue, [](HloInstruction* instr) {
        instr->set_frontend_attribute(kUniqueIdAttr,
                                      absl::StrCat(instr->unique_id()));
      });

  std::unique_ptr<HloModule> clone = module->Clone(kCloneSuffix);
  hlo_query::ForEachInstructionWithPred(
      *clone, HloPredicateTrue, [](HloInstruction* instr) {
        EXPECT_EQ(instr->get_frontend_attribute(kUniqueIdAttr),
                  absl::StrCat(instr->unique_id()))
            << "unique_id differs for " << instr->ToString();
      });
}

TEST(HloModuleTest, AbslHashInstructionOrdering) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  // Add.0 and add.1 are swapped.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule HashTest
      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.1 = f32[32,32] add(b, c)    // Swapped with below
        add.0 = f32[32,32] add(a, b)    // Swapped with above
        ROOT result = f32[32,32] add(add.0, add.1)
      }
    )"));

  EXPECT_EQ(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, AbslHashInstructionOpcodes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  // Second add changed to sub
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule HashTest
      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] subtract(b, c)  // Changed from add to subtract
        ROOT result = f32[32,32] add(add.0, add.1)
      }
    )"));

  EXPECT_NE(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, AbslHashInstructionShapes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  // Second add has different shape.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule HashTest
      ENTRY main {
        // Shapes changed from [32,32] to [16,16]
        a = f32[16,16] parameter(0)
        b = f32[16,16] parameter(1)
        c = f32[16,16] parameter(2)
        add.0 = f32[16,16] add(a, b)
        add.1 = f32[16,16] add(b, c)
        ROOT result = f32[16,16] add(add.0, add.1)
      }
    )"));

  EXPECT_NE(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, AbslHashInstructionNaming) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  // Add x to all names
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        // All names changed to <name>x
        ax = f32[32,32] parameter(0)
        bx = f32[32,32] parameter(1)
        cx = f32[32,32] parameter(2)
        add.0x = f32[32,32] add(ax, bx)
        add.1x = f32[32,32] add(bx, cx)
        ROOT resultx = f32[32,32] add(add.0x, add.1x)
      }
      )"));

  EXPECT_EQ(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, AbslHashGraphChanges) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  // Changed from (a+b)+(b+c) to ((a+b)+c)+a
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(add.0, c)       // Changed from add(b, c)
        ROOT result = f32[32,32] add(add.1, a) // Changed from add(add.0, add.1)
      }
      )"));

  EXPECT_NE(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, AbslHashParameterChanges) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(0)
        b = f32[32,32] parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  // Change parameter numbers
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
      HloModule HashTest

      ENTRY main {
        a = f32[32,32] parameter(1)  // Changed from parameter(0)
        b = f32[32,32] parameter(0)  // Changed from parameter(1)
        c = f32[32,32] parameter(2)
        add.0 = f32[32,32] add(a, b)
        add.1 = f32[32,32] add(b, c)
        ROOT result = f32[32,32] add(add.0, add.1)
      }
      )"));

  EXPECT_NE(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, AbslHashConstantValues) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module1,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule HashTest

    ENTRY main {
      a = s32[32,32] parameter(0)
      c = s32[] constant(42)
      b = s32[32,32] broadcast(c), dimensions={}
      ROOT result = s32[32,32] add(a, b)
    }
      )"));

  // Changed from 42 to 43
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module2,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule HashTest

    ENTRY main {
      a = s32[32,32] parameter(0)
      c = s32[] constant(43)  // Changed from constant(42)
      b = s32[32,32] broadcast(c), dimensions={}
      ROOT result = s32[32,32] add(a, b)
    }
      )"));

  EXPECT_NE(absl::HashOf(*module1), absl::HashOf(*module2));
}

TEST(HloModuleTest, CheckToStringHonorsDebugOptions) {
  // Check that the debug options xla_dump_large_constants,
  // xla_syntax_sugar_async_ops are honored.
  const char* hlo = R"(
  HloModule test

  async_computation {
    a = f32[32,32] parameter(0)
    b = f32[32,32] parameter(1)
    ROOT result = f32[32,32] subtract(a, b)
  }

  ENTRY main {
    a = f32[32,32] parameter(0)
    b = f32[32,32] parameter(1)
    c = f32[32,32] parameter(2)
    add = f32[32,32] add(a, b), metadata={op_type="add", op_name="my_add", source_file="my_file.cc", source_line=123}
    large_constant = f32[16]{0} constant({42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42})
    async_start = ((f32[32,32], f32[32,32]), f32[32,32]) async-start(add, c), calls=async_computation
    async_done = f32[32,32] async-done(async_start)
    ROOT result = tuple(async_done, large_constant)
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));
  DebugOptions& db_options = module->mutable_config().mutable_debug_options();
  // Setting non-default values for these w.r.t the PrintOptions class.
  db_options.set_xla_dump_large_constants(true);
  db_options.set_xla_dump_disable_metadata(true);
  db_options.set_xla_syntax_sugar_async_ops(false);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), R"(
    // CHECK:     {{.+}} = f32[32,32]{1,0} add({{.+}}){{$}}
    // CHECK-NOT: subtract-start
    // CHECK-DAG: {{.+}} = f32[16]{0} constant({42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42})
    // CHECK-DAG: {{.+}} = ((f32[32,32]{1,0}, f32[32,32]{1,0}), f32[32,32]{1,0}) async-start({{.+}})
    // CHECK-NOT: subtract-done
  )"));
  EXPECT_TRUE(filecheck_matched);
}

TEST(HloModuleTest, TestCallersAndCallees) {
  const char* hlo = R"(
    HloModule jit_h

    f {
      p0 = f32[] parameter(0)
      ROOT sine.4 = f32[] sine(p0)
    }

    g {
      p0 = f32[] parameter(0)
      call.f.0 = f32[] call(p0), to_apply=f
      ROOT call.f.1 = f32[] call(call.f.0), to_apply=f
    }

    h {
      ROOT p0 = f32[] parameter(0)
    }

    uncalled {
      p0 = f32[] parameter(0)
      ROOT call.h = f32[] call(p0), to_apply=h
    }

    ENTRY main {
      Arg_0.1 = f32[] parameter(0)
      call.f.2 = f32[] call(Arg_0.1), to_apply=f
      call.g.0 = f32[] call(call.f.2), to_apply=g
      ROOT call.g.1 = f32[] call(call.g.0), to_apply=g
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));
  EXPECT_EQ(module->computation_count(), 5);
  HloComputation* main = module->GetComputationWithName("main");
  HloComputation* f = module->GetComputationWithName("f");
  HloComputation* g = module->GetComputationWithName("g");
  HloComputation* h = module->GetComputationWithName("h");
  HloComputation* uncalled = module->GetComputationWithName("uncalled");
  EXPECT_THAT(main->callee_computations(),
              ElementsAre(std::make_pair(f, 1), std::make_pair(g, 2)));
  EXPECT_THAT(f->callee_computations(), ElementsAre());
  EXPECT_THAT(g->callee_computations(), ElementsAre(std::make_pair(f, 2)));
  EXPECT_THAT(f->caller_computations(),
              ElementsAre(std::make_pair(g, 2), std::make_pair(main, 1)));
  EXPECT_THAT(g->caller_computations(), ElementsAre(std::make_pair(main, 2)));

  HloInstruction* call_f_0 = g->GetInstructionWithName("call.f.0");
  HloInstruction* call_f_1 = g->GetInstructionWithName("call.f.1");
  HloInstruction* call_f_2 = main->GetInstructionWithName("call.f.2");
  HloInstruction* call_g_0 = main->GetInstructionWithName("call.g.0");
  HloInstruction* call_g_1 = main->GetInstructionWithName("call.g.1");
  HloInstruction* call_h = uncalled->GetInstructionWithName("call.h");

  EXPECT_THAT(f->caller_instructions(),
              UnorderedElementsAre(call_f_0, call_f_1, call_f_2));
  EXPECT_THAT(g->caller_instructions(),
              UnorderedElementsAre(call_g_0, call_g_1));
  EXPECT_THAT(h->caller_instructions(), ElementsAre(call_h));
  EXPECT_THAT(uncalled->caller_instructions(), IsEmpty());
}

TEST(HloModuleTest, MultipleCallsFromOneInstruction) {
  const char* hlo = R"(
    f {
      tparam = f32[4] parameter(0)
      ROOT tuple = (f32[4]) tuple(tparam)
    }

    g {
      fparam = f32[4] parameter(0)
      ROOT tuple = (f32[4]) tuple(fparam)
    }

    ENTRY main {
      p0 = f32[4] parameter(0)
      b0 = s32[] parameter(1)
      ROOT conditional = (f32[4]) conditional(b0, p0, p0, p0),
        branch_computations={f, f, g}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));
  EXPECT_EQ(module->computation_count(), 3);
  HloComputation* main = module->GetComputationWithName("main");
  HloComputation* f = module->GetComputationWithName("f");
  HloComputation* g = module->GetComputationWithName("g");

  HloInstruction* conditional = main->GetInstructionWithName("conditional");

  EXPECT_THAT(f->caller_instructions(), ElementsAre(conditional));
  EXPECT_THAT(g->caller_instructions(), ElementsAre(conditional));
}

TEST(HloModuleTest, TestUniqueIdIs64Bits) {
  const char* hlo = R"(
    f {
      ROOT tparam = f32[4] parameter(0)
    }

    g {
      ROOT fparam = f32[4] parameter(0)
    }

    ENTRY main {
      p0 = f32[4] parameter(0)
      b0 = f32[4] parameter(1)
      call.f.0 = f32[4] call(p0), to_apply=f
      call.g.0 = f32[4] call(b0), to_apply=g
      ROOT sum = f32[4] add(call.f.0, call.g.0)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo));
  HloComputation* f = module->GetComputationWithName("f");
  HloInstruction* tparam = f->GetInstructionWithName("tparam");
  HloComputation* g = module->GetComputationWithName("g");
  HloInstruction* fparam = g->GetInstructionWithName("fparam");

  // Upper 32 bits should make them different
  EXPECT_NE(tparam->unique_id(), fparam->unique_id());
  // Lower 32 bits should be preserved and therefore the same
  EXPECT_EQ(tparam->unique_id() & 0xFFFFFFFF, fparam->unique_id() & 0xFFFFFFFF);
  TF_EXPECT_OK(module->CheckUniqueNamesAndIdsForComputationsAndInstructions());
}

TEST(HloModuleTest, TestRemapInstructionIdsResolvesOperands) {
  HloModuleProto hlo_module_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(R"(
  name: "hlo_module_proto"
  entry_computation_id: 1
computations {
  name: "basic_computation"
  id: 2
  root_id: 1
  instructions {
    name: "parameter.0"
    opcode: "parameter"
    id: 1
  }
  instructions {
    name: "parameter.1"
    opcode: "parameter"
    id: 2
  }
  instructions {
    name: "add.0"
    opcode: "add"
    id: 3
    operand_ids: 1
    operand_ids: 2
  }
  instructions {
    name: "add.1"
    opcode: "add"
    id: 4
    operand_ids: 1
    operand_ids: 3
  }
  instructions {
    name: "add.2"
    opcode: "add"
    id: 5
    operand_ids: 2
    operand_ids: 3
  }
}

computations {
  name: "entry_computation"
  id: 1
  root_id: 12884901895
  instructions {
    name: "Arg_0.1"
    opcode: "parameter"
    id: 12884901889
  }
  instructions {
    name: "slice.2"
    opcode: "slice"
    id: 12884901890
    operand_ids: 1
  }
  instructions {
    name: "squeeze.2"
    opcode: "reshape"
    id: 12884901891
    operand_ids: 2
  }
  instructions {
    name: "add.39"
    opcode: "broadcast"
    id: 12884901892
    operand_ids: 3
  }
  instructions {
    name: "iota_2x32_shape.1"
    opcode: "iota"
    id: 12884901893
  }
  instructions {
    name: "slice.3"
    opcode: "slice"
    id: 12884901894
    operand_ids: 1
  }
  instructions {
    name: "squeeze.3"
    opcode: "reshape"
    id: 7
    operand_ids: 6
  }
}

)",
                                                         &hlo_module_proto));

  TF_ASSERT_OK_AND_ASSIGN(HloModuleProto remapped_hlo_module_proto,
                          HloModule::RemapInstructionIds(hlo_module_proto));

  HloInstructionProto squeeze_3_instr =
      remapped_hlo_module_proto.computations(1).instructions(6);

  // Instruction squeeze.3's operand is slice.3, which should be remapped to
  // id 5.
  EXPECT_THAT(
      remapped_hlo_module_proto.computations(1).instructions(6).operand_ids(),
      ElementsAre(5));
  // squeeze.3 is the root because its local id matches the local part of the
  // root id specified in the proto.
  EXPECT_THAT(remapped_hlo_module_proto.computations(1).instructions(6).id(),
              Eq(remapped_hlo_module_proto.computations(1).root_id()));
}

TEST(HloModuleTest, LoadAndFixNonConsecutiveInstructionIds) {
  xla::HloModuleProto hlo_module_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        name: "some_module"
        entry_computation_name: "entry_computation"
        computations {
          name: "comp2"
          instructions {
            name: "arg0.comp2"
            opcode: "parameter"
            shape {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            id: 21474836499
          }
          instructions {
            name: "arg1.comp2"
            opcode: "parameter"
            shape {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            parameter_number: 1
            id: 21474836480
          }
          instructions {
            name: "add.comp2"
            opcode: "tuple"
            shape {
              element_type: TUPLE
              tuple_shapes {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
            }
            id: 21474836488
            operand_ids: 0
            operand_ids: 19
          }
          instructions {
            name: "XLA_Retvals.comp2"
            opcode: "tuple"
            shape {
              element_type: TUPLE
              tuple_shapes {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
            }
            id: 21474836487
            operand_ids: 0
          }
          id: 21
          root_id: 21474836487
        }
        computations {
          name: "entry_computation"
          instructions {
            name: "arg0.1"
            opcode: "parameter"
            shape {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            id: 4294967297
          }
          instructions {
            name: "arg1.1"
            opcode: "parameter"
            shape {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
            parameter_number: 1
            id: 4294967298
          }
          instructions {
            name: "XLA_Retvals.1"
            opcode: "tuple"
            shape {
              element_type: TUPLE
              tuple_shapes {
                element_type: S32
                layout { tail_padding_alignment_in_elements: 1 }
              }
            }
            id: 4294967303
            operand_ids: 1
          }
          id: 1
          root_id: 4294967303
        }
        host_program_shape {
          parameters {
            element_type: S32
            layout { tail_padding_alignment_in_elements: 1 }
          }
          parameters {
            element_type: S32
            layout { tail_padding_alignment_in_elements: 1 }
          }
          result {
            element_type: TUPLE
            tuple_shapes {
              element_type: S32
              layout { tail_padding_alignment_in_elements: 1 }
            }
          }
          parameter_names: "arg0"
          parameter_names: "arg1"
        }
        id: 1
        entry_computation_id: 1
        schedule {
          sequences {
            key: 1
            value {
              instruction_ids: 4294967297
              instruction_ids: 4294967298
              instruction_ids: 4294967303
            }
          }
          sequences {
            key: 21
            value {
              instruction_ids: 21474836499
              instruction_ids: 21474836480
              instruction_ids: 21474836488
              instruction_ids: 21474836487
            }
          }
        }
      )pb",
      &hlo_module_proto));

  TF_ASSERT_OK_AND_ASSIGN(HloModuleConfig config,
                          xla::HloModule::CreateModuleConfigFromProto(
                              hlo_module_proto, xla::DebugOptions()));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      xla::HloModule::CreateFromProto(hlo_module_proto, config,
                                      /* prohibit_empty_literal= */ true,
                                      /* comp_envs= */ nullptr,
                                      /* preserve_instruction_ids= */ false));

  EXPECT_EQ(module->computation_count(), 2);
  HloComputation* entry_computation = module->entry_computation();
  HloComputation* computation_2 = *std::next(module->computations().begin());
  EXPECT_EQ(entry_computation->instruction_count(), 3);

  EXPECT_EQ(computation_2->instruction_count(), 4);
  // Check that ids are consecutive
  EXPECT_THAT(entry_computation->instructions(),
              ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                          Property(&xla::HloInstruction::local_id, 1),
                          Property(&xla::HloInstruction::local_id, 2)));
  // Check correct operand translation for entry computation
  EXPECT_EQ(entry_computation->parameter_instruction(0)->name(), "arg0.1");
  EXPECT_EQ(entry_computation->parameter_instruction(0)->local_id(), 0);
  EXPECT_THAT(entry_computation->root_instruction()->operands(),
              ElementsAre(entry_computation->parameter_instruction(0)));
  // Check correct operand translation for computation 2
  EXPECT_THAT(computation_2->parameter_instructions(),
              ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                          Property(&xla::HloInstruction::local_id, 1)));
  EXPECT_THAT(computation_2->parameter_instructions(),
              ElementsAre(Property(&xla::HloInstruction::name, "arg0.comp2"),
                          Property(&xla::HloInstruction::name, "arg1.comp2")));
  // Retvals has operand with local id 0, which in the proto was arg1.comp2
  EXPECT_THAT(computation_2->root_instruction()->operands(),
              ElementsAre(computation_2->parameter_instruction(1)));
  // Check operands for add.comp2
  EXPECT_THAT(computation_2->GetInstructionWithName("add.comp2")->operands(),
              ElementsAre(computation_2->parameter_instruction(1),
                          computation_2->parameter_instruction(0)));
  // Check Hlo Schedule
  EXPECT_THAT(
      module->schedule().GetOrCreateSequence(entry_computation).instructions(),
      ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                  Property(&xla::HloInstruction::local_id, 1),
                  Property(&xla::HloInstruction::local_id, 2)));
  EXPECT_THAT(
      module->schedule().GetOrCreateSequence(computation_2).instructions(),
      ElementsAre(Property(&xla::HloInstruction::local_id, 0),
                  Property(&xla::HloInstruction::local_id, 1),
                  Property(&xla::HloInstruction::local_id, 2),
                  Property(&xla::HloInstruction::local_id, 3)));
}

TEST(HloModuleTest, TestHloModuleToFromProtoInvarianceInComputation) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(R"(
  HloModule test_module, is_scheduled=true, entry_computation_layout={(f32[]{:T(256)}, f32[100]{0:T(256)}, f32[100]{0:T(256)})->f32[100]{0:T(256)}}

  %fused_computation (param_0.1: f32[100], param_1.3: f32[100], param_2.1: f32[]) -> f32[100] {
    %param_2.1 = f32[]{:T(256)S(6)} parameter(2)
    %broadcast.1 = f32[100]{0:T(256)} broadcast(%param_2.1), dimensions={}
    %param_0.1 = f32[100]{0:T(256)} parameter(0)
    %param_1.3 = f32[100]{0:T(256)} parameter(1)
    %multiply.1 = f32[100]{0:T(256)} multiply(%broadcast.1, %param_1.3)
    %add.1 = f32[100]{0:T(256)} add(%multiply.1, %param_0.1)
    ROOT %subtract.1 = f32[100]{0:T(256)} subtract(%add.1, %param_0.1)
  }

  ENTRY %EntryComputation (p: f32[], p1: f32[100], p2: f32[100]) -> f32[100] {
    %p = f32[]{:T(256)} parameter(0)
    %copy = f32[]{:T(256)S(6)} copy(%p)
    %p2 = f32[100]{0:T(256)} parameter(2)
    %p1 = f32[100]{0:T(256)} parameter(1)
    ROOT %add_subtract_fusion = f32[100]{0:T(256)} fusion(%p2, %p1, %copy), kind=kLoop, calls=%fused_computation
                            })"));
  HloModuleProto module_proto = module->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module_from_proto,
      HloModule::CreateFromProto(module_proto, module->config(),
                                 /*buffer_assignment_proto=*/nullptr,
                                 /*preserve_instruction_ids=*/true));

  EXPECT_THAT(
      module_proto.computations(),
      Pointwise(EqualsProto(), module_from_proto->ToProto().computations()));
}

TEST(HloModuleTest, TestCreateFromProtoUpdatesBufferAssignment) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(R"(
  HloModule test_module, is_scheduled=true, entry_computation_layout={(f32[]{:T(256)}, f32[100]{0:T(256)}, f32[100]{0:T(256)})->f32[100]{0:T(256)}}

  %fused_computation (param_0.1: f32[100], param_1.3: f32[100], param_2.1: f32[]) -> f32[100] {
    %param_2.1 = f32[]{:T(256)S(6)} parameter(2)
    %broadcast.1 = f32[100]{0:T(256)} broadcast(%param_2.1), dimensions={}
    %param_0.1 = f32[100]{0:T(256)} parameter(0)
    %param_1.3 = f32[100]{0:T(256)} parameter(1)
    %multiply.1 = f32[100]{0:T(256)} multiply(%broadcast.1, %param_1.3)
    %add.1 = f32[100]{0:T(256)} add(%multiply.1, %param_0.1)
    ROOT %subtract.1 = f32[100]{0:T(256)} subtract(%add.1, %param_0.1)
  }

  ENTRY %EntryComputation (p: f32[], p1: f32[100], p2: f32[100]) -> f32[100] {
    %p = f32[]{:T(256)} parameter(0)
    %copy = f32[]{:T(256)S(6)} copy(%p)
    %p2 = f32[100]{0:T(256)} parameter(2)
    %p1 = f32[100]{0:T(256)} parameter(1)
    ROOT %add_subtract_fusion = f32[100]{0:T(256)} fusion(%p2, %p1, %copy), kind=kLoop, calls=%fused_computation
                            })"));

  TF_ASSERT_OK_AND_ASSIGN(
      HloModuleConfig config,
      HloModule::CreateModuleConfigFromShape(
          module->entry_computation()->ComputeProgramShape(), DebugOptions()));

  module->set_config(std::move(config));

  // Create and save the HLO proto and the buffer assignment proto for the HLO
  // module.
  HloProto opt_hlo_module_proto = MakeHloProto(*module);

  AliasInfo alias_info;
  BufferValue::SizeFunction buffer_size_func =
      [](const BufferValue& buffer) -> int64_t {
    return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
  };

  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer_assignment,
      BufferAssigner::Run(
          /*module=*/module.get(),
          /*hlo_ordering=*/
          std::make_unique<DependencyHloOrdering>(module.get()),
          /*buffer_size=*/std::move(buffer_size_func),
          /*alias_info=*/&alias_info,
          /*color_alignment=*/[](LogicalBuffer::Color) -> int64_t { return 1; },
          /*options=*/std::move(opts)));

  BufferAssignmentProto buffer_assignment_proto = buffer_assignment->ToProto();
  *opt_hlo_module_proto.mutable_buffer_assignment() = buffer_assignment_proto;

  // Replace instruction ids with non-consecutive ones
  absl::flat_hash_map<std::string, std::string> instruction_id_remap_map = {
      {"4294967298", "4294967323"},
      {"4294967299", "4294967324"},
      {"4294967296", "4294967363"},
      {"4294967297", "4294967423"},
      {"4294967300", "4294967523"}};

  std::string opt_hlo_module_proto_str;
  ASSERT_TRUE(tsl::protobuf::TextFormat::PrintToString(
      opt_hlo_module_proto, &opt_hlo_module_proto_str));

  ASSERT_GT(
      absl::StrReplaceAll(instruction_id_remap_map, &opt_hlo_module_proto_str),
      5);

  // Load modified HloProto from string and reassign ids instead of preserving
  // them.
  HloProto opt_hlo_module_proto_modified;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      opt_hlo_module_proto_str, &opt_hlo_module_proto_modified));

  // Recreate the hlo module from the altered protos.
  TF_ASSERT_OK_AND_ASSIGN(
      HloModuleConfig module_config_recreated,
      HloModule::CreateModuleConfigFromProto(
          opt_hlo_module_proto_modified.hlo_module(), DebugOptions()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module_recreated,
      HloModule::CreateFromProto(
          opt_hlo_module_proto_modified.hlo_module(), module_config_recreated,
          opt_hlo_module_proto_modified.mutable_buffer_assignment(),
          /*preserve_instruction_ids=*/false));

  buffer_size_func = [](const BufferValue& buffer) -> int64_t {
    return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
  };
  // Will fail if buffer assignment is not updated in the HLO proto.
  TF_EXPECT_OK(BufferAssignment::FromProto(
      opt_hlo_module_proto_modified.buffer_assignment(),
      hlo_module_recreated.get(), std::move(buffer_size_func), &alias_info));
}

}  // namespace
}  // namespace xla
