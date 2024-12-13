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
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

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
  TF_CHECK_OK(m1.set_schedule(schedule));
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

  EXPECT_EQ(m1.CrossProgramPrefetches().front().alt_memory_offset,
            m2->CrossProgramPrefetches().front().alt_memory_offset);

  EXPECT_EQ(m1.computation_count(), m2->computation_count());
  size_t i = 0;
  for (auto it1 = m1.computations().begin(), it2 = m2->computations().begin();
       it1 != m1.computations().end() && it2 != m2->computations().end();
       ++it1, ++it2) {
    const HloComputation *c1 = *it1, *c2 = *it2;
    EXPECT_EQ(GetCloneName(c1->name()), c2->name())
        << "Computation sequence mismatch at " << i;
    EXPECT_EQ(GetCloneName(m1.mutable_computation(i)->name()),
              m2->mutable_computation(i)->name())
        << "Indexing computation sequence mismatch at " << i;
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

}  // namespace
}  // namespace xla
