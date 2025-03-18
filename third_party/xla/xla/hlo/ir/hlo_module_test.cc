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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

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

}  // namespace
}  // namespace xla
