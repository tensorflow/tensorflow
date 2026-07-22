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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/codegen/tools/test_lib.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"

struct Flags {
  std::string input_file = "";
  float abs_error_bound = 1e-6;
  float rel_error_bound = 1e-6;
  bool has_bijection_inputs = false;
};

Flags& flags = *new Flags;

namespace xla::cpu {
namespace {

using CpuCorrectnessTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(CpuCorrectnessTest, RunAndCompare) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadTestModule(flags.input_file));
  auto preprocessor = [](HloModule* mod) {
    for (HloComputation* comp : mod->computations()) {
      if (comp->root_instruction()->opcode() == HloOpcode::kTuple) continue;
      std::vector<HloInstruction*> dus_insts;
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kDynamicUpdateSlice) {
          dus_insts.push_back(inst);
        }
      }
      for (HloInstruction* inst : dus_insts) {
        HloInstruction* zero =
            comp->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::Zero(inst->shape().element_type())));
        if (inst->shape().dimensions_size() > 0) {
          zero = comp->AddInstruction(
              HloInstruction::CreateBroadcast(inst->shape(), zero, {}));
        }
        HloInstruction* add = comp->AddInstruction(HloInstruction::CreateBinary(
            inst->shape(), HloOpcode::kAdd, inst, zero));
        std::vector<HloInstruction*> users;
        for (HloInstruction* user : inst->users()) {
          if (user != add) {
            users.push_back(user);
          }
        }
        TF_CHECK_OK(inst->ReplaceUsesWith(users, add));
      }
    }
  };
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module),
      ErrorSpec{flags.abs_error_bound, flags.rel_error_bound},
      /*reference_preprocessor=*/preprocessor,
      /*test_preprocessor=*/preprocessor));
}

}  // namespace
}  // namespace xla::cpu

int main(int argc, char* argv[]) {
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("abs_error_bound", &flags.abs_error_bound,
                "Absolute error bound."),
      tsl::Flag("rel_error_bound", &flags.rel_error_bound,
                "Relative error bound."),
      tsl::Flag(
          "bijection_inputs",
          [](std::string val) {
            flags.has_bijection_inputs = true;
            return true;
          },
          "", "Bijection inputs flag."),
      tsl::Flag(
          "bijection_outputs", [](std::string val) { return true; }, "",
          "Bijection outputs flag."),
  };

  xla::AppendDebugOptionsFlags(&flag_list);
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parseResult = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parseResult || argc != 2) {
    LOG(ERROR) << "\n" << usage;
    return 1;
  }

  flags.input_file = argv[1];
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
