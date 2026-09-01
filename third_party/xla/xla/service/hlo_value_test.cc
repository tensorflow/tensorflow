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

#include "xla/service/hlo_value.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

void BM_IsRootOf_BadScenario(benchmark::State& state) {
  int num_positions = state.range(0);

  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});

  HloModuleConfig config;
  HloModule module("TestModule", config);

  HloComputation::Builder builder("TestComputation");
  builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "param_0"));
  HloComputation* computation = module.AddEntryComputation(builder.Build());

  HloComputation::Builder builder2("TestComputation2");
  HloInstruction* non_root_inst = builder2.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param_1"));
  module.AddEmbeddedComputation(builder2.Build());

  HloValue value(1, non_root_inst, /*index=*/{});

  std::vector<std::unique_ptr<HloInstruction>> dummy_insts;
  std::vector<HloPosition> dummy_positions;
  for (int i = 0; i < num_positions; ++i) {
    auto inst = HloInstruction::CreateParameter(0, shape, "dummy_param");
    dummy_positions.push_back({inst.get(), {}});
    dummy_insts.push_back(std::move(inst));
  }

  value.SetPositions(dummy_positions);

  for (auto _ : state) {
    bool is_root = value.IsRootOf(computation);
    benchmark::DoNotOptimize(is_root);
  }
}

BENCHMARK(BM_IsRootOf_BadScenario)->Range(8, 16384);

TEST(HloValueTest, DummyTest) {}

}  // namespace
}  // namespace xla
