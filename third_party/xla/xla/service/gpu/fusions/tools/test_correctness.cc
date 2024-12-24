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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/service/gpu/fusions/tools/test_lib.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/shape.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

struct Flags {
  std::string input_file = "";
  float abs_error_bound = 1e-4;
  float rel_error_bound = 1e-4;
  std::vector<std::pair<std::string, std::vector<int64_t>>> bijection_inputs;
  std::vector<std::string> bijection_outputs;
};

Flags& flags = *new Flags;

namespace xla {
namespace gpu {
namespace {

using CorrectnessTest = HloTestBase;

const Shape& GetFirstArrayShape(const Shape& shape) {
  if (shape.IsArray()) {
    return shape;
  }
  CHECK(shape.IsTuple());
  return GetFirstArrayShape(shape.tuple_shapes(0));
}

absl::Status TestBijection(const IndexingMap& map,
                           absl::Span<int64_t const> shape) {
  std::vector<Interval> intervals;
  for (int64_t size : shape) {
    intervals.push_back({0, size - 1});
  }
  auto status = VerifyBijection(map, intervals);
  if (status.ok()) return status;
  return absl::FailedPreconditionError(
      absl::StrCat(status.message(), " in map ", ToString(map)));
}

TEST_F(CorrectnessTest, RunAndCompare) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadTestModule(flags.input_file));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module),
      ErrorSpec{flags.abs_error_bound, flags.rel_error_bound}));
}

absl::StatusOr<int64_t> GetHeroIndex(absl::string_view name,
                                     const HloFusionAnalysis& analysis) {
  for (auto [index, hero] : llvm::enumerate(analysis.fusion_heroes())) {
    if (hero.name() == name) {
      return index;
    }
  }
  return absl::NotFoundError(absl::StrCat("Hero ", name, " not found"));
}

std::pair<std::string, std::vector<int64_t>> ParseHeroAndIds(
    absl::string_view hero_and_ids) {
  std::pair<absl::string_view, absl::string_view> hero_and_ids_pair =
      absl::StrSplit(hero_and_ids, ':');
  std::vector<int64_t> ids;
  for (absl::string_view id : absl::StrSplit(hero_and_ids_pair.second, ',')) {
    ids.push_back(std::stoi(std::string(absl::StripAsciiWhitespace(id))));
  }
  return {std::string(absl::StripAsciiWhitespace(hero_and_ids_pair.first)),
          ids};
}

TEST_F(CorrectnessTest, InputIndexingIsBijection) {
  auto context = GetMlirContextForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadTestModule(flags.input_file));
  TF_ASSERT_OK_AND_ASSIGN(auto emitter_data, GetMlirFusionEmitter(*module));
  for (const auto& [hero_name, ids] : flags.bijection_inputs) {
    TF_ASSERT_OK_AND_ASSIGN(int64_t hero_index,
                            GetHeroIndex(hero_name, *emitter_data->analysis));
    for (int64_t id : ids) {
      auto indexing = emitter_data->emitter->ComputeThreadIdToInputIndexing(
          hero_index, id, &context);
      ASSERT_TRUE(indexing.has_value());
      TF_ASSERT_OK(TestBijection(*indexing,
                                 emitter_data->analysis->fusion_hero(hero_index)
                                     .GetOperand(id)
                                     .shape()
                                     .dimensions()))
          << "Expected operand " << id << " of " << hero_name << " (root index "
          << hero_index << ") to be read exactly once.";
    }
  }
}

TEST_F(CorrectnessTest, OutputIndexingIsBijection) {
  auto context = GetMlirContextForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadTestModule(flags.input_file));
  TF_ASSERT_OK_AND_ASSIGN(auto emitter_data, GetMlirFusionEmitter(*module));
  for (const auto& hero_name : flags.bijection_outputs) {
    TF_ASSERT_OK_AND_ASSIGN(int64_t hero_index,
                            GetHeroIndex(hero_name, *emitter_data->analysis));
    auto indexing = emitter_data->emitter->ComputeThreadIdToOutputIndexing(
        hero_index, &context);
    ASSERT_TRUE(indexing.has_value());
    TF_ASSERT_OK(TestBijection(
        *indexing, GetFirstArrayShape(
                       emitter_data->analysis->fusion_root(hero_index).shape())
                       .dimensions()))
        << "Expected output of " << hero_name << " (root index " << hero_index
        << ") to be written exactly once.";
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char* argv[]) {
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("abs_error_bound", &flags.abs_error_bound,
                "Absolute error bound."),
      tsl::Flag("rel_error_bound", &flags.rel_error_bound,
                "Relative error bound."),
      tsl::Flag(
          "bijection_inputs",
          [](std::string name_and_ids) {
            if (name_and_ids.empty()) return false;
            flags.bijection_inputs.push_back(
                xla::gpu::ParseHeroAndIds(name_and_ids));
            return true;
          },
          "",
          "The name of a hero followed by operand ids that should be read "
          "exactly once, i.e. there's a bijection between a subset of threads "
          "and the input shape. Example: 'reduction0: 0, 1'."),
      tsl::Flag(
          "bijection_outputs",
          [](std::string name) {
            if (name.empty()) return false;
            flags.bijection_outputs.push_back(name);
            return true;
          },
          "",
          "The name of a hero whose outputs should be written exactly once, "
          "i.e. there's a bijection between a subset of threads and the output "
          "shape.")};

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
