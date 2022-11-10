/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"

#include <map>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/test.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

class TuplePointsToAnalysisTest : public HloTestBase {
 protected:
  // Builds a module with the given entry computation and runs points to
  // analysis.
  void BuildModuleAndRunAnalysis(std::unique_ptr<HloComputation> computation) {
    BuildModule(std::move(computation));
    RunAnalysis();
  }

  void BuildModule(std::unique_ptr<HloComputation> computation) {
    module_ = CreateNewVerifiedModule();
    module_->AddEntryComputation(std::move(computation));
  }

  void RunAnalysis() {
    CHECK_NOTNULL(module_.get());
    points_to_analysis_ = TuplePointsToAnalysis::Run(module_.get()).value();
  }

  // Returns the LogicalBuffer defined at the given instruction and
  // index. CHECKs if no buffer is defined at that point.
  const LogicalBuffer* const GetBuffer(const HloInstruction* instruction,
                                       const ShapeIndex& index) {
    const auto& pointed_to =
        points_to_analysis_->GetPointsToSet(instruction).element(index);
    CHECK_EQ(1, pointed_to.size());
    CHECK_EQ(instruction, pointed_to[0]->instruction());
    CHECK(index == pointed_to[0]->index());
    return pointed_to[0];
  }

  // Checks that the given points-to set contains exactly (unordered) the given
  // LogicalBuffers.
  void ExpectHasBuffers(const PointsToSet::BufferList& points_to_set,
                        absl::Span<const LogicalBuffer* const> buffers) {
    std::vector<const LogicalBuffer*> vec(buffers.begin(), buffers.end());
    EXPECT_THAT(points_to_set, UnorderedElementsAreArray(vec));
  }

  // Checks that the given points-to set contains exactly (unordered) the
  // top-level buffers of the given instructions.
  void ExpectHasTopLevelBuffers(
      const PointsToSet::BufferList& points_to_set,
      absl::Span<HloInstruction* const> instructions) {
    PointsToSet::BufferList buffers;
    for (auto instruction : instructions) {
      buffers.push_back(GetBuffer(instruction, /*index=*/{}));
    }
    ExpectHasBuffers(points_to_set, buffers);
  }

  // Overload which takes a set instead of a vector.
  void ExpectHasTopLevelBuffers(
      const PointsToSet::BufferSet& points_to_set,
      absl::Span<HloInstruction* const> instructions) {
    ExpectHasTopLevelBuffers(
        PointsToSet::BufferList(points_to_set.begin(), points_to_set.end()),
        instructions);
  }

  // Checks that the buffer defined at the given instruction and index has
  // aliases which are exactly (unordered) the given instruction/index pairs.
  void ExpectHasBufferAliases(
      const HloInstruction* instruction, const ShapeIndex& index,
      absl::Span<const std::pair<HloInstruction*, ShapeIndex>> expected) {
    const LogicalBuffer* buffer =
        points_to_analysis_->GetBufferDefinedAt(instruction, index).value();
    std::vector<BufferAlias> expected_aliases;
    expected_aliases.reserve(expected.size());
    for (auto& pair : expected) {
      expected_aliases.push_back(BufferAlias(pair.first, pair.second));
    }
    EXPECT_THAT(points_to_analysis_->GetBufferAliases(*buffer),
                UnorderedElementsAreArray(expected_aliases));
  }

  std::unique_ptr<HloModule> module_;
  std::unique_ptr<TuplePointsToAnalysis> points_to_analysis_;
};

}  // namespace
}  // namespace xla
