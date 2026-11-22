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

#include "xla/hlo/pass/hlo_pipeline_builder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/pass/hlo_pipeline_editor.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compilation_stats.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::SizeIs;

// Names of the passes in the order they ran.
struct RunLog {
  std::vector<std::string> runs;
};

// Logs every run and reports a change for the first `changes` runs.
class LoggingPass : public HloModulePass {
 public:
  LoggingPass(absl::string_view name, RunLog& log, int64_t changes)
      : name_(name), log_(log), changes_(changes) {}

  absl::string_view name() const override { return name_; }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    log_.runs.push_back(name_);
    return changes_-- > 0;
  }

 private:
  std::string name_;
  RunLog& log_;
  int64_t changes_;
};

class LoggingPassFactory : public HloPassFactory {
 public:
  explicit LoggingPassFactory(absl::string_view name) : name_(name) {}

  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext& context,
      const HloPassOptions& options) const override {
    ABSL_ASSIGN_OR_RETURN(RunLog & log, context.Get<RunLog>());
    return std::make_unique<LoggingPass>(
        name_, log, options.Get<int64_t>("changes").value_or(0));
  }

 private:
  std::string name_;
};

class HloPipelineBuilderTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    ASSERT_OK(registry_.Register(std::make_unique<LoggingPassFactory>("a")));
    ASSERT_OK(registry_.Register(std::make_unique<LoggingPassFactory>("b")));
    ASSERT_OK(context_.Set(log_));
  }

  absl::StatusOr<std::unique_ptr<HloPassPipeline>> Build() {
    return BuildHloPassPipeline(proto_, "root", context_, stats_.get(),
                                registry_);
  }

  // Builds and runs `proto_`, returning the names of the passes that ran.
  absl::StatusOr<std::vector<std::string>> BuildAndRun() {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(R"hlo(
      HloModule m
      ENTRY e {
        ROOT p = f32[] parameter(0)
      })hlo"));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloPassPipeline> pipeline, Build());
    ABSL_RETURN_IF_ERROR(pipeline->Run(module.get()).status());
    return log_.runs;
  }

  HloPassRegistry registry_;
  std::unique_ptr<CompilationStats> stats_ = CompilationStats::MakeStats();
  RunLog log_;
  HloPassContext context_;
  HloPipelineProto proto_;
  HloPipelineEditor root_{proto_};
};

TEST_F(HloPipelineBuilderTest, BuildsNestedStructure) {
  root_.AppendPass("a");
  root_.AppendSequence("inner").AppendPass("b");
  root_.AppendPass("a", {}, /*max_fixed_point_iterations=*/3);
  root_.AppendSequence("simplification", /*max_fixed_point_iterations=*/3)
      .AppendPass("b");

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloPassPipeline> pipeline, Build());

  // (name, IsPassPipeline) of each top-level pass.
  std::vector<std::pair<std::string, bool>> passes;
  for (int i = 0; i < pipeline->PassesSize(); ++i) {
    const HloPassInterface& pass = pipeline->GetPass(i);
    passes.emplace_back(pass.name(), pass.IsPassPipeline());
  }
  EXPECT_EQ(pipeline->name(), "root");
  // A fixed-point node keeps the name and kind of the node it reruns.
  EXPECT_THAT(passes,
              ElementsAre(Pair("a", false), Pair("inner", true),
                          Pair("a", false), Pair("simplification", true)));
}

TEST_F(HloPipelineBuilderTest, RunsNodesInOrder) {
  root_.AppendPass("b");
  HloPipelineEditor inner = root_.AppendSequence("inner");
  inner.AppendPass("a");
  inner.AppendPass("b");
  root_.AppendPass("a");

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(ElementsAre("b", "a", "b", "a")));
}

TEST_F(HloPipelineBuilderTest, SingleIterationRunsOnce) {
  root_.AppendPass("a", {{"changes", 100}}, /*max_fixed_point_iterations=*/1);

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(ElementsAre("a")));
}

TEST_F(HloPipelineBuilderTest, FixedPointRerunsPassUntilNoChange) {
  root_.AppendPass("a", {{"changes", 2}}, /*max_fixed_point_iterations=*/10);

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(ElementsAre("a", "a", "a")));
}

TEST_F(HloPipelineBuilderTest, FixedPointStopsAtMaxIterations) {
  root_.AppendPass("a", {{"changes", 100}}, /*max_fixed_point_iterations=*/3);

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(SizeIs(3)));
}

TEST_F(HloPipelineBuilderTest, FixedPointRerunsWholeSequence) {
  HloPipelineEditor simplification =
      root_.AppendSequence("simplification", /*max_fixed_point_iterations=*/10);
  simplification.AppendPass("a", {{"changes", 1}});
  simplification.AppendPass("b");

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(ElementsAre("a", "b", "a", "b")));
}

TEST_F(HloPipelineBuilderTest, NestedFixedPointRerunsInnerFixedPoint) {
  root_.AppendSequence("outer", /*max_fixed_point_iterations=*/10)
      .AppendPass("a", {{"changes", 1}}, /*max_fixed_point_iterations=*/10);

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(ElementsAre("a", "a", "a")));
}

TEST_F(HloPipelineBuilderTest, NestedPipelinesShareCompilationStats) {
  root_.AppendPass("a");
  root_.AppendSequence("simplification", /*max_fixed_point_iterations=*/10)
      .AppendPass("b", {{"changes", 1}});

  EXPECT_THAT(BuildAndRun(), IsOkAndHolds(ElementsAre("a", "b", "b")));
  EXPECT_EQ(stats_->GetPassesSize(), 3);
}

TEST_F(HloPipelineBuilderTest, UnknownPassFailsWithPipelinePath) {
  root_.AppendSequence("inner").AppendPass("unknown");

  EXPECT_THAT(Build(), StatusIs(absl::StatusCode::kNotFound,
                                HasSubstr("pipeline 'root': pipeline 'inner': "
                                          "HloPassRegistry has no pass "
                                          "'unknown'")));
}

TEST_F(HloPipelineBuilderTest, UnconsumedOptionFails) {
  root_.AppendPass("a", {{"misspelled", 1}});

  EXPECT_THAT(Build(), StatusIs(absl::StatusCode::kInvalidArgument,
                                HasSubstr("misspelled")));
}

TEST_F(HloPipelineBuilderTest, MissingContextObjectFails) {
  root_.AppendPass("a");

  EXPECT_THAT(BuildHloPassPipeline(proto_, "root", HloPassContext(),
                                   /*compilation_stats=*/nullptr, registry_),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("HloPassContext has no")));
}

TEST_F(HloPipelineBuilderTest, SequenceWithOptionsFails) {
  HloPipelineNodeProto& node = *proto_.mutable_root()->add_nodes();
  node.set_name("inner");
  node.mutable_sub_nodes();
  (*node.mutable_options())["changes"].set_int_val(1);

  EXPECT_THAT(Build(), StatusIs(absl::StatusCode::kInvalidArgument,
                                HasSubstr("'inner' has options")));
}

}  // namespace
}  // namespace xla
