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

#include "xla/hlo/tools/comparison/comparison_tool.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_options.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tools/debug_event.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::IsEmpty;
using ::testing::IsNan;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::Partially;
using ::xla::LogData;
using ::xla::LogHloOutputMetadata;
using ::xla::OriginalValueProto;

// Mock for ComparisonTool to test its protected/virtual methods
class MockComparisonTool : public ComparisonTool {
 public:
  explicit MockComparisonTool(const ComparisonOptions& options,
                              tsl::thread::ThreadPool* async_queue = nullptr)
      : ComparisonTool(options, async_queue) {}

  // Make CreateTensorSummary public for testing.
  using ComparisonTool::CreateTensorSummary;

  MOCK_METHOD(absl::Status, RegisterOriginalHloModuleImpl,
              (const HloModuleProto& module), (override));
  MOCK_METHOD(absl::Status, RegisterRun,
              (int32_t logical_device_id, uint64_t run_id,
               absl::string_view hlo_module_name),
              (override));
  MOCK_METHOD(absl::Status, FinishRun,
              (int32_t logical_device_id, uint64_t run_id,
               absl::string_view hlo_module_name),
              (override));
  MOCK_METHOD(absl::Status, ProcessTensorSummary,
              (absl::string_view hlo_module_name, const TensorSummary& summary),
              (override));
};

LogData CreateLogData(absl::string_view module_name,
                      absl::string_view instruction_name,
                      bool with_original_value = true) {
  LogData log_record;
  LogHloOutputMetadata* hlo_meta = log_record.mutable_hlo_output_metadata();
  hlo_meta->set_module_name(module_name);
  hlo_meta->set_instruction_name(instruction_name);

  if (with_original_value) {
    OriginalValueProto* original_value = hlo_meta->mutable_original_value();
    OriginalValueElementProto* leaf = original_value->add_elements();
    leaf->mutable_original_array()->set_instruction_name("original_op_name");
  }
  return log_record;
}

class ComparisonToolTest : public ::testing::Test {
 public:
  void SetUp() override {
    pool_ = std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(),
                                                      "test_pool", 1);
  }
  void WaitUntilComplete() {
    pool_.reset();
    pool_ = std::make_unique<tsl::thread::ThreadPool>(tsl::Env::Default(),
                                                      "test_pool", 1);
  }
  std::unique_ptr<tsl::thread::ThreadPool> pool_;
  std::unique_ptr<MockComparisonTool> mock_tool_;
};

// Tests for CreateTensorSummary
TEST_F(ComparisonToolTest, CreateTensorSummaryEmptyLiteral) {
  LogData log_record = CreateLogData("test_module", "empty_op");
  Literal literal = LiteralUtil::CreateR1<float>({});
  ComparisonOptions options;
  options.set_comparison_variant(
      ComparisonOptions::COMPARISON_VARIANT_BASELINE);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);

  EXPECT_THAT(summary.metadata().comparison_variant(),
              Eq(ComparisonOptions::COMPARISON_VARIANT_BASELINE));
  EXPECT_THAT(summary.metadata().hlo_module_name(), StrEq("test_module"));
  EXPECT_THAT(summary.shape().dimensions_size(), Eq(1));
  EXPECT_THAT(summary.shape().dimensions(0), Eq(0));
  EXPECT_THAT(summary.mean(), IsNan());
  EXPECT_THAT(summary.min(), IsNan());
  EXPECT_THAT(summary.max(), IsNan());
  EXPECT_THAT(summary.stddev(), IsNan());
  EXPECT_THAT(summary.samples(), IsEmpty());
  EXPECT_THAT(summary.checksum(), StrEq(std::string(8, '\0')));
}

TEST_F(ComparisonToolTest, CreateTensorSummaryAllZeros) {
  ComparisonOptions options;
  options.set_min_sample_count(5);  // Should sample all 4 elements
  options.set_max_sample_count(5);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "zeros_op");
  Literal literal = LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);

  EXPECT_THAT(summary.mean(), FloatEq(0.0f));
  EXPECT_THAT(summary.min(), FloatEq(0.0f));
  EXPECT_THAT(summary.max(), FloatEq(0.0f));
  EXPECT_THAT(summary.stddev(), FloatEq(0.0f));
  EXPECT_THAT(summary.non_zero_mean(), IsNan());
  EXPECT_THAT(summary.non_zero_stddev(), IsNan());
  EXPECT_THAT(summary.checksum(), StrEq("\xCB"
                                        "B\xA3\xAF\xC0\x94"
                                        "4\xF6"));
  EXPECT_THAT(summary.samples(), ElementsAre(FloatEq(0.0f), FloatEq(0.0f),
                                             FloatEq(0.0f), FloatEq(0.0f)));
}

TEST_F(ComparisonToolTest, CreateTensorSummarySimpleFloatsAndSampling) {
  ComparisonOptions options;
  options.set_min_sample_count(3);
  options.set_max_sample_count(16);
  options.set_sample_ratio(0.5f);
  options.set_sample_seed(12345);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "floats_op");
  Literal literal = LiteralUtil::CreateR1<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f});

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);

  EXPECT_THAT(summary.mean(), FloatEq(10.0f));
  EXPECT_THAT(summary.min(), FloatEq(1.0f));
  EXPECT_THAT(summary.max(), FloatEq(19.0f));
  EXPECT_THAT(summary.stddev(), FloatEq(5.4772258f));
  EXPECT_THAT(summary.non_zero_mean(), FloatEq(10.0f));
  EXPECT_THAT(summary.non_zero_stddev(), FloatEq(5.4772258f));

  // Checksum and samples are deterministic with seed.
  EXPECT_THAT(summary.checksum(), Not(IsEmpty()));
  EXPECT_THAT(summary.checksum(), StrEq("\xDD\xEE\aF\x1B?%\xE6"));
  EXPECT_THAT(summary.samples(), SizeIs(9));
  EXPECT_THAT(summary.samples(), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 7.0f, 8.0f,
                                             10.0f, 13.0f, 18.0f));
}

TEST_F(ComparisonToolTest, CreateTensorSummaryMoreSamples) {
  ComparisonOptions options;
  options.set_min_sample_count(100);
  options.set_max_sample_count(1000);
  options.set_sample_ratio(0.25f);
  options.set_sample_seed(12345);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "floats_op");
  std::vector<float> numbers(2000);
  std::iota(numbers.begin(), numbers.end(), 0);
  Literal literal = LiteralUtil::CreateR1<float>(numbers);

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);

  EXPECT_THAT(summary.samples(), SizeIs(500));

  // Assert that all elements are distinct
  absl::flat_hash_set<float> unique_samples(summary.samples().begin(),
                                            summary.samples().end());
  EXPECT_THAT(summary.samples(), SizeIs(unique_samples.size()));
}

TEST_F(ComparisonToolTest,
       CreateTensorSummarySamplingAllElementsDueToMinCount) {
  ComparisonOptions options;
  options.set_min_sample_count(5);  // min_sample_count > num_elements
  options.set_max_sample_count(5);
  options.set_sample_ratio(0.1f);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "sample_all_min");
  Literal literal = LiteralUtil::CreateR1<int32_t>({10, 20, 30});

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);
  EXPECT_THAT(summary.samples(),
              ElementsAre(FloatEq(10.0f), FloatEq(20.0f), FloatEq(30.0f)));
}

TEST_F(ComparisonToolTest, CreateTensorSummarySamplingMaxCount) {
  ComparisonOptions options;
  options.set_min_sample_count(1);
  options.set_max_sample_count(5);
  // Try to sample all, but max_sample_count limits it
  options.set_sample_ratio(1.0f);
  options.set_sample_seed(1);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "sample_max");
  std::vector<float> values;
  values.reserve(100);
  for (int i = 0; i < 100; ++i) {
    values.push_back(static_cast<float>(i));
  }
  Literal literal = LiteralUtil::CreateR1<float>(values);

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);
  EXPECT_THAT(summary.samples(), ElementsAre(9.0f, 30.0f, 31.0f, 57.0f, 78.0f));
}

TEST_F(ComparisonToolTest, CreateTensorSummaryWithNaNs) {
  ComparisonOptions options;
  options.set_min_sample_count(3);  // Sample all
  options.set_max_sample_count(3);
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "nans_op");
  Literal literal = LiteralUtil::CreateR1<float>(
      {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f});

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);

  EXPECT_THAT(summary.mean(), FloatEq(2.0f));  // (1+3)/2
  EXPECT_THAT(summary.min(), FloatEq(1.0f));
  EXPECT_THAT(summary.max(), FloatEq(3.0f));
  EXPECT_THAT(summary.stddev(),
              FloatEq(1.0f));  // Var = ((1-2)^2 + (3-2)^2)/2 = 1

  EXPECT_THAT(summary.samples(),
              ElementsAre(FloatEq(1.0f), IsNan(), FloatEq(3.0f)));
}

TEST_F(ComparisonToolTest, CreateTensorSummaryPredType) {
  ComparisonOptions options;
  options.set_min_sample_count(3);  // Sample all
  options.set_max_sample_count(3);  // Sample all
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "pred_op");
  Literal literal = LiteralUtil::CreateR1<bool>({true, false, true});

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);
  EXPECT_THAT(summary.mean(), FloatEq(2.0f / 3.0f));
  EXPECT_THAT(summary.min(), FloatEq(0.0f));
  EXPECT_THAT(summary.max(), FloatEq(1.0f));
  EXPECT_THAT(summary.stddev(), FloatEq(std::sqrt(2.0f / 9.0f)));
  EXPECT_THAT(summary.samples(),
              ElementsAre(FloatEq(1.0f), FloatEq(0.0f), FloatEq(1.0f)));
}

TEST_F(ComparisonToolTest, CreateTensorSummaryWithScopes) {
  mock_tool_ =
      std::make_unique<MockComparisonTool>(ComparisonOptions(), pool_.get());

  LogData log_record = CreateLogData("test_module", "scopes_op");
  LogHloOutputMetadata* hlo_meta = log_record.mutable_hlo_output_metadata();
  auto* scope = hlo_meta->add_scopes();
  scope->mutable_original_value()
      ->add_elements()
      ->mutable_original_array()
      ->set_instruction_name("scope1");

  Literal literal = LiteralUtil::CreateR0<float>(42.0f);

  TensorSummary summary = mock_tool_->CreateTensorSummary(log_record, literal);
  EXPECT_THAT(
      summary.metadata().original_positions(),
      ElementsAre(Partially(EqualsProto(R"pb(instruction_name: "scope1")pb")),
                  Partially(EqualsProto(
                      R"pb(instruction_name: "original_op_name")pb"))));
}

// Tests for RecordTensor
TEST_F(ComparisonToolTest, RecordTensorHloModuleNoMatch) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("no_match_for_this");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "some_op");
  auto literal = std::make_shared<Literal>(LiteralUtil::CreateR0<float>(1.0f));

  EXPECT_CALL(*mock_tool_, ProcessTensorSummary).Times(0);

  mock_tool_->RecordTensor(log_record, literal);
  WaitUntilComplete();
}

TEST_F(ComparisonToolTest, RecordTensorHloModuleMatchButNoOriginalValue) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("test_module");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record =
      CreateLogData("test_module", "some_op", /*with_original_value=*/false);
  auto literal = std::make_shared<Literal>(LiteralUtil::CreateR0<float>(1.0f));

  EXPECT_CALL(*mock_tool_, ProcessTensorSummary).Times(0);

  mock_tool_->RecordTensor(log_record, literal);
  WaitUntilComplete();
}

TEST_F(ComparisonToolTest, RecordTensorSuccessfulAsyncProcessing) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("test_module");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module", "some_op");
  auto literal = std::make_shared<Literal>(LiteralUtil::CreateR0<float>(1.0f));

  EXPECT_CALL(*mock_tool_, ProcessTensorSummary(StrEq("test_module"), _))
      .WillOnce(Return(absl::OkStatus()));

  mock_tool_->RecordTensor(log_record, literal);
  WaitUntilComplete();
}

TEST_F(ComparisonToolTest, RecordTensorAsyncProcessingReturnsError) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("test_module_fail");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  LogData log_record = CreateLogData("test_module_fail", "fail_op");
  auto literal = std::make_shared<Literal>(LiteralUtil::CreateR0<float>(1.0f));

  EXPECT_CALL(*mock_tool_, ProcessTensorSummary(StrEq("test_module_fail"), _))
      .WillOnce(Return(absl::InternalError("Simulated processing error")));

  mock_tool_->RecordTensor(log_record, literal);
  WaitUntilComplete();
}

// Tests for RegisterOriginalHloModule
TEST_F(ComparisonToolTest, RegisterOriginalHloModuleMatchingRegex) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("match_this_module");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  HloModule module("match_this_module", HloModuleConfig{});
  HloModuleProto module_proto = module.ToProto();

  EXPECT_CALL(*mock_tool_,
              RegisterOriginalHloModuleImpl(testing::Ref(module_proto)))
      .WillOnce(Return(absl::OkStatus()));

  ASSERT_OK(mock_tool_->RegisterOriginalHloModule(module_proto));
}

TEST_F(ComparisonToolTest, RegisterOriginalHloModuleNonMatchingRegex) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("specific_module_name");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  HloModule module("another_module_name", HloModuleConfig{});
  HloModuleProto module_proto = module.ToProto();

  EXPECT_CALL(*mock_tool_, RegisterOriginalHloModuleImpl(_)).Times(0);

  ASSERT_OK(mock_tool_->RegisterOriginalHloModule(module_proto));
}

TEST_F(ComparisonToolTest, RegisterOriginalHloModuleCalledDuplicateCalls) {
  ComparisonOptions options;
  options.set_hlo_module_name_regex("match_this_module");
  mock_tool_ = std::make_unique<MockComparisonTool>(options, pool_.get());

  HloModule module("match_this_module", HloModuleConfig{});
  HloModuleProto module_proto = module.ToProto();

  EXPECT_CALL(*mock_tool_,
              RegisterOriginalHloModuleImpl(testing::Ref(module_proto)))
      .WillOnce(Return(absl::OkStatus()));

  ASSERT_OK(mock_tool_->RegisterOriginalHloModule(module_proto));
  ASSERT_OK(mock_tool_->RegisterOriginalHloModule(module_proto));
}

}  // namespace
}  // namespace xla::numerics::comparison
