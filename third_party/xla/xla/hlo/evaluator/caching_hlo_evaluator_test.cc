/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/evaluator/caching_hlo_evaluator.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator_interface.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Eq;
using ::testing::Return;

class MockHloEvaluator : public HloEvaluatorInterface {
 public:
  MOCK_METHOD(absl::StatusOr<Literal>, Evaluate,
              (const HloComputation&, absl::Span<const Literal* const>),
              (override));

  MOCK_METHOD(void, ResetVisitStates, (), (override));

  MOCK_METHOD(void, set_dynamic_dimension_inference,
              (DynamicDimensionInference*), (override));

  MOCK_METHOD(void, set_use_fast_path, (bool), (override));

  MOCK_METHOD(void, set_custom_call_handler, (CustomCallHandler), (override));
};

class CachingHloEvaluatorTest : public ::testing::Test {
 public:
  void SetUp() override {
    ASSERT_OK(tsl::Env::Default()->CreateDir(cache_dir_));
  }
  void TearDown() override {
    int64_t num_files_deleted = 0;
    int64_t num_dirs_deleted = 0;
    ASSERT_OK(tsl::Env::Default()->DeleteRecursively(
        cache_dir_, &num_files_deleted, &num_dirs_deleted));
  }

  std::unique_ptr<CachingHloEvaluator> CreateEvaluator(
      CachingHloEvaluator::Mode mode, MockHloEvaluator*& wrapped) {
    auto mock = std::make_unique<::testing::StrictMock<MockHloEvaluator>>();
    wrapped = mock.get();
    return std::make_unique<CachingHloEvaluator>(std::move(mock), cache_dir_,
                                                 mode);
  }

  // Builds a fake HloComputation that produces a single zero F32 constant val.
  static std::unique_ptr<HloComputation> CreateFakeComputation() {
    HloComputation::Builder builder("test");
    HloInstruction* root = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(0.0f)));
    return builder.Build(root);
  }

  absl::StatusOr<int64_t> ChildCount() const {
    std::vector<std::string> children;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->GetChildren(cache_dir_, &children));
    return children.size();
  }

  absl::string_view cache_dir() const { return cache_dir_; }

 private:
  const std::string cache_dir_ =
      tsl::io::JoinPath(::testing::TempDir(), "cache_dir");
};

TEST_F(CachingHloEvaluatorTest, WriteToCacheRepeatedly) {
  MockHloEvaluator* wrapped = nullptr;
  std::unique_ptr<CachingHloEvaluator> evaluator =
      CreateEvaluator(CachingHloEvaluator::kWrite, wrapped);
  EXPECT_CALL(*wrapped, Evaluate(_, _))
      .WillRepeatedly([]() -> absl::StatusOr<Literal> {
        return LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
      });

  const std::unique_ptr<HloComputation> computation = CreateFakeComputation();
  const Literal arg0 =
      LiteralUtil::CreateR0<bfloat16>(static_cast<bfloat16>(100.0f));
  const Literal arg1 =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});

  // For each unique invocation, we expect a new file to be written.
  ASSERT_THAT(ChildCount(), IsOkAndHolds(0));
  ASSERT_OK(evaluator->Evaluate(*computation, {}));
  ASSERT_THAT(ChildCount(), IsOkAndHolds(1));
  ASSERT_OK(evaluator->Evaluate(*computation, {&arg0}));
  ASSERT_THAT(ChildCount(), IsOkAndHolds(2));
  ASSERT_OK(evaluator->Evaluate(*computation, {&arg1}));
  ASSERT_THAT(ChildCount(), IsOkAndHolds(3));
  ASSERT_OK(evaluator->Evaluate(*computation, {&arg0, &arg1}));
  ASSERT_THAT(ChildCount(), IsOkAndHolds(4));

  // Repeated invocations do not affect the cache.
  ASSERT_OK(evaluator->Evaluate(*computation, {&arg1}));
  ASSERT_THAT(ChildCount(), IsOkAndHolds(4));
}

TEST_F(CachingHloEvaluatorTest, ReadFromCache) {
  const std::unique_ptr<HloComputation> computation = CreateFakeComputation();
  const Literal arg =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  const Literal result = LiteralUtil::CreateR0(1337.0f);
  MockHloEvaluator* wrapped = nullptr;

  // Write to the cache first so we can read from it.
  std::unique_ptr<CachingHloEvaluator> write_evaluator =
      CreateEvaluator(CachingHloEvaluator::kWrite, wrapped);
  EXPECT_CALL(*wrapped, Evaluate(_, _))
      .WillOnce(Return(LiteralUtil::CreateR0(1337.0f)));
  ASSERT_THAT(write_evaluator->Evaluate(*computation, {&arg}),
              IsOkAndHolds(Eq(std::ref(result))));

  // Read from the cache.
  std::unique_ptr<CachingHloEvaluator> read_evaluator =
      CreateEvaluator(CachingHloEvaluator::kRead, wrapped);
  // The wrapped evaluator should never be called.
  EXPECT_CALL(*wrapped, Evaluate(_, _)).Times(0);
  ASSERT_THAT(read_evaluator->Evaluate(*computation, {&arg}),
              IsOkAndHolds(Eq(std::ref(result))));

  // Cache miss.
  ASSERT_THAT(read_evaluator->Evaluate(*computation, {}),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(CachingHloEvaluatorTest, ReadAndEvaluateIfCacheMiss) {
  MockHloEvaluator* wrapped = nullptr;
  std::unique_ptr<CachingHloEvaluator> evaluator = CreateEvaluator(
      CachingHloEvaluator::kReadAndEvaluateIfCacheMiss, wrapped);
  const Literal result = LiteralUtil::CreateR0(1337.0f);

  // Cache miss, but still returns the correct result because of the fallback.
  const std::unique_ptr<HloComputation> computation = CreateFakeComputation();
  EXPECT_CALL(*wrapped, Evaluate(_, _))
      .WillOnce(Return(LiteralUtil::CreateR0(1337.0f)));
  ASSERT_THAT(evaluator->Evaluate(*computation, {}),
              IsOkAndHolds(Eq(std::ref(result))));
}

}  // namespace
}  // namespace xla
