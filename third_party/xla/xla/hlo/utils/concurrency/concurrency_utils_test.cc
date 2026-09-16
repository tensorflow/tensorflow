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

#include "xla/hlo/utils/concurrency/concurrency_utils.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla::concurrency {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;

template <typename T>
struct WrappedT {
  T val;
};

TEST(ForEachTest, IterVariantConcurrentlyIncrementsIntegers) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 5);

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  ASSERT_OK(ForEach(
      v.begin(), v.end(),
      [](int* element) {
        ++(*element);
        return absl::OkStatus();
      },
      *pool.AsExecutor()));

  EXPECT_EQ(v0, kx0 + 1);
  EXPECT_EQ(v1, kx1 + 1);
  EXPECT_EQ(v2, kx2 + 1);
}

TEST(ForEachTest, NonOkStatusPropagatesAsTheFinalResult) {
  const absl::Status status = absl::CancelledError("Test Error");

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 3);

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  EXPECT_THAT(
      ForEach(
          v.begin(), v.end(), [&status](int* element) { return status; },
          *pool.AsExecutor()),
      StatusIs(absl::StatusCode::kCancelled));
}

TEST(ForEachTest, ActionReturnedValuesCollected) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 3);

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      (ForEach<int>(
          v.begin(), v.end(),
          [](int* element) -> absl::StatusOr<int> { return ++(*element); },
          *pool.AsExecutor())));

  EXPECT_EQ(v0, kx0 + 1);
  EXPECT_EQ(v1, kx1 + 1);
  EXPECT_EQ(v2, kx2 + 1);

  EXPECT_THAT(result, ElementsAreArray({1, 2, 3}));
}

TEST(ForEachTest, FailureOfTheFirstActionPropagates) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 3);

  constexpr int kx0 = 0;
  constexpr int kx1 = 1;
  constexpr int kx2 = 2;

  int v0 = kx0;
  int v1 = kx1;
  int v2 = kx2;

  std::vector<int*> v = {&v0, &v1, &v2};

  EXPECT_THAT(ForEach<int>(
                  v.begin(), v.end(),
                  [](int* element) -> absl::StatusOr<int> {
                    if (*element % 2 == 1)
                      return absl::CancelledError("Force a failure.");
                    return ++(*element);
                  },
                  *pool.AsExecutor())
                  .status(),
              StatusIs(absl::StatusCode::kCancelled));
}

class HloComputationTest : public HloHardwareIndependentTestBase {
 protected:
  HloComputationTest() = default;

  // Create a computation which takes a scalar and returns its negation.
  std::unique_ptr<HloComputation> CreateNegateComputation(
      absl::string_view name = "Negate") {
    auto builder = HloComputation::Builder(name);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, r0f32_, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, param));
    return builder.Build();
  }

  std::unique_ptr<HloModule> CreateNegateModule() {
    auto module =
        std::make_unique<HloModule>("NegateModule", HloModuleConfig{});
    module->AddComputation(CreateNegateComputation("Negate0"), true);
    module->AddComputation(CreateNegateComputation("Negate1"), false);
    module->AddComputation(CreateNegateComputation("Negate2"), false);

    return module;
  };

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});

  tsl::thread::ThreadPool pool_{tsl::Env::Default(), "test", 5};
};

TEST_F(HloComputationTest, ForEachHloComputationBasicCall) {
  auto comp0 = CreateNegateComputation();
  auto comp1 = CreateNegateComputation();
  auto comp2 = CreateNegateComputation();

  std::vector<HloComputation*> v = {comp0.get(), comp1.get(), comp2.get()};

  auto result = ForEachHloComputation<bool, WrappedT<bool>>(
      v.begin(), v.end(),
      [](HloComputation* comp) -> absl::StatusOr<WrappedT<bool>> {
        return WrappedT<bool>{true};
      },
      [](std::vector<WrappedT<bool>>& results) {
        return std::any_of(results.begin(), results.end(),
                           [](WrappedT<bool> b) { return b.val; });
      },
      *pool_.AsExecutor());
  // For compatibility with OpenXLA.
  ASSERT_OK(result.status());
  EXPECT_EQ(*result, true);
}

TEST_F(HloComputationTest, ForEachHloComputationSpanBasicCall) {
  auto comp0 = CreateNegateComputation();
  auto comp1 = CreateNegateComputation();
  auto comp2 = CreateNegateComputation();

  std::vector<HloComputation*> v = {comp0.get(), comp1.get(), comp2.get()};

  auto result = ForEachHloComputation<bool, WrappedT<bool>>(
      v,
      [](HloComputation* comp) -> absl::StatusOr<WrappedT<bool>> {
        return WrappedT<bool>{true};
      },
      [](std::vector<WrappedT<bool>>& results) {
        return std::any_of(results.begin(), results.end(),
                           [](WrappedT<bool> b) { return b.val; });
      },
      *pool_.AsExecutor());
  // For compatibility with OpenXLA.
  ASSERT_OK(result.status());
  EXPECT_EQ(*result, true);
}

TEST_F(HloComputationTest, ForEachHloComputationModuleBasicCall) {
  auto module = CreateNegateModule();

  auto result = ForEachHloComputation<int, WrappedT<bool>>(
      module.get(),
      [](HloComputation* comp) -> absl::StatusOr<WrappedT<bool>> {
        return WrappedT<bool>{true};
      },
      [](std::vector<WrappedT<bool>>& results) { return results.size(); },
      *pool_.AsExecutor());
  // For compatibility with OpenXLA.
  ASSERT_OK(result.status());
  EXPECT_EQ(*result, 3);
}

TEST_F(HloComputationTest, ForEachNonfusionHloComputationModuleBasicCall) {
  auto module = CreateNegateModule();

  auto result = ForEachNonfusionHloComputation<int, WrappedT<bool>>(
      module.get(), {},
      [](HloComputation* comp) -> absl::StatusOr<WrappedT<bool>> {
        return WrappedT<bool>{true};
      },
      [](std::vector<WrappedT<bool>>& results) { return results.size(); },
      *pool_.AsExecutor());
  // For compatibility with OpenXLA.
  ASSERT_OK(result.status());
  EXPECT_EQ(*result, 3);
}

}  // namespace
}  // namespace xla::concurrency

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

void BM_ForEachIterateCost(::testing::benchmark::State& state) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 5);
  int num_items = state.range(0);
  absl::flat_hash_set<int> values;
  values.reserve(num_items);
  for (int i = 0; i < num_items; ++i) {
    values.insert(i);
  }

  for (auto _ : state) {
    absl::Status status = xla::concurrency::ForEach(
        values.begin(), values.end(),
        [](int element) {
          benchmark::DoNotOptimize(element);
          return absl::OkStatus();
        },
        *pool.AsExecutor());
    CHECK_OK(status);
  }
}

BENCHMARK(BM_ForEachIterateCost)->Arg(10000)->Arg(100000);

GTEST_API_ int main(int argc, char** argv) {
  // Only run benchmarks if `--benchmark_filter` is set.
  for (int i = 1; i < argc; ++i) {
    if (absl::StartsWith(argv[i], "--benchmark_filter=")) {
      ::benchmark::Initialize(&argc, argv);
      ::testing::InitGoogleTest(&argc, argv);
      ::benchmark::RunSpecifiedBenchmarks();
      return 0;
    }
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
