/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/split_provider.h"

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

std::vector<int64_t> GetCardinalities(
    const std::vector<std::unique_ptr<SplitProvider>>& split_providers) {
  std::vector<int64_t> cardinalities;
  for (const auto& split_provider : split_providers) {
    cardinalities.push_back(split_provider->Cardinality());
  }
  return cardinalities;
}

TEST(SplitProviderTest, RangeCardinality) {
  DatasetDef range_dataset = testing::RangeDataset(10);
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(range_dataset, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers), UnorderedElementsAre(10));
}

class RepeatedSplitProviderTest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {
 public:
  int64_t Range() const { return std::get<0>(GetParam()); }
  int64_t RepeatCount() const { return std::get<1>(GetParam()); }
  int64_t ExpectedCardinality() const { return std::get<2>(GetParam()); }
};

// Test cases for the `RepeatedDatasetCardinality` test. The tuples specify
// {range, repeat count, expected cardinality}.
constexpr std::array<std::tuple<int64_t, int64_t, int64_t>, 5>
    kRepeatedSplitProviderTestCases{{{9, 9, 81},
                                     {9, 0, 0},
                                     {9, -1, kInfiniteCardinality},
                                     {0, -1, 0},
                                     {-1, 1, 0}}};

TEST_P(RepeatedSplitProviderTest, RepeatedDatasetCardinality) {
  TF_ASSERT_OK_AND_ASSIGN(
      DatasetDef repeated_dataset,
      testing::GetTestDataset(
          "repeated_dataset",
          {absl::StrCat(Range()), absl::StrCat(RepeatCount())}));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(repeated_dataset, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers),
              ElementsAre(ExpectedCardinality()));
}

INSTANTIATE_TEST_SUITE_P(MyGroup, RepeatedSplitProviderTest,
                         ::testing::ValuesIn(kRepeatedSplitProviderTestCases));

TEST(SplitProviderTest, EnumerateCardinality) {
  TF_ASSERT_OK_AND_ASSIGN(DatasetDef enumerate_dataset,
                          testing::GetTestDataset("enumerate_dataset"));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(enumerate_dataset, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers),
              UnorderedElementsAre(3, kInfiniteCardinality));
}

TEST(SplitProviderTest, ChooseFromDatasetsCardinality) {
  TF_ASSERT_OK_AND_ASSIGN(DatasetDef sample_from_datasets,
                          testing::GetTestDataset("choose_from_datasets"));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(sample_from_datasets, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers),
              UnorderedElementsAre(5, 5, 5, kInfiniteCardinality));
}

TEST(SplitProviderTest, SampleFromDatasetsCardinality) {
  TF_ASSERT_OK_AND_ASSIGN(DatasetDef sample_from_datasets,
                          testing::GetTestDataset("sample_from_datasets"));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(sample_from_datasets, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers),
              UnorderedElementsAre(5, 5, 5, kInfiniteCardinality));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
