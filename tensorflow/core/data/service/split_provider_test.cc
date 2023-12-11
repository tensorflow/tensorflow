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

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

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

TEST(SplitProviderTest, EnumerateCardinality) {
  TF_ASSERT_OK_AND_ASSIGN(DatasetDef enumerate_dataset,
                          testing::EnumerateDataset());
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(enumerate_dataset, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers),
              UnorderedElementsAre(3, kInfiniteCardinality));
}

TEST(SplitProviderTest, SampleFromDatasetsCardinality) {
  TF_ASSERT_OK_AND_ASSIGN(DatasetDef sample_from_datasets,
                          testing::SampleFromDatasets());
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_ASSERT_OK(CreateSplitProviders(sample_from_datasets, split_providers));
  EXPECT_THAT(GetCardinalities(split_providers),
              UnorderedElementsAre(5, 5, 5, kInfiniteCardinality));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
