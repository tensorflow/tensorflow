/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/test_util.h"

#include <memory>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace testing {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::testing::IsEmpty;

StatusOr<std::vector<std::vector<Tensor>>> GetIteratorOutput(
    standalone::Iterator& iterator) {
  bool end_of_input = false;
  std::vector<std::vector<Tensor>> result;
  while (!end_of_input) {
    std::vector<tensorflow::Tensor> outputs;
    TF_RETURN_IF_ERROR(iterator.GetNext(&outputs, &end_of_input));
    if (!end_of_input) {
      result.push_back(outputs);
    }
  }
  return result;
}

TEST(TestUtilTest, RangeSquareDataset) {
  const auto dataset_def = RangeSquareDataset(/*range=*/10);
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<Tensor>> result,
                          GetIteratorOutput(*iterator));

  ASSERT_EQ(result.size(), 10);
  for (int i = 0; i < result.size(); ++i) {
    test::ExpectEqual(result[i][0], Tensor(int64{i * i}));
  }
}

TEST(TestUtilTest, EmptyDataset) {
  const auto dataset_def = RangeSquareDataset(/*range=*/0);
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput(*iterator), IsOkAndHolds(IsEmpty()));
}

}  // namespace
}  // namespace testing
}  // namespace data
}  // namespace tensorflow
