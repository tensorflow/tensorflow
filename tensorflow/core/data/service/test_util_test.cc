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

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace testing {
namespace {

using ::tensorflow::testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::IsEmpty;

template <class T>
StatusOr<std::vector<T>> GetIteratorOutput(standalone::Iterator& iterator) {
  std::vector<T> result;
  for (bool end_of_sequence = false; !end_of_sequence;) {
    std::vector<tensorflow::Tensor> tensors;
    TF_RETURN_IF_ERROR(iterator.GetNext(&tensors, &end_of_sequence));
    if (end_of_sequence) {
      break;
    }
    if (tensors.size() != 1) {
      return errors::Internal("GetNext Tensor size is not 1.");
    }
    result.push_back(tensors[0].unaligned_flat<T>().data()[0]);
  }
  return result;
}

TEST(TestUtilTest, RangeDataset) {
  const auto dataset_def = RangeDataset(/*range=*/10);
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput<int64_t>(*iterator),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(TestUtilTest, RangeSquareDataset) {
  const auto dataset_def = RangeSquareDataset(/*range=*/10);
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput<int64_t>(*iterator),
              IsOkAndHolds(ElementsAre(0, 1, 4, 9, 16, 25, 36, 49, 64, 81)));
}

TEST(TestUtilTest, InfiniteDataset) {
  const auto dataset_def = InfiniteDataset();
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));

  // Verifies the first 10 elements.
  for (int64_t i = 0; i < 10; ++i) {
    std::vector<tensorflow::Tensor> outputs;
    bool end_of_sequence;
    TF_ASSERT_OK(iterator->GetNext(&outputs, &end_of_sequence));
    test::ExpectEqual(outputs[0], Tensor(i));
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
  EXPECT_THAT(GetIteratorOutput<int64_t>(*iterator), IsOkAndHolds(IsEmpty()));
}

TEST(TestUtilTest, InterleaveTextline) {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK_AND_ASSIGN(const DatasetDef dataset_def,
                          InterleaveTextlineDataset(filenames, {"0", "1"}));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput<tstring>(*iterator),
              IsOkAndHolds(ElementsAre("0", "1")));
}

TEST(TestUtilTest, InterleaveTextlineWithNewLines) {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK_AND_ASSIGN(
      const DatasetDef dataset_def,
      InterleaveTextlineDataset(filenames, {"0\n2\n4\n6\n8", "1\n3\n5\n7\n9"}));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput<tstring>(*iterator),
              IsOkAndHolds(ElementsAre("0", "1", "2", "3", "4", "5", "6", "7",
                                       "8", "9")));
}

TEST(TestUtilTest, InterleaveTextlineEmptyFiles) {
  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK_AND_ASSIGN(const DatasetDef dataset_def,
                          InterleaveTextlineDataset(filenames, {"", ""}));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput<tstring>(*iterator), IsOkAndHolds(IsEmpty()));
}

TEST(TestUtilTest, GetTestDataset) {
  TF_ASSERT_OK_AND_ASSIGN(const DatasetDef dataset_def,
                          GetTestDataset("choose_from_datasets"));
  standalone::Dataset::Params params;
  std::unique_ptr<standalone::Dataset> dataset;
  TF_ASSERT_OK(
      standalone::Dataset::FromGraph(params, dataset_def.graph(), &dataset));
  std::unique_ptr<standalone::Iterator> iterator;
  TF_ASSERT_OK(dataset->MakeIterator(&iterator));
  EXPECT_THAT(GetIteratorOutput<tstring>(*iterator),
              IsOkAndHolds(ElementsAre("a", "b", "c", "a", "b", "c", "a", "b",
                                       "c", "a", "b", "c", "a", "b", "c")));
}

}  // namespace
}  // namespace testing
}  // namespace data
}  // namespace tensorflow
