/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <utility>
#include <vector>

#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {
namespace standalone {
namespace {

using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

class TestDataset {
 public:
  explicit TestDataset(const DatasetDef& dataset_def) {
    TF_CHECK_OK(
        Dataset::FromGraph(Dataset::Params(), dataset_def.graph(), &dataset_));
  }

  absl::StatusOr<std::unique_ptr<Iterator>> MakeIterator() const {
    std::unique_ptr<Iterator> iterator;
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(&iterator));
    return iterator;
  }

 private:
  std::unique_ptr<Dataset> dataset_;
};

template <class T>
StatusOr<T> GetNext(Iterator& iterator) {
  std::vector<Tensor> result;
  bool end_of_sequence = false;
  TF_RETURN_IF_ERROR(iterator.GetNext(&result, &end_of_sequence));
  if (end_of_sequence) {
    return errors::OutOfRange("iterator has reached the end of sequence.");
  }
  if (result.size() != 1) {
    return errors::Internal("GetNext result Tensor size should be 1.");
  }
  return result[0].unaligned_flat<T>().data()[0];
}

TEST(TaskRunnerCheckpointTest, SaveAndRestoreFromCheckpoints) {
  int64_t range = 10;
  TestDataset dataset(testing::RangeDataset(range));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Iterator> iterator,
                          dataset.MakeIterator());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Tensor> saved_iterator, iterator->Save());

  for (int64_t i = 0; i < range; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(iterator, dataset.MakeIterator());
    TF_ASSERT_OK(iterator->Restore(saved_iterator));
    EXPECT_THAT(GetNext<int64_t>(*iterator), absl_testing::IsOkAndHolds(i));
    TF_ASSERT_OK_AND_ASSIGN(saved_iterator, iterator->Save());
  }
}

TEST(TaskRunnerCheckpointTest, EmptyDataset) {
  TestDataset dataset(testing::RangeDataset(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Iterator> iterator,
                          dataset.MakeIterator());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Tensor> saved_iterator, iterator->Save());

  TF_ASSERT_OK_AND_ASSIGN(iterator, dataset.MakeIterator());
  TF_ASSERT_OK(iterator->Restore(saved_iterator));
  EXPECT_THAT(GetNext<int64_t>(*iterator),
              absl_testing::StatusIs(error::OUT_OF_RANGE));
}

TEST(TaskRunnerCheckpointTest, EndOfSequenceIterator) {
  TestDataset dataset(testing::RangeDataset(0));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Iterator> iterator,
                          dataset.MakeIterator());
  EXPECT_THAT(GetNext<int64_t>(*iterator),
              absl_testing::StatusIs(error::OUT_OF_RANGE));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Tensor> saved_iterator, iterator->Save());
  TF_ASSERT_OK_AND_ASSIGN(iterator, dataset.MakeIterator());
  TF_ASSERT_OK(iterator->Restore(saved_iterator));
  EXPECT_THAT(GetNext<int64_t>(*iterator),
              absl_testing::StatusIs(error::OUT_OF_RANGE));
}

}  // namespace
}  // namespace standalone
}  // namespace data
}  // namespace tensorflow
