/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/initializable_lookup_table.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace lookup {
namespace {

// Yields the given (keys, values) pairs one batch at a time, like the dataset
// iterator used by InitializeTableFromDataset.
class BatchIterator : public InitializableLookupTable::InitTableIterator {
 public:
  explicit BatchIterator(std::vector<std::pair<Tensor, Tensor>> batches)
      : batches_(std::move(batches)) {}

  void Next() override { ++index_; }
  bool Valid() const override { return index_ < batches_.size(); }
  const Tensor& keys() const override { return batches_[index_].first; }
  const Tensor& values() const override { return batches_[index_].second; }
  absl::Status status() const override {
    return Valid() ? absl::OkStatus() : absl::OutOfRangeError("No more data.");
  }
  int64_t total_size() const override { return 0; }

 private:
  std::vector<std::pair<Tensor, Tensor>> batches_;
  size_t index_ = 0;
};

core::RefCountPtr<HashTable<int64_t, int64_t>> MakeTable() {
  return core::RefCountPtr<HashTable<int64_t, int64_t>>(
      new HashTable<int64_t, int64_t>(nullptr, nullptr));
}

TEST(InitializableLookupTableTest, InsertsEveryBatch) {
  auto table = MakeTable();
  BatchIterator iter({
      {test::AsTensor<int64_t>({1}), test::AsTensor<int64_t>({10})},
      {test::AsTensor<int64_t>({2, 3}), test::AsTensor<int64_t>({20, 30})},
  });
  TF_ASSERT_OK(table->Initialize(iter));
  EXPECT_EQ(table->size(), 3u);

  Tensor keys = test::AsTensor<int64_t>({1, 2, 3, 4});
  Tensor values(DT_INT64, keys.shape());
  TF_ASSERT_OK(
      table->Find(nullptr, keys, &values, test::AsScalar<int64_t>(-1)));
  test::ExpectTensorEqual<int64_t>(values,
                                   test::AsTensor<int64_t>({10, 20, 30, -1}));
}

TEST(InitializableLookupTableTest, RejectsFirstBatchWithMismatchedShapes) {
  auto table = MakeTable();
  BatchIterator iter({
      {test::AsTensor<int64_t>({1, 2}), test::AsTensor<int64_t>({10})},
  });
  EXPECT_THAT(table->Initialize(iter),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Expected shape")));
  EXPECT_FALSE(table->is_initialized());
}

TEST(InitializableLookupTableTest, RejectsLaterBatchWithMoreKeysThanValues) {
  auto table = MakeTable();
  // Only the second batch is malformed: inserting it would read one value per
  // key, past the end of its single-element values tensor.
  BatchIterator iter({
      {test::AsTensor<int64_t>({1}), test::AsTensor<int64_t>({10})},
      {test::AsTensor<int64_t>({2, 3, 4, 5}), test::AsTensor<int64_t>({20})},
  });
  EXPECT_THAT(table->Initialize(iter),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Expected shape")));
  EXPECT_FALSE(table->is_initialized());
}

TEST(InitializableLookupTableTest, RejectsLaterBatchWithExtraValueDims) {
  auto table = MakeTable();
  BatchIterator iter({
      {test::AsTensor<int64_t>({1}), test::AsTensor<int64_t>({10})},
      {test::AsTensor<int64_t>({2}),
       test::AsTensor<int64_t>({20, 21}, TensorShape({1, 2}))},
  });
  EXPECT_THAT(table->Initialize(iter),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("Expected shape")));
  EXPECT_FALSE(table->is_initialized());
}

}  // namespace
}  // namespace lookup
}  // namespace tensorflow
