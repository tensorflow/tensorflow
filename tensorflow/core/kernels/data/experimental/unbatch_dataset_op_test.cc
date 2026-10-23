/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "unbatch_dataset";

class UnbatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  UnbatchDatasetParams(T input_dataset_params, DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       std::string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back("input_dataset");
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    return absl::OkStatus();
  }

  std::string dataset_type() const override { return "Unbatch"; }
};

class UnbatchDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Restores the iterator from a hand-written checkpoint that claims the
  // input iterator is exhausted and that a batch of `tensors` is partially
  // consumed, with `current_index` elements already produced out of
  // `batch_size`.
  absl::Status RestoreFromCheckpoint(const UnbatchDatasetParams& dataset_params,
                                     int64_t current_index, int64_t batch_size,
                                     const std::vector<Tensor>& tensors,
                                     std::unique_ptr<IteratorBase>* iterator) {
    VariantTensorDataWriter writer;
    TF_RETURN_IF_ERROR(writer.WriteScalar(
        FullName(iterator_->prefix(), "input_impl_empty"), int64_t{1}));
    TF_RETURN_IF_ERROR(writer.WriteScalar(
        FullName(iterator_->prefix(), "current_index"), current_index));
    TF_RETURN_IF_ERROR(
        writer.WriteScalar(FullName(iterator_->prefix(), "n"), batch_size));
    for (int i = 0; i < tensors.size(); ++i) {
      TF_RETURN_IF_ERROR(writer.WriteTensor(
          FullName(iterator_->prefix(), absl::StrCat("tensors[", i, "]")),
          tensors[i]));
    }
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    return RestoreIterator(iterator_ctx_.get(), &reader,
                           dataset_params.iterator_prefix(), *dataset_,
                           iterator);
  }
};

// Two batches of two scalars each: [[0, 1], [2, 3]].
UnbatchDatasetParams UnbatchDatasetParams1() {
  return UnbatchDatasetParams(TensorSliceDatasetParams(
                                  /*components=*/{CreateTensor<int64_t>(
                                      TensorShape{2, 2}, {0, 1, 2, 3})},
                                  /*node_name=*/"tensor_slice_dataset"),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<UnbatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/UnbatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}})}};
}

ITERATOR_GET_NEXT_TEST_P(UnbatchDatasetOpTest, UnbatchDatasetParams,
                         GetNextTestCases())

std::vector<IteratorSaveAndRestoreTestCase<UnbatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/UnbatchDatasetParams1(),
           /*breakpoints=*/{0, 1, 3, 4, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(UnbatchDatasetOpTest, UnbatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(UnbatchDatasetOpTest, RestoreConsistentPartialBatch) {
  auto dataset_params = UnbatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(RestoreFromCheckpoint(
      dataset_params, /*current_index=*/1, /*batch_size=*/2,
      {CreateTensor<int64_t>(TensorShape{2}, {10, 20})}, &iterator));
}

// The checkpoint records a batch of 2, and `current_index` is past `n`.
TEST_F(UnbatchDatasetOpTest, RestoreRejectsIndexPastBatch) {
  auto dataset_params = UnbatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  EXPECT_THAT(
      RestoreFromCheckpoint(
          dataset_params, /*current_index=*/5, /*batch_size=*/2,
          {CreateTensor<int64_t>(TensorShape{2}, {10, 20})}, &iterator),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Invalid current_index")));
}

TEST_F(UnbatchDatasetOpTest, RestoreRejectsBatchSizeMismatch) {
  auto dataset_params = UnbatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  EXPECT_THAT(RestoreFromCheckpoint(
                  dataset_params, /*current_index=*/1, /*batch_size=*/100,
                  {CreateTensor<int64_t>(TensorShape{2}, {10, 20})}, &iterator),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("batch size")));
}

TEST_F(UnbatchDatasetOpTest, RestoreRejectsNegativeIndex) {
  auto dataset_params = UnbatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  EXPECT_THAT(
      RestoreFromCheckpoint(
          dataset_params, /*current_index=*/-1, /*batch_size=*/2,
          {CreateTensor<int64_t>(TensorShape{2}, {10, 20})}, &iterator),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Invalid current_index")));
}

TEST_F(UnbatchDatasetOpTest, RestoreRejectsScalarComponent) {
  auto dataset_params = UnbatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  EXPECT_THAT(RestoreFromCheckpoint(
                  dataset_params, /*current_index=*/0,
                  /*batch_size=*/1,
                  {CreateTensor<int64_t>(TensorShape{}, {10})}, &iterator),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("non-scalar")));
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
