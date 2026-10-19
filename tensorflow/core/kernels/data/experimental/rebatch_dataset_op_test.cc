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

constexpr char kNodeName[] = "rebatch_dataset_v2";

class RebatchDatasetV2Params : public DatasetParams {
 public:
  template <typename T>
  RebatchDatasetV2Params(T input_dataset_params,
                         std::vector<int64_t> batch_sizes, bool drop_remainder,
                         DataTypeVector output_dtypes,
                         std::vector<PartialTensorShape> output_shapes,
                         std::string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        batch_sizes_(std::move(batch_sizes)),
        drop_remainder_(drop_remainder) {
    op_version_ = 2;
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(
                TensorShape({static_cast<int64_t>(batch_sizes_.size())}),
                batch_sizes_),
            CreateTensor<bool>(TensorShape({}), {drop_remainder_})};
  }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back("input_dataset");
    input_names->emplace_back("batch_sizes");
    input_names->emplace_back("drop_remainder");
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    return absl::OkStatus();
  }

  std::string dataset_type() const override { return "Rebatch"; }

 private:
  std::vector<int64_t> batch_sizes_;
  bool drop_remainder_;
};

class RebatchDatasetV2OpTest : public DatasetOpsTestBase {
 protected:
  // Restores the iterator from a hand-written checkpoint that claims the
  // input iterator is exhausted and that the next batch should continue from
  // `offset` within the batch `tensors`, using batch size number
  // `batch_sizes_index`.
  absl::Status RestoreFromCheckpoint(
      const RebatchDatasetV2Params& dataset_params, int64_t batch_sizes_index,
      int64_t offset, const std::vector<Tensor>& tensors,
      std::unique_ptr<IteratorBase>* iterator) {
    VariantTensorDataWriter writer;
    TF_RETURN_IF_ERROR(writer.WriteScalar(
        FullName(iterator_->prefix(), "input_impl_empty"), ""));
    TF_RETURN_IF_ERROR(writer.WriteScalar(
        FullName(iterator_->prefix(), "batch_sizes_index"), batch_sizes_index));
    TF_RETURN_IF_ERROR(
        writer.WriteScalar(FullName(iterator_->prefix(), "offset"), offset));
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

// Two input batches of four scalars each, rebatched into batches of two.
RebatchDatasetV2Params RebatchDatasetV2Params1() {
  return RebatchDatasetV2Params(
      TensorSliceDatasetParams(
          /*components=*/{CreateTensor<int64_t>(TensorShape{2, 4},
                                                {0, 1, 2, 3, 4, 5, 6, 7})},
          /*node_name=*/"tensor_slice_dataset"),
      /*batch_sizes=*/{2},
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<RebatchDatasetV2Params>> GetNextTestCases() {
  return {{/*dataset_params=*/RebatchDatasetV2Params1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({2}),
                                  {{0, 1}, {2, 3}, {4, 5}, {6, 7}})}};
}

ITERATOR_GET_NEXT_TEST_P(RebatchDatasetV2OpTest, RebatchDatasetV2Params,
                         GetNextTestCases())

std::vector<IteratorSaveAndRestoreTestCase<RebatchDatasetV2Params>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/RebatchDatasetV2Params1(),
           /*breakpoints=*/{0, 1, 3, 4, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({2}),
                                  {{0, 1}, {2, 3}, {4, 5}, {6, 7}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(RebatchDatasetV2OpTest, RebatchDatasetV2Params,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(RebatchDatasetV2OpTest, RestoreConsistentPartialBatch) {
  auto dataset_params = RebatchDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(RestoreFromCheckpoint(
      dataset_params, /*batch_sizes_index=*/0, /*offset=*/2,
      {CreateTensor<int64_t>(TensorShape{4}, {10, 11, 12, 13})}, &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  TF_ASSERT_OK(
      iterator->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence));
  EXPECT_FALSE(end_of_sequence);
  TF_EXPECT_OK(ExpectEqual(out_tensors,
                           {CreateTensor<int64_t>(TensorShape{2}, {12, 13})},
                           /*compare_order=*/true));
}

// The dataset has a single batch size, so index 1 is past `batch_sizes_`.
TEST_F(RebatchDatasetV2OpTest, RestoreRejectsBatchSizesIndexOutOfRange) {
  auto dataset_params = RebatchDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  for (const int64_t batch_sizes_index : {int64_t{1}, int64_t{-1}}) {
    EXPECT_THAT(RestoreFromCheckpoint(dataset_params, batch_sizes_index,
                                      /*offset=*/-1, {}, &iterator),
                absl_testing::StatusIs(
                    absl::StatusCode::kInvalidArgument,
                    ::testing::HasSubstr("Invalid batch_sizes_index")))
        << "batch_sizes_index=" << batch_sizes_index;
  }
}

// The restored batch has four elements, so these offsets would slice it out
// of bounds.
TEST_F(RebatchDatasetV2OpTest, RestoreRejectsOffsetOutOfRange) {
  auto dataset_params = RebatchDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  for (const int64_t offset : {int64_t{4}, int64_t{100}, int64_t{-2}}) {
    EXPECT_THAT(RestoreFromCheckpoint(
                    dataset_params, /*batch_sizes_index=*/0, offset,
                    {CreateTensor<int64_t>(TensorShape{4}, {10, 11, 12, 13})},
                    &iterator),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       ::testing::HasSubstr("Invalid offset")))
        << "offset=" << offset;
  }
}

TEST_F(RebatchDatasetV2OpTest, RestoreRejectsScalarComponent) {
  auto dataset_params = RebatchDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  std::unique_ptr<IteratorBase> iterator;
  EXPECT_THAT(RestoreFromCheckpoint(
                  dataset_params, /*batch_sizes_index=*/0,
                  /*offset=*/0, {CreateTensor<int64_t>(TensorShape{}, {10})},
                  &iterator),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("non-scalar")));
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
