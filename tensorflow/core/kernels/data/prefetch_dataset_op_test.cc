/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "prefetch_dataset";

class PrefetchDatasetOpTest : public DatasetOpsTestBase {};

class PrefetchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  PrefetchDatasetParams(T input_dataset_params, int64_t buffer_size,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        int64_t slack_period, bool legacy_autotune,
                        int64_t buffer_size_min, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        buffer_size_(buffer_size),
        slack_period_(slack_period),
        legacy_autotune_(legacy_autotune),
        buffer_size_min_(buffer_size_min) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {buffer_size_})};
  }

  absl::Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(PrefetchDatasetOp::kInputDataset);
    input_names->emplace_back(PrefetchDatasetOp::kBufferSize);
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("slack_period", slack_period_);
    attr_vector->emplace_back("legacy_autotune", legacy_autotune_);
    attr_vector->emplace_back("buffer_size_min", buffer_size_min_);
    attr_vector->emplace_back("metadata", "");
    return absl::OkStatus();
  }

  string dataset_type() const override {
    return PrefetchDatasetOp::kDatasetType;
  }

 private:
  int64_t buffer_size_;
  int64_t slack_period_;
  bool legacy_autotune_;
  int64_t buffer_size_min_;
};

// Test case 1: positive buffer size.
PrefetchDatasetParams PrefetchDatasetParams1() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/5,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 2: zero buffer size.
PrefetchDatasetParams PrefetchDatasetParams2() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/0,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 3: autotune buffer size.
PrefetchDatasetParams PrefetchDatasetParams3() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 4: slack_period > 0.
PrefetchDatasetParams PrefetchDatasetParams4() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/5,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 5: legacy_autotune = false.
PrefetchDatasetParams PrefetchDatasetParams5() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/5,
      /*legacy_autotune=*/false,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 6: buffer_size_min > 0.
PrefetchDatasetParams PrefetchDatasetParams6() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/3,
      /*node_name=*/kNodeName);
}

PrefetchDatasetParams InvalidBufferSizePrefetchDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-2,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<PrefetchDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/PrefetchDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/PrefetchDatasetParams2(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams3(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams4(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams5(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams6(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1},
           {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})}};
}

ITERATOR_GET_NEXT_TEST_P(PrefetchDatasetOpTest, PrefetchDatasetParams,
                         GetNextTestCases())

TEST_F(PrefetchDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(PrefetchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(PrefetchDatasetOp::kDatasetType)));
}

TEST_F(PrefetchDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(PrefetchDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<PrefetchDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/PrefetchDatasetParams1(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams2(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams3(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams4(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams5(),
           /*expected_cardinality=*/10}};
}

DATASET_CARDINALITY_TEST_P(PrefetchDatasetOpTest, PrefetchDatasetParams,
                           CardinalityTestCases())

TEST_F(PrefetchDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(PrefetchDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(PrefetchDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      PrefetchDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<PrefetchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/PrefetchDatasetParams1(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/PrefetchDatasetParams2(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams3(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams4(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams5(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1},
           {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(PrefetchDatasetOpTest, PrefetchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(PrefetchDatasetOpTest, InvalidBufferSize) {
  auto dataset_params = InvalidBufferSizePrefetchDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(), error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
