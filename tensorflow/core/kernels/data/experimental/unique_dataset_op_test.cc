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
#include "tensorflow/core/kernels/data/experimental/unique_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "unique_dataset";

class UniqueDatasetParams : public DatasetParams {
 public:
  template <typename T>
  UniqueDatasetParams(T input_dataset_params, DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      kNodeName) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(UniqueDatasetOp::kInputDataset);
    return absl::OkStatus();
  }

  Status GetAttributes(AttributeVector* attributes) const override {
    *attributes = {{"output_types", output_dtypes_},
                   {"output_shapes", output_shapes_},
                   {"metadata", ""}};
    return absl::OkStatus();
  }

  string dataset_type() const override { return UniqueDatasetOp::kDatasetType; }
};

class UniqueDatasetOpTest : public DatasetOpsTestBase {};

UniqueDatasetParams NormalCaseParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{12, 1},
                             {1, 1, 2, 3, 5, 8, 13, 3, 21, 8, 8, 34})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(tensor_slice_dataset_params,
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams LastRecordIsDuplicateParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{11, 1},
                             {1, 1, 2, 3, 5, 8, 13, 3, 21, 8, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams AllRecordsTheSameParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{5, 1}, {1, 1, 1, 1, 1})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams EmptyInputParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{0, 1}, {})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams StringParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(
          TensorShape{11, 1},
          {"one", "One", "two", "three", "five", "eight", "thirteen",
           "twenty-one", "eight", "eight", "thirty-four"})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_STRING},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

// Two components in input dataset --> Should result in error during dataset
// construction
UniqueDatasetParams TwoComponentsParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {
          CreateTensor<int64_t>(TensorShape{1, 1}, {1}),
          CreateTensor<int64_t>(TensorShape{1, 1}, {42}),
      },
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1}), PartialTensorShape({1})});
}

// Zero components in input dataset --> Should result in error during dataset
// construction
UniqueDatasetParams NoInputParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({})});
}

// Floating-point --> Should result in error during dataset construction
UniqueDatasetParams FP32Params() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<float>(TensorShape{1, 1}, {3.14})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_FLOAT},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

std::vector<GetNextTestCase<UniqueDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}, {34}})},
          {/*dataset_params=*/LastRecordIsDuplicateParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}})},
          {/*dataset_params=*/AllRecordsTheSameParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{1}})},
          {/*dataset_params=*/EmptyInputParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {})},
          {/*dataset_params=*/StringParams(),
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"one"},
                                                     {"One"},
                                                     {"two"},
                                                     {"three"},
                                                     {"five"},
                                                     {"eight"},
                                                     {"thirteen"},
                                                     {"twenty-one"},
                                                     {"thirty-four"}})}};
}

ITERATOR_GET_NEXT_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                         GetNextTestCases())

TEST_F(UniqueDatasetOpTest, DatasetNodeName) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(UniqueDatasetOpTest, DatasetTypeString) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(UniqueDatasetOp::kDatasetType)));
}

TEST_F(UniqueDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(UniqueDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({1})}));
}

std::vector<CardinalityTestCase<UniqueDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_cardinality=*/kUnknownCardinality},
          // Current implementation doesn't propagate input cardinality of zero
          // to its output cardinality.
          {/*dataset_params=*/EmptyInputParams(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<UniqueDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/StringParams(),
           /*expected_output_dtypes=*/{DT_STRING}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<UniqueDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/StringParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(UniqueDatasetOpTest, IteratorPrefix) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      UniqueDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<UniqueDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*breakpoints=*/{0, 2, 6, 8},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}, {34}})},
          {/*dataset_params=*/LastRecordIsDuplicateParams(),
           /*breakpoints=*/{0, 2, 6, 8},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

class ParameterizedInvalidInputTest
    : public UniqueDatasetOpTest,
      public ::testing::WithParamInterface<UniqueDatasetParams> {};

TEST_P(ParameterizedInvalidInputTest, InvalidInput) {
  auto dataset_params = GetParam();
  auto result = Initialize(dataset_params);
  EXPECT_FALSE(result.ok());
}

INSTANTIATE_TEST_SUITE_P(FilterDatasetOpTest, ParameterizedInvalidInputTest,
                         ::testing::ValuesIn({TwoComponentsParams(),
                                              NoInputParams(), FP32Params()}));

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
