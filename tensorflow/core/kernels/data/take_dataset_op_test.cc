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
#include "tensorflow/core/kernels/data/take_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "take_dataset";

class TakeDatasetParams : public DatasetParams {
 public:
  template <typename T>
  TakeDatasetParams(T input_dataset_params, int count,
                    DataTypeVector output_dtypes,
                    std::vector<PartialTensorShape> output_shapes,
                    string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name), DatasetParamsType::Take),
        count_(CreateTensor<int64>(TensorShape({}), {count})) {
    auto input_dataset_params_ptr =
        std::make_shared<T>(std::move(input_dataset_params));
    input_dataset_params_group_.emplace_back(
        std::make_pair(std::move(input_dataset_params_ptr), Tensor()));
  }

  Status GetInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    inputs->reserve(input_dataset_params_group_.size() + 1);
    for (auto& pair : input_dataset_params_group_) {
      if (!IsDatasetTensor(pair.second)) {
        inputs->clear();
        return errors::Internal(
            "The input dataset is not populated as the dataset tensor yet.");
      } else {
        inputs->emplace_back(TensorValue(&pair.second));
      }
    }
    inputs->emplace_back(TensorValue(&count_));
    return Status::OK();
  }

  Status GetInputPlaceholder(
      std::vector<string>* input_placeholder) const override {
    input_placeholder->reserve(input_dataset_params_group_.size() + 1);
    input_placeholder->emplace_back(TakeDatasetOp::kInputDataset);
    input_placeholder->emplace_back(TakeDatasetOp::kCount);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{TakeDatasetOp::kOutputShapes, output_shapes_},
                    {TakeDatasetOp::kOutputTypes, output_dtypes_}};
    return Status::OK();
  }

 private:
  Tensor count_;
};

class TakeDatasetOpTest : public DatasetOpsTestBaseV2 {};

// Test case 1: take fewer than input size.
TakeDatasetParams TakeLessTakeDatasetParams() {
  return TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                           /*count=*/4,
                           /*output_dtypes=*/{DT_INT64},
                           /*output_shapes=*/{PartialTensorShape({1})},
                           /*node_name=*/kNodeName);
}

// Test case 2: take more than input size.
TakeDatasetParams TakeMoreTakeDatasetParams() {
  return TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                           /*count=*/25,
                           /*output_dtypes=*/{DT_INT64},
                           /*output_shapes=*/{PartialTensorShape({1})},
                           /*node_name=*/kNodeName);
}

// Test case 3: take all of input.
TakeDatasetParams TakeAllTakeDatasetParams() {
  return TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                           /*count=*/-1,
                           /*output_dtypes=*/{DT_INT64},
                           /*output_shapes=*/{PartialTensorShape({1})},
                           /*node_name=*/kNodeName);
}

// Test case 4: take nothing.
TakeDatasetParams TakeNothingTakeDatasetParams() {
  return TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                           /*count=*/0,
                           /*output_dtypes=*/{DT_INT64},
                           /*output_shapes=*/{PartialTensorShape({1})},
                           /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<TakeDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/TakeLessTakeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}})},
          {/*dataset_params=*/TakeMoreTakeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64>(
               TensorShape({}),
               {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
          {/*dataset_params=*/TakeAllTakeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64>(
               TensorShape({}),
               {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
          {/*dataset_params=*/TakeNothingTakeDatasetParams(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(TakeDatasetOpTest, TakeDatasetParams,
                         GetNextTestCases())

TEST_F(TakeDatasetOpTest, DatasetNodeName) {
  auto dataset_params = TakeLessTakeDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(TakeDatasetOpTest, DatasetTypeString) {
  auto dataset_params = TakeLessTakeDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(TakeDatasetOp::kDatasetType)));
}

TEST_F(TakeDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = TakeLessTakeDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<TakeDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/TakeLessTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/TakeMoreTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/TakeAllTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/TakeNothingTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(TakeDatasetOpTest, TakeDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<TakeDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/TakeLessTakeDatasetParams(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/TakeMoreTakeDatasetParams(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/TakeAllTakeDatasetParams(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/TakeNothingTakeDatasetParams(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(TakeDatasetOpTest, TakeDatasetParams,
                           CardinalityTestCases())

TEST_F(TakeDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = TakeLessTakeDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<TakeDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/TakeLessTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/TakeMoreTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/TakeAllTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/TakeNothingTakeDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(TakeDatasetOpTest, TakeDatasetParams,
                              IteratorOutputShapesTestCases())

std::vector<IteratorPrefixTestCase<TakeDatasetParams>>
IteratorPrefixTestCases() {
  return {{/*dataset_params=*/TakeLessTakeDatasetParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "FiniteTake", TakeLessTakeDatasetParams().iterator_prefix())},
          {/*dataset_params=*/TakeMoreTakeDatasetParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "FiniteTake", TakeMoreTakeDatasetParams().iterator_prefix())},
          {/*dataset_params=*/TakeAllTakeDatasetParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "FiniteTake", TakeAllTakeDatasetParams().iterator_prefix())},
          {/*dataset_params=*/TakeNothingTakeDatasetParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "EmptyTake", TakeNothingTakeDatasetParams().iterator_prefix())}};
}

ITERATOR_PREFIX_TEST_P(TakeDatasetOpTest, TakeDatasetParams,
                       IteratorPrefixTestCases())

std::vector<IteratorSaveAndRestoreTestCase<TakeDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/TakeLessTakeDatasetParams(),
           /*breakpoints=*/{0, 2, 5, 11},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}})},
          {/*dataset_params=*/TakeMoreTakeDatasetParams(),
           /*breakpoints=*/{0, 2, 5, 11},
           /*expected_outputs=*/
           CreateTensors<int64>(
               TensorShape({}),
               {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
          {/*dataset_params=*/TakeAllTakeDatasetParams(),
           /*breakpoints=*/{0, 2, 5, 11},
           /*expected_outputs=*/
           CreateTensors<int64>(
               TensorShape({}),
               {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
          {/*dataset_params=*/TakeNothingTakeDatasetParams(),
           /*breakpoints=*/{0, 2, 5, 11},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(TakeDatasetOpTest, TakeDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
