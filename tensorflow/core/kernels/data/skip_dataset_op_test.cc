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
#include "tensorflow/core/kernels/data/skip_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "skip_dataset";

class SkipDatasetParams : public DatasetParams {
 public:
  template <typename T>
  SkipDatasetParams(T input_dataset_params, int64 count,
                    DataTypeVector output_dtypes,
                    std::vector<PartialTensorShape> output_shapes,
                    string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        count_(count) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64>(TensorShape({}), {count_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(SkipDatasetOp::kInputDataset);
    input_names->emplace_back(SkipDatasetOp::kCount);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back(SkipDatasetOp::kOutputTypes, output_dtypes_);
    attr_vector->emplace_back(SkipDatasetOp::kOutputShapes, output_shapes_);
    return Status::OK();
  }

  string dataset_type() const override { return SkipDatasetOp::kDatasetType; }

 private:
  int64 count_;
};

class SkipDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: skip fewer than input size.
SkipDatasetParams SkipDatasetParams1() {
  return SkipDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*count=*/4,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 2: skip more than input size.
SkipDatasetParams SkipDatasetParams2() {
  return SkipDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*count=*/25,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 3: skip exactly the input size.
SkipDatasetParams SkipDatasetParams3() {
  return SkipDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*count=*/10,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 4: skip nothing.
SkipDatasetParams SkipDatasetParams4() {
  return SkipDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*count=*/0,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 5: set -1 for `count` to skip the entire dataset.
SkipDatasetParams SkipDatasetParams5() {
  return SkipDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 10, 1),
      /*count=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<SkipDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/SkipDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape{}, {{4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/SkipDatasetParams2(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/SkipDatasetParams3(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/SkipDatasetParams4(),
       /*expected_outputs=*/
       CreateTensors<int64>(
           TensorShape{}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/SkipDatasetParams5(),
       /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(SkipDatasetOpTest, SkipDatasetParams,
                         GetNextTestCases())

TEST_F(SkipDatasetOpTest, DatasetNodeName) {
  auto dataset_params = SkipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(SkipDatasetOpTest, DatasetTypeString) {
  auto dataset_params = SkipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(SkipDatasetOp::kDatasetType)));
}

TEST_F(SkipDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = SkipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(SkipDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = SkipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<SkipDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/SkipDatasetParams1(),
           /*expected_cardinality=*/6},
          {/*dataset_params=*/SkipDatasetParams2(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/SkipDatasetParams3(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/SkipDatasetParams4(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/SkipDatasetParams5(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(SkipDatasetOpTest, SkipDatasetParams,
                           CardinalityTestCases())

TEST_F(SkipDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = SkipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(SkipDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = SkipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

std::vector<IteratorPrefixTestCase<SkipDatasetParams>>
IteratorPrefixTestCases() {
  return {{/*dataset_params=*/SkipDatasetParams1(),
           /*expected_iterator_prefix=*/
           name_utils::IteratorPrefix("FiniteSkip",
                                      SkipDatasetParams1().iterator_prefix())},
          {/*dataset_params=*/SkipDatasetParams2(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "FiniteSkip", SkipDatasetParams2().iterator_prefix())},
          {/*dataset_params=*/SkipDatasetParams3(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "FiniteSkip", SkipDatasetParams3().iterator_prefix())},
          {/*dataset_params=*/SkipDatasetParams4(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "FiniteSkip", SkipDatasetParams4().iterator_prefix())},
          {/*dataset_params=*/SkipDatasetParams5(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               "EmptySkip", SkipDatasetParams5().iterator_prefix())}};
}

ITERATOR_PREFIX_TEST_P(SkipDatasetOpTest, SkipDatasetParams,
                       IteratorPrefixTestCases())

std::vector<IteratorSaveAndRestoreTestCase<SkipDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/SkipDatasetParams1(),
       /*breakpoints*/ {0, 2, 7},
       /*expected_outputs=*/
       CreateTensors<int64>(TensorShape{}, {{4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/SkipDatasetParams2(),
       /*breakpoints*/ {0, 2, 5},
       /*expected_outputs=*/{}},
      {/*dataset_params=*/SkipDatasetParams3(),
       /*breakpoints*/ {0, 2, 5},
       /*expected_outputs=*/{}},
      {/*dataset_params=*/SkipDatasetParams4(),
       /*breakpoints*/ {0, 2, 5, 11},
       /*expected_outputs=*/
       CreateTensors<int64>(
           TensorShape{}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/SkipDatasetParams5(),
       /*breakpoints*/ {0, 2, 5},
       /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(SkipDatasetOpTest, SkipDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
