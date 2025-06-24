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
#include "tensorflow/core/kernels/data/repeat_dataset_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "repeat_dataset";

class RepeatDatasetParams : public DatasetParams {
 public:
  template <typename T>
  RepeatDatasetParams(T input_dataset_params, int64_t count,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        count_(count) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {count_})};
  }

  absl::Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(RepeatDatasetOp::kInputDataset);
    input_names->emplace_back(RepeatDatasetOp::kCount);
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("metadata", "");
    return absl::OkStatus();
  }

  string dataset_type() const override { return RepeatDatasetOp::kDatasetType; }

 private:
  int64_t count_;
};

class RepeatDatasetOpTest : public DatasetOpsTestBase {};

RepeatDatasetParams FiniteRepeatDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{2, 2}, {1, 2, 3, 4}),
                      CreateTensor<tstring>(TensorShape{2, 1}, {"a", "b"})},
      /*node_name=*/"tensor_slice");
  return RepeatDatasetParams(
      /*input_dataset_params=*/std::move(tensor_slice_dataset_params),
      /*count=*/2,
      /*output_dtypes=*/{DT_INT64, DT_STRING},
      /*output_shapes=*/{PartialTensorShape({2}), PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

RepeatDatasetParams EmptyRepeatDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{2, 2}, {1, 2, 3, 4}),
                      CreateTensor<tstring>(TensorShape{2, 1}, {"a", "b"})},
      /*node_name=*/"tensor_slice");
  return RepeatDatasetParams(
      /*input_dataset_params=*/std::move(tensor_slice_dataset_params),
      /*count=*/0,
      /*output_dtypes=*/{DT_INT64, DT_STRING},
      /*output_shapes=*/{PartialTensorShape({2}), PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

RepeatDatasetParams ForeverRepeatDatasetParams() {
  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{2, 1}, {1, 2})},
      /*node_name=*/"tensor_slice");
  return RepeatDatasetParams(
      /*input_dataset_params=*/std::move(tensor_slice_dataset_params),
      /*count=*/-1,
      /*output_dtypes=*/{DT_INT64, DT_STRING},
      /*output_shapes=*/{PartialTensorShape({2}), PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<RepeatDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/FiniteRepeatDatasetParams(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2}, {1, 2}),
            CreateTensor<tstring>(TensorShape{1}, {"a"}),
            CreateTensor<int64_t>(TensorShape{2}, {3, 4}),
            CreateTensor<tstring>(TensorShape{1}, {"b"}),
            CreateTensor<int64_t>(TensorShape{2}, {1, 2}),
            CreateTensor<tstring>(TensorShape{1}, {"a"}),
            CreateTensor<int64_t>(TensorShape{2}, {3, 4}),
            CreateTensor<tstring>(TensorShape{1}, {"b"})}},
          {/*dataset_params=*/EmptyRepeatDatasetParams(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/
           ForeverRepeatDatasetParams(),
           // Use the first group of the repeated tensors to represent the
           // infinite outputs.
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{1}, {1}),
            CreateTensor<int64_t>(TensorShape{1}, {2})}}};
}

class ParameterizedIteratorGetNextOpTest
    : public RepeatDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<RepeatDatasetParams>> {};

TEST_P(ParameterizedIteratorGetNextOpTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  auto expected_outputs_it = test_case.expected_outputs.begin();
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;

  if (dataset_->Cardinality() == kInfiniteCardinality) {
    // We test only a finite number of steps of the infinite sequence.
    for (int i = 0; i < 100; ++i) {
      out_tensors.clear();
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      for (const auto& tensor : out_tensors) {
        TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
        expected_outputs_it++;
        // In the forever-repeat test case, the first group of the repeated
        // tensors is used to represent the expected outputs, so the iterator
        // of the expected outputs needs to be reset once it reaches the end.
        if (expected_outputs_it == test_case.expected_outputs.end()) {
          expected_outputs_it = test_case.expected_outputs.begin();
        }
      }
    }
    EXPECT_FALSE(end_of_sequence);
  } else {
    while (!end_of_sequence) {
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      if (!end_of_sequence) {
        for (const auto& tensor : out_tensors) {
          EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
          TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
          expected_outputs_it++;
        }
      }
    }
    EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
  }
}

INSTANTIATE_TEST_SUITE_P(RepeatDatasetOpTest,
                         ParameterizedIteratorGetNextOpTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(RepeatDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FiniteRepeatDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(RepeatDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FiniteRepeatDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(RepeatDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<RepeatDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/FiniteRepeatDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING}},
          {/*dataset_params=*/EmptyRepeatDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING}},
          {/*dataset_params=*/ForeverRepeatDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(RepeatDatasetOpTest, RepeatDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<RepeatDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/FiniteRepeatDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({2}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/EmptyRepeatDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({2}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/ForeverRepeatDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(RepeatDatasetOpTest, RepeatDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<RepeatDatasetParams>>
DatasetCardinalityTestCases() {
  return {{FiniteRepeatDatasetParams(), /*expected_cardinality=*/4},
          {EmptyRepeatDatasetParams(), /*expected_cardinality=*/0},
          {ForeverRepeatDatasetParams(),
           /*expected_cardinality=*/kInfiniteCardinality}};
}

DATASET_CARDINALITY_TEST_P(RepeatDatasetOpTest, RepeatDatasetParams,
                           DatasetCardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<RepeatDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/FiniteRepeatDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING}},
          {/*dataset_params=*/EmptyRepeatDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING}},
          {/*dataset_params=*/ForeverRepeatDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(RepeatDatasetOpTest, RepeatDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<RepeatDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/FiniteRepeatDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({2}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/EmptyRepeatDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({2}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/ForeverRepeatDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(RepeatDatasetOpTest, RepeatDatasetParams,
                              IteratorOutputShapesTestCases())

std::vector<IteratorPrefixTestCase<RepeatDatasetParams>>
IteratorPrefixTestCases() {
  return {
      {/*dataset_params=*/FiniteRepeatDatasetParams(),
       /*expected_iterator_prefix=*/
       name_utils::IteratorPrefix(
           "FiniteRepeat", FiniteRepeatDatasetParams().iterator_prefix())},
      {/*dataset_params=*/EmptyRepeatDatasetParams(),
       /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
           "EmptyRepeat", EmptyRepeatDatasetParams().iterator_prefix())},
      {/*dataset_params=*/ForeverRepeatDatasetParams(),
       /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
           "ForeverRepeat", ForeverRepeatDatasetParams().iterator_prefix())}};
}

ITERATOR_PREFIX_TEST_P(RepeatDatasetOpTest, RepeatDatasetParams,
                       IteratorPrefixTestCases())

std::vector<IteratorSaveAndRestoreTestCase<RepeatDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/FiniteRepeatDatasetParams(),
           /*breakpoints*/ {0, 1, 3},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2}, {1, 2}),
            CreateTensor<tstring>(TensorShape{1}, {"a"}),
            CreateTensor<int64_t>(TensorShape{2}, {3, 4}),
            CreateTensor<tstring>(TensorShape{1}, {"b"}),
            CreateTensor<int64_t>(TensorShape{2}, {1, 2}),
            CreateTensor<tstring>(TensorShape{1}, {"a"}),
            CreateTensor<int64_t>(TensorShape{2}, {3, 4}),
            CreateTensor<tstring>(TensorShape{1}, {"b"})}},
          {/*dataset_params=*/EmptyRepeatDatasetParams(),
           /*breakpoints*/ {0, 1, 3},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/
           ForeverRepeatDatasetParams(),
           /*breakpoints*/ {0, 1, 3},
           // Use the first group of the repeated tensors to represent the
           // infinite outputs.
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{1}, {1}),
            CreateTensor<int64_t>(TensorShape{1}, {2})}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public RepeatDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<RepeatDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, Roundtrip) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  auto expected_outputs_it = test_case.expected_outputs.begin();
  bool end_of_sequence = dataset_->Cardinality() == 0;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  std::vector<int> breakpoints = GetParam().breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration < breakpoint) {
      out_tensors.clear();
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      if (!end_of_sequence) {
        for (auto& tensor : out_tensors) {
          EXPECT_NE(expected_outputs_it, test_case.expected_outputs.end());
          TF_EXPECT_OK(ExpectEqual(tensor, *expected_outputs_it));
          expected_outputs_it++;
        }
      }
      cur_iteration++;
      if (dataset_->Cardinality() == kInfiniteCardinality &&
          expected_outputs_it == test_case.expected_outputs.end()) {
        expected_outputs_it = test_case.expected_outputs.begin();
      }
    }

    if (breakpoint >= dataset_->Cardinality()) {
      if (dataset_->Cardinality() == kInfiniteCardinality) {
        EXPECT_FALSE(end_of_sequence);
      } else {
        EXPECT_TRUE(end_of_sequence);
        EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
      }
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    RepeatDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
