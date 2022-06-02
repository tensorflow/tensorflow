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
#include "tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "assert_prev_dataset";

// Returns a stringified `NameAttrList`, the input to `AssertPrevDatasetOp`.
std::string GetTransformation(
    absl::string_view name,
    std::initializer_list<std::pair<std::string, bool>> attrs = {}) {
  NameAttrList message;
  message.set_name(absl::StrCat(name, "Dataset"));
  for (const auto& attr : attrs) {
    AttrValue value;
    value.set_b(attr.second);
    message.mutable_attr()->insert({attr.first, value});
  }
  std::string output;
  protobuf::TextFormat::PrintToString(message, &output);
  return output;
}

class AssertPrevDatasetParams : public DatasetParams {
 public:
  template <typename T>
  AssertPrevDatasetParams(T input_dataset_params,
                          const std::vector<tstring>& transformations,
                          DataTypeVector output_dtypes,
                          std::vector<PartialTensorShape> output_shapes,
                          string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        transformations_(transformations) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    int num_transformations = transformations_.size();
    return {CreateTensor<tstring>(TensorShape({num_transformations}),
                                  transformations_)};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->reserve(input_dataset_params_.size() + 1);
    input_names->emplace_back(AssertPrevDatasetOp::kInputDataset);
    input_names->emplace_back(AssertPrevDatasetOp::kTransformations);
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{AssertPrevDatasetOp::kOutputShapes, output_shapes_},
                    {AssertPrevDatasetOp::kOutputTypes, output_dtypes_}};
    return OkStatus();
  }

  string dataset_type() const override {
    return AssertPrevDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> transformations_;
};

class AssertPrevDatasetOpTest : public DatasetOpsTestBase {};

AssertPrevDatasetParams AssertPrevDatasetParams1() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType)},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams AssertPrevDatasetParams2() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType),
       GetTransformation(RangeDatasetOp::kDatasetType)},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams AssertPrevDatasetParams2WithAttrs() {
  TakeDatasetParams take_dataset_params = TakeDatasetParams(
      TensorSliceDatasetParams(
          /*components=*/
          {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                 {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*node_name=*/"tensor_slice_dataset"),
      /*count=*/3,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType),
       GetTransformation(TensorSliceDatasetOp::kDatasetType,
                         {{"is_files", false}})},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams InvalidAssertPrevDatasetParams() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/{GetTransformation("Whoops")},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams ShortAssertPrevDatasetParams() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType),
       GetTransformation(RangeDatasetOp::kDatasetType),
       GetTransformation("Whoops")},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<AssertPrevDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/AssertPrevDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/AssertPrevDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})}};
}

ITERATOR_GET_NEXT_TEST_P(AssertPrevDatasetOpTest, AssertPrevDatasetParams,
                         GetNextTestCases())

TEST_F(AssertPrevDatasetOpTest, DatasetNodeName) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(AssertPrevDatasetOpTest, DatasetAttrs) {
  auto dataset_params = AssertPrevDatasetParams2WithAttrs();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(AssertPrevDatasetOpTest, DatasetTypeString) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(AssertPrevDatasetOp::kDatasetType)));
}

TEST_F(AssertPrevDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(AssertPrevDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(AssertPrevDatasetOpTest, Cardinality) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(/*expected_cardinality=*/3));
}

TEST_F(AssertPrevDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(AssertPrevDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(AssertPrevDatasetOpTest, IteratorPrefix) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      AssertPrevDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<AssertPrevDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/AssertPrevDatasetParams1(),
           /*breakpoints*/ {0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/AssertPrevDatasetParams2(),
           /*breakpoints*/ {0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(AssertPrevDatasetOpTest,
                                 AssertPrevDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(AssertPrevDatasetOpTest, InvalidArguments) {
  auto dataset_params = InvalidAssertPrevDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(AssertPrevDatasetOpTest, ShortAssertPrev) {
  auto dataset_params = ShortAssertPrevDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
