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
#include "tensorflow/core/kernels/data/experimental/assert_next_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "assert_next_dataset";

class AssertNextDatasetParams : public DatasetParams {
 public:
  template <typename T>
  AssertNextDatasetParams(T input_dataset_params,
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
    input_names->emplace_back(AssertNextDatasetOp::kInputDataset);
    input_names->emplace_back(AssertNextDatasetOp::kTransformations);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{AssertNextDatasetOp::kOutputShapes, output_shapes_},
                    {AssertNextDatasetOp::kOutputTypes, output_dtypes_}};
    return Status::OK();
  }

  string dataset_type() const override {
    return AssertNextDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> transformations_;
};

class AssertNextDatasetOpTest : public DatasetOpsTestBaseV2 {};

AssertNextDatasetParams AssertNextDatasetParams1() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertNextDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/{TakeDatasetOp::kDatasetType},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertNextDatasetParams AssertNextDatasetParams2() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertNextDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {TakeDatasetOp::kDatasetType, RangeDatasetOp::kDatasetType},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertNextDatasetParams InvalidAssertNextDatasetParams() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertNextDatasetParams(std::move(take_dataset_params),
                                 /*transformations=*/{"Whoops"},
                                 /*output_dtypes=*/{DT_INT64},
                                 /*output_shapes=*/{PartialTensorShape({})},
                                 /*node_name=*/kNodeName);
}

AssertNextDatasetParams ShortAssertNextDatasetParams() {
  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertNextDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {TakeDatasetOp::kDatasetType, RangeDatasetOp::kDatasetType, "Whoops"},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<AssertNextDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/AssertNextDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/AssertNextDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})}};
}

ITERATOR_GET_NEXT_TEST_P(AssertNextDatasetOpTest, AssertNextDatasetParams,
                         GetNextTestCases())

TEST_F(AssertNextDatasetOpTest, DatasetNodeName) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(AssertNextDatasetOpTest, DatasetTypeString) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(AssertNextDatasetOp::kDatasetType)));
}

TEST_F(AssertNextDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(AssertNextDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(AssertNextDatasetOpTest, Cardinality) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(/*expected_cardinality=*/3));
}

TEST_F(AssertNextDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(AssertNextDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(AssertNextDatasetOpTest, IteratorPrefix) {
  auto dataset_params = AssertNextDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      AssertNextDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<AssertNextDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/AssertNextDatasetParams1(),
           /*breakpoints*/ {0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/AssertNextDatasetParams2(),
           /*breakpoints*/ {0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(AssertNextDatasetOpTest,
                                 AssertNextDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(AssertNextDatasetOpTest, InvalidArguments) {
  auto dataset_params = InvalidAssertNextDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(AssertNextDatasetOpTest, ShortAssertNext) {
  auto dataset_params = ShortAssertNextDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
