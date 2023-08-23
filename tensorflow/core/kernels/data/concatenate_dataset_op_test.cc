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
#include "tensorflow/core/kernels/data/concatenate_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "concatenate_dataset";

// Test case 1: same shape.
ConcatenateDatasetParams SameShapeConcatenateDatasetParams() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{2, 2},
                                            {{1, 2, 3, 4}, {5, 6, 7, 8}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(
          TensorShape{2, 2}, {{11, 12, 13, 14}, {15, 16, 17, 18}}),
      /*node_name=*/"tensor_slice_1");
  return ConcatenateDatasetParams(
      std::move(tensor_slice_dataset_params_0),
      std::move(tensor_slice_dataset_params_1),
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2}), PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

// Test case 2: different shape.
ConcatenateDatasetParams DifferentShapeConcatenateDatasetParams() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{2, 3}, {1, 2, 3, 4, 5, 6}),
       CreateTensor<int64_t>(TensorShape{2, 2}, {7, 8, 9, 10})},
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{2, 2}, {11, 12, 13, 14}),
       CreateTensor<int64_t>(TensorShape{2, 1}, {15, 16})},
      /*node_name=*/"tensor_slice_1");
  return ConcatenateDatasetParams(
      std::move(tensor_slice_dataset_params_0),
      std::move(tensor_slice_dataset_params_1),
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1}), PartialTensorShape({-1})},
      /*node_name=*/kNodeName);
}

// Test case 3: different dtypes
ConcatenateDatasetParams DifferentDtypeConcatenateDatasetParams() {
  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{2, 2}, {{1, 2, 3, 4}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/
      CreateTensors<double>(TensorShape{2, 2}, {{1.0, 2.0, 3.0, 4.0}}),
      /*node_name=*/"tensor_slice_1");
  return ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                                  std::move(tensor_slice_dataset_params_1),
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({2})},
                                  /*node_name=*/kNodeName);
}

class ConcatenateDatasetOpTest : public DatasetOpsTestBase {};

std::vector<GetNextTestCase<ConcatenateDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({2}), {{1, 2},
                                                     {5, 6},
                                                     {3, 4},
                                                     {7, 8},
                                                     {11, 12},
                                                     {15, 16},
                                                     {13, 14},
                                                     {17, 18}})},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{3}, {1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{3}, {4, 5, 6}),
            CreateTensor<int64_t>(TensorShape{2}, {9, 10}),
            CreateTensor<int64_t>(TensorShape{2}, {11, 12}),
            CreateTensor<int64_t>(TensorShape{1}, {15}),
            CreateTensor<int64_t>(TensorShape{2}, {13, 14}),
            CreateTensor<int64_t>(TensorShape{1}, {16})}}};
}

ITERATOR_GET_NEXT_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                         GetNextTestCases())

TEST_F(ConcatenateDatasetOpTest, DifferentDtypes) {
  auto dataset_params = DifferentDtypeConcatenateDatasetParams();

  EXPECT_EQ(Initialize(dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(ConcatenateDatasetOpTest, DatasetNodeName) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ConcatenateDatasetOpTest, DatasetTypeString) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ConcatenateDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<ConcatenateDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           SameShapeConcatenateDatasetParams().output_dtypes()},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           DifferentShapeConcatenateDatasetParams().output_dtypes()}};
}

DATASET_OUTPUT_DTYPES_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<ConcatenateDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_shapes*/
           SameShapeConcatenateDatasetParams().output_shapes()},
          {/*dataset_params=*/
           DifferentShapeConcatenateDatasetParams(),
           /*expected_output_shapes*/
           DifferentShapeConcatenateDatasetParams().output_shapes()}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ConcatenateDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_cardinality=*/4}};
}

DATASET_CARDINALITY_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<ConcatenateDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           SameShapeConcatenateDatasetParams().output_dtypes()},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           DifferentShapeConcatenateDatasetParams().output_dtypes()}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(ConcatenateDatasetOpTest,
                              ConcatenateDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<ConcatenateDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_shapes=*/
           SameShapeConcatenateDatasetParams().output_shapes()},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_output_shapes=*/
           DifferentShapeConcatenateDatasetParams().output_shapes()}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ConcatenateDatasetOpTest,
                              ConcatenateDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ConcatenateDatasetOpTest, IteratorPrefix) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ConcatenateDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ConcatenateDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({2}), {{1, 2},
                                                     {5, 6},
                                                     {3, 4},
                                                     {7, 8},
                                                     {11, 12},
                                                     {15, 16},
                                                     {13, 14},
                                                     {17, 18}})},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{3}, {1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{3}, {4, 5, 6}),
            CreateTensor<int64_t>(TensorShape{2}, {9, 10}),
            CreateTensor<int64_t>(TensorShape{2}, {11, 12}),
            CreateTensor<int64_t>(TensorShape{1}, {15}),
            CreateTensor<int64_t>(TensorShape{2}, {13, 14}),
            CreateTensor<int64_t>(TensorShape{1}, {16})}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ConcatenateDatasetOpTest,
                                 ConcatenateDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
