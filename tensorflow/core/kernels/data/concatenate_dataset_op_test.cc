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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/platform/test.h"

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

template <typename Tag>
struct AccessResult {
  static typename Tag::type ptr;
};
template <typename Tag>
typename Tag::type AccessResult<Tag>::ptr;

template <typename Tag, typename Tag::type p>
struct AccessRob {
  struct Filler {
    Filler() { AccessResult<Tag>::ptr = p; }
  };
  static Filler filler;
};
template <typename Tag, typename Tag::type p>
typename AccessRob<Tag, p>::Filler AccessRob<Tag, p>::filler;

struct MakeIteratorInternalTag {
  typedef std::unique_ptr<IteratorBase> (DatasetBase::*type)(
      const std::string&) const;
};
template class AccessRob<MakeIteratorInternalTag,
                         &DatasetBase::MakeIteratorInternal>;

TEST_F(ConcatenateDatasetOpTest, UninitializedIteratorRestore) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));

  std::vector<const DatasetBase*> inputs;
  const DatasetBase* target_dataset = dataset_;
  while (target_dataset->type_string() != "ConcatenateDataset") {
    inputs.clear();
    TF_ASSERT_OK(target_dataset->InputDatasets(&inputs));
    ASSERT_FALSE(inputs.empty());
    target_dataset = inputs[0];
  }

  auto method = AccessResult<MakeIteratorInternalTag>::ptr;
  auto iterator = (target_dataset->*method)("test_prefix");

  IteratorContext iter_ctx(IteratorContext::Params(iterator_ctx_.get()));
  std::vector<const VariantTensorData*> data;
  VariantTensorDataReader reader(data);
  absl::Status s = iterator->Restore(&iter_ctx, &reader);

  EXPECT_EQ(s.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(s.message(),
            "`Initialize` should be called before saving/restoring from "
            "tf.data checkpoints.");
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
