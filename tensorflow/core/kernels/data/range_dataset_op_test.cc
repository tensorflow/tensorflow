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
#include "tensorflow/core/kernels/data/range_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "range_dataset";

class RangeDatasetOpTest : public DatasetOpsTestBaseV2<RangeDatasetParams> {
 public:
  Status Initialize(RangeDatasetParams* range_dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime({}, cpu_num_));

    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*range_dataset_params, &dataset_kernel_));
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(range_dataset_params->MakeInputs(&inputs));
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(
        iterator_ctx_.get(), range_dataset_params->iterator_prefix,
        &iterator_));
    return Status::OK();
  }

 protected:
  Status MakeDatasetOpKernel(
      const RangeDatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* range_dataset_op_kernel) override {
    NodeDef node_def = test::function::NDef(
        dataset_params.node_name,
        name_utils::OpName(RangeDatasetOp::kDatasetType),
        {RangeDatasetOp::kStart, RangeDatasetOp::kStop, RangeDatasetOp::kStep},
        {{RangeDatasetOp::kOutputTypes, dataset_params.output_dtypes},
         {RangeDatasetOp::kOutputShapes, dataset_params.output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, range_dataset_op_kernel));
    return Status::OK();
  }
};

RangeDatasetParams PositiveStepRangeDatasetParams() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/3,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

RangeDatasetParams NegativeStepRangeDatasetParams() {
  return {/*start=*/10,
          /*stop=*/0,
          /*step=*/-3,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

RangeDatasetParams ZeroStepRangeDatasetParams() {
  return {/*start=*/10,
          /*stop=*/0,
          /*step=*/0,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<RangeDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{10}, {7}, {4}, {1}})}};
}

ITERATOR_GET_NEXT_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                         GetNextTestCases())

TEST_F(RangeDatasetOpTest, DatasetNodeName) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(range_dataset_params.node_name));
}

TEST_F(RangeDatasetOpTest, DatasetTypeString) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(RangeDatasetOp::kDatasetType)));
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<RangeDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*expected_cardinality=*/4}};
}

DATASET_CARDINALITY_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                           CardinalityTestCases())

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(RangeDatasetOpTest, IteratorPrefix) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(&range_dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      RangeDatasetOp::kDatasetType, range_dataset_params.iterator_prefix)));
}

std::vector<IteratorSaveAndRestoreTestCase<RangeDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({}), {{10}, {7}, {4}, {1}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(RangeDatasetOpTest, ZeroStep) {
  auto range_dataset_params = ZeroStepRangeDatasetParams();
  EXPECT_EQ(Initialize(&range_dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
