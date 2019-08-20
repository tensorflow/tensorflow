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
#include "tensorflow/core/kernels/data/batch_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "batch_dataset_v2";
constexpr int kOpVersion = 2;

class BatchDatasetParams : public DatasetParams {
 public:
  BatchDatasetParams(int64 num_input_elements, int64 batch_size,
                     bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_input_elements(num_input_elements),
        batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
        drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        parallel_copy(parallel_copy) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (!IsDatasetTensor(input_dataset)) {
      return errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset), TensorValue(&batch_size),
               TensorValue(&drop_remainder)};
    return Status::OK();
  }

  int64 num_input_elements;
  Tensor input_dataset;
  Tensor batch_size;
  Tensor drop_remainder;
  bool parallel_copy;
};

class BatchDatasetOpTest : public DatasetOpsTestBaseV2<BatchDatasetParams> {
 public:
  Status Initialize(BatchDatasetParams* batch_dataset_params) override {
    TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime({}, cpu_num_));

    // Populate the `input_dataset` in `batch_dataset_params_`.
    RangeDatasetParams input_dataset_params(
        0, batch_dataset_params->num_input_elements, 1, {DT_INT64},
        {PartialTensorShape({})}, "range_dataset");
    TF_RETURN_IF_ERROR(MakeRangeDataset(input_dataset_params,
                                        &batch_dataset_params->input_dataset));
    // Create the dataset kernel.
    TF_RETURN_IF_ERROR(
        MakeDatasetOpKernel(*batch_dataset_params, &dataset_kernel_));
    // Create the inputs for the dataset op.
    gtl::InlinedVector<TensorValue, 4> inputs;
    TF_RETURN_IF_ERROR(batch_dataset_params->MakeInputs(&inputs));
    // Creat the dataset context.
    TF_RETURN_IF_ERROR(
        CreateDatasetContext(dataset_kernel_.get(), &inputs, &dataset_ctx_));
    // Create the dataset.
    TF_RETURN_IF_ERROR(
        CreateDataset(dataset_kernel_.get(), dataset_ctx_.get(), &dataset_));

    // Create the iterator context.
    TF_RETURN_IF_ERROR(
        CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
    // Create the iterator.
    TF_RETURN_IF_ERROR(dataset_->MakeIterator(
        iterator_ctx_.get(), batch_dataset_params->iterator_prefix,
        &iterator_));
    return Status::OK();
  }

 protected:
  // Creates a new `BatchDataset` op kernel.
  Status MakeDatasetOpKernel(
      const BatchDatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* batch_dataset_op_kernel) override {
    name_utils::OpNameParams params;
    params.op_version = kOpVersion;
    NodeDef node_def = test::function::NDef(
        dataset_params.node_name,
        name_utils::OpName(BatchDatasetOp::kDatasetType, params),
        {BatchDatasetOp::kInputDataset, BatchDatasetOp::kBatchSize,
         BatchDatasetOp::kDropRemainder},
        {{BatchDatasetOp::kParallelCopy, dataset_params.parallel_copy},
         {BatchDatasetOp::kOutputTypes, dataset_params.output_dtypes},
         {BatchDatasetOp::kOutputShapes, dataset_params.output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, batch_dataset_op_kernel));
    return Status::OK();
  }
};

// Test Case 1: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDatasetParams1() {
  return {/*num_input_elements=*/12,
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/kNodeName};
}

// Test Case 2: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDatasetParams2() {
  return {/*num_input_elements=*/12,
          /*batch_size=*/4,
          /*drop_remainder=*/true,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/kNodeName};
}

// Test Case 3: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDatasetParams3() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/3,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/kNodeName};
}

// Test Case 4: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDatasetParams4() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/3,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/kNodeName};
}

// Test Case 5: test BatchDatasetV2 with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDatasetParams5() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/12,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({12})},
          /*node_name=*/kNodeName};
}

// Test Case 6: test BatchDatasetV2 with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDatasetParams6() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/12,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/kNodeName};
}

// Test Case 7: test BatchDatasetV2 with `drop_remainder` = false and
// the output of the input dataset is empty.
BatchDatasetParams BatchDatasetParams7() {
  return {/*num_input_elements=*/0,
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/kNodeName};
}

// Test Case 8: test BatchDatasetV2 with an invalid batch size
BatchDatasetParams InvalidBatchSizeBatchDatasetParams() {
  return {/*num_input_elements=*/10,
          /*batch_size=*/-1,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<BatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({3}),
                                {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({10}),
                                {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})},

          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                         GetNextTestCases())

TEST_F(BatchDatasetOpTest, DatasetNodeName) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&batch_dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(batch_dataset_params.node_name));
}

TEST_F(BatchDatasetOpTest, DatasetTypeString) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&batch_dataset_params));
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(BatchDatasetOp::kDatasetType, params)));
}

TEST_F(BatchDatasetOpTest, DatasetOutputDtypes) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&batch_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<BatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<BatchDatasetParams>> CardinalityTestCases() {
  return {
      {/*dataset_params=*/BatchDatasetParams1(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams2(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams3(), /*expected_cardinality=*/4},
      {/*dataset_params=*/BatchDatasetParams4(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams5(), /*expected_cardinality=*/0},
      {/*dataset_params=*/BatchDatasetParams6(), /*expected_cardinality=*/1},
      {/*dataset_params=*/BatchDatasetParams7(), /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                           CardinalityTestCases())

TEST_F(BatchDatasetOpTest, IteratorOutputDtypes) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&batch_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<BatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(BatchDatasetOpTest, IteratorOutputPrefix) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(&batch_dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      BatchDatasetOp::kDatasetType, batch_dataset_params.iterator_prefix,
      params)));
}

std::vector<IteratorSaveAndRestoreTestCase<BatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({10}),
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(BatchDatasetOpTest, InvalidBatchSize) {
  auto batch_dataset_params = InvalidBatchSizeBatchDatasetParams();
  EXPECT_EQ(Initialize(&batch_dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
