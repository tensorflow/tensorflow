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
constexpr char kIteratorPrefix[] = "Iterator";

class BatchDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `BatchDataset` op kernel.
  Status CreateBatchDatasetOpKernel(
      bool parallel_copy, const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      std::unique_ptr<OpKernel>* batch_dataset_op_kernel) {
    name_utils::OpNameParams params;
    params.op_version = kOpVersion;
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(BatchDatasetOp::kDatasetType, params),
        {BatchDatasetOp::kInputDataset, BatchDatasetOp::kBatchSize,
         BatchDatasetOp::kDropRemainder},
        {{BatchDatasetOp::kParallelCopy, parallel_copy},
         {BatchDatasetOp::kOutputTypes, output_types},
         {BatchDatasetOp::kOutputShapes, output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, batch_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `BatchDataset` op kernel context
  Status CreateBatchDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

class BatchDatasetParams : public DatasetParams {
 public:
  BatchDatasetParams(int64 start, int64 stop, int64 step, int64 batch_size,
                     bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        range_dataset_params(start, stop, step, {DT_INT64},
                             {PartialTensorShape({})}, ""),
        batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
        drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        parallel_copy(parallel_copy) {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    if (input_dataset.NumElements() == 0 ||
        input_dataset.dtype() != DT_VARIANT) {
      return tensorflow::errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    }
    *inputs = {TensorValue(&input_dataset), TensorValue(&batch_size),
               TensorValue(&drop_remainder)};
    return Status::OK();
  }

  RangeDatasetParams range_dataset_params;  // Used to create the input dataset.
  Tensor input_dataset;
  Tensor batch_size;
  Tensor drop_remainder;
  bool parallel_copy;
};

// Test Case 1: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDataset1() {
  return {/*start=*/0,
          /*stop=*/12,
          /*step=*/1,
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 2: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDataset2() {
  return {/*start=*/0,
          /*stop=*/12,
          /*step=*/1,
          /*batch_size=*/4,
          /*drop_remainder=*/true,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 3: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDataset3() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*batch_size=*/3,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 4: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDataset4() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*batch_size=*/3,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 5: test BatchDatasetV2 with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDataset5() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*batch_size=*/12,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({12})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 6: test BatchDatasetV2 with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDataset6() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*batch_size=*/12,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 7: test BatchDatasetV2 with `drop_remainder` = false and
// the output of the input dataset is empty.
BatchDatasetParams BatchDataset7() {
  return {/*start=*/0,
          /*stop=*/0,
          /*step=*/1,
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/"batch_dataset"};
}

// Test Case 8: test BatchDatasetV2 with an invalid batch size
BatchDatasetParams InvalidBatchSizeBatchDataset() {
  return {/*start=*/0,
          /*stop=*/10,
          /*step=*/1,
          /*batch_size=*/-1,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/"batch_dataset"};
}

class ParameterizedGetNextTest : public BatchDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<BatchDatasetParams>> {};

GetNextTestCase<BatchDatasetParams> GetNextTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({4}),
                               {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})};
}

GetNextTestCase<BatchDatasetParams> GetNextTestCase2() {
  return {/*dataset_params=*/BatchDataset2(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({4}),
                               {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})};
}

GetNextTestCase<BatchDatasetParams> GetNextTestCase3() {
  return {/*dataset_params=*/BatchDataset3(),
          /*expected_outputs=*/
          {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
           CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
           CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
           CreateTensor<int64>(TensorShape({1}), {9})}};
}

GetNextTestCase<BatchDatasetParams> GetNextTestCase4() {
  return {/*dataset_params=*/BatchDataset4(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({3}),
                               {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})};
}

GetNextTestCase<BatchDatasetParams> GetNextTestCase5() {
  return {/*dataset_params=*/BatchDataset5(),
          /*expected_outputs=*/{}};
}

GetNextTestCase<BatchDatasetParams> GetNextTestCase6() {
  return {/*dataset_params=*/BatchDataset6(),
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({10}),
                               {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})};
}

GetNextTestCase<BatchDatasetParams> GetNextTestCase7() {
  return {/*dataset_params=*/BatchDataset7(),
          /*expected_outputs=*/{}};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(batch_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  TF_ASSERT_OK(CheckIteratorGetNext(iterator.get(), iterator_ctx.get(),
                                    test_case.expected_outputs,
                                    /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedGetNextTest,
    ::testing::ValuesIn(std::vector<GetNextTestCase<BatchDatasetParams>>(
        {GetNextTestCase1(), GetNextTestCase2(), GetNextTestCase3(),
         GetNextTestCase4(), GetNextTestCase5(), GetNextTestCase6(),
         GetNextTestCase7()})));

DatasetNodeNameTestCase<BatchDatasetParams> DatasetNodeNameTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_node_name=*/kNodeName};
}

TEST_F(BatchDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetNodeNameTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(CheckDatasetNodeName(*batch_dataset, kNodeName));
}

DatasetTypeStringTestCase<BatchDatasetParams> DatasetTypeStringTestCase1() {
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_dataset_type_string=*/
          name_utils::OpName(BatchDatasetOp::kDatasetType, params)};
}

TEST_F(BatchDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetTypeStringTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(CheckDatasetTypeString(*batch_dataset,
                                      test_case.expected_dataset_type_string));
}

DatasetOutputDtypesTestCase<BatchDatasetParams> DatasetOutputDtypesTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(BatchDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetOutputDtypesTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(CheckDatasetOutputDtypes(*batch_dataset,
                                        test_case.expected_output_dtypes));
}

class ParameterizedDatasetOutputShapesTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          DatasetOutputShapesTestCase<BatchDatasetParams>> {};

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_output_shapes=*/{PartialTensorShape({4})}};
}

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase2() {
  return {/*dataset_params=*/BatchDataset2(),
          /*expected_output_shapes=*/{PartialTensorShape({4})}};
}

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase3() {
  return {/*dataset_params=*/BatchDataset3(),
          /*expected_output_shapes=*/{PartialTensorShape({-1})}};
}

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase4() {
  return {/*dataset_params=*/BatchDataset4(),
          /*expected_output_shapes=*/{PartialTensorShape({3})}};
}

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase5() {
  return {/*dataset_params=*/BatchDataset5(),
          /*expected_output_shapes=*/{PartialTensorShape({12})}};
}

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase6() {
  return {/*dataset_params=*/BatchDataset6(),
          /*expected_output_shapes=*/{PartialTensorShape({-1})}};
}

DatasetOutputShapesTestCase<BatchDatasetParams> DatasetOutputShapesTestCase7() {
  return {/*dataset_params=*/BatchDataset7(),
          /*expected_output_shapes=*/{PartialTensorShape({4})}};
}

TEST_P(ParameterizedDatasetOutputShapesTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(CheckDatasetOutputShapes(*batch_dataset,
                                        test_case.expected_output_shapes));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedDatasetOutputShapesTest,
    ::testing::ValuesIn(
        std::vector<DatasetOutputShapesTestCase<BatchDatasetParams>>(
            {DatasetOutputShapesTestCase1(), DatasetOutputShapesTestCase2(),
             DatasetOutputShapesTestCase3(), DatasetOutputShapesTestCase4(),
             DatasetOutputShapesTestCase5(), DatasetOutputShapesTestCase6(),
             DatasetOutputShapesTestCase7()})));

class ParameterizedCardinalityTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          CardinalityTestCase<BatchDatasetParams>> {};

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_cardinality=*/3};
}

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase2() {
  return {/*dataset_params=*/BatchDataset2(),
          /*expected_cardinality=*/3};
}

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase3() {
  return {/*dataset_params=*/BatchDataset3(),
          /*expected_cardinality=*/4};
}

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase4() {
  return {/*dataset_params=*/BatchDataset4(),
          /*expected_cardinality=*/3};
}

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase5() {
  return {/*dataset_params=*/BatchDataset5(),
          /*expected_cardinality=*/0};
}

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase6() {
  return {/*dataset_params=*/BatchDataset6(),
          /*expected_cardinality=*/1};
}

CardinalityTestCase<BatchDatasetParams> CardinalityTestCase7() {
  return {/*dataset_params=*/BatchDataset7(),
          /*expected_cardinality=*/0};
}

TEST_P(ParameterizedCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(
      CheckDatasetCardinality(*batch_dataset, test_case.expected_cardinality));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedCardinalityTest,
    ::testing::ValuesIn(std::vector<CardinalityTestCase<BatchDatasetParams>>(
        {CardinalityTestCase1(), CardinalityTestCase2(), CardinalityTestCase3(),
         CardinalityTestCase4(), CardinalityTestCase5(), CardinalityTestCase6(),
         CardinalityTestCase7()})));

DatasetSaveTestCase<BatchDatasetParams> DatasetSaveTestCase1() {
  return {/*dataset_params=*/BatchDataset1()};
}

TEST_F(BatchDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = DatasetSaveTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(CheckDatasetSave(*batch_dataset));
}

IsStatefulTestCase<BatchDatasetParams> IsStatefulTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_stateful=*/false};
}

TEST_F(BatchDatasetOpTest, IsStateful) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IsStatefulTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  TF_ASSERT_OK(
      CheckDatasetIsStateful(*batch_dataset, test_case.expected_stateful));
}

IteratorOutputDtypesTestCase<BatchDatasetParams>
IteratorOutputDtypesTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_output_dtypes=*/{DT_INT64}};
}

TEST_F(BatchDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IteratorOutputDtypesTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(batch_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  TF_ASSERT_OK(
      CheckIteratorOutputDtypes(*iterator, test_case.expected_output_dtypes));
}

class ParameterizedIteratorOutputShapesTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorOutputShapesTestCase<BatchDatasetParams>> {};

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_output_shapes=*/{PartialTensorShape({4})}};
}

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase2() {
  return {/*dataset_params=*/BatchDataset2(),
          /*expected_output_shapes=*/{PartialTensorShape({4})}};
}

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase3() {
  return {/*dataset_params=*/BatchDataset3(),
          /*expected_output_shapes=*/{PartialTensorShape({-1})}};
}

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase4() {
  return {/*dataset_params=*/BatchDataset4(),
          /*expected_output_shapes=*/{PartialTensorShape({3})}};
}

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase5() {
  return {/*dataset_params=*/BatchDataset5(),
          /*expected_output_shapes=*/{PartialTensorShape({12})}};
}

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase6() {
  return {/*dataset_params=*/BatchDataset6(),
          /*expected_output_shapes=*/{PartialTensorShape({-1})}};
}

IteratorOutputShapesTestCase<BatchDatasetParams>
IteratorOutputShapesTestCase7() {
  return {/*dataset_params=*/BatchDataset7(),
          /*expected_output_shapes=*/{PartialTensorShape({4})}};
}

TEST_P(ParameterizedIteratorOutputShapesTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(batch_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  TF_ASSERT_OK(
      CheckIteratorOutputShapes(*iterator, test_case.expected_output_shapes));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedIteratorOutputShapesTest,
    ::testing::ValuesIn(
        std::vector<IteratorOutputShapesTestCase<BatchDatasetParams>>(
            {IteratorOutputShapesTestCase1(), IteratorOutputShapesTestCase2(),
             IteratorOutputShapesTestCase3(), IteratorOutputShapesTestCase4(),
             IteratorOutputShapesTestCase5(), IteratorOutputShapesTestCase6(),
             IteratorOutputShapesTestCase7()})));

IteratorPrefixTestCase<BatchDatasetParams> IteratorOutputPrefixTestCase1() {
  name_utils::IteratorPrefixParams params;
  params.op_version = kOpVersion;
  return {/*dataset_params=*/BatchDataset1(),
          /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
              BatchDatasetOp::kDatasetType, kIteratorPrefix, params)};
}

TEST_F(BatchDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = IteratorOutputPrefixTestCase1();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(batch_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  TF_ASSERT_OK(
      CheckIteratorPrefix(*iterator, test_case.expected_iterator_prefix));
}

class ParameterizedIteratorSaveAndRestoreTest
    : public BatchDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<BatchDatasetParams>> {};

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase1() {
  return {/*dataset_params=*/BatchDataset1(),
          /*breakpoints=*/{0, 1, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({4}),
                               {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})};
}

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase2() {
  return {/*dataset_params=*/BatchDataset2(),
          /*breakpoints=*/{0, 1, 5},
          /*expected_outputs=*/
          CreateTensors<int64>(TensorShape({4}),
                               {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})};
}

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase3() {
  return {/*dataset_params=*/BatchDataset3(),
          /*breakpoints=*/{0, 1, 5},
          /*expected_outputs=*/
          {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
           CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
           CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
           CreateTensor<int64>(TensorShape({1}), {9})}};
}

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase4() {
  return {/*dataset_params=*/BatchDataset4(),
          /*breakpoints=*/{0, 1, 5},
          /*expected_outputs=*/
          {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
           CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
           CreateTensor<int64>(TensorShape({3}), {6, 7, 8})}};
}

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase5() {
  return {/*dataset_params=*/BatchDataset5(),
          /*breakpoints=*/{0, 1, 5},
          /*expected_outputs=*/{}};
}

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase6() {
  return {
      /*dataset_params=*/BatchDataset6(),
      /*breakpoints=*/{0, 1, 5},
      /*expected_outputs=*/
      {CreateTensor<int64>(TensorShape({10}), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}};
}

IteratorSaveAndRestoreTestCase<BatchDatasetParams>
IteratorSaveAndRestoreTestCase7() {
  return {/*dataset_params=*/BatchDataset7(),
          /*breakpoints=*/{0, 1, 5},
          /*expected_outputs=*/{}};
}

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = GetParam();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  TF_ASSERT_OK(CreateDataset(batch_dataset_kernel.get(),
                             batch_dataset_context.get(), &batch_dataset));
  core::ScopedUnref scoped_unref(batch_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(batch_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(batch_dataset->MakeIterator(iterator_ctx.get(), kIteratorPrefix,
                                           &iterator));

  TF_ASSERT_OK(CheckIteratorSaveAndRestore(
      *batch_dataset, iterator_ctx.get(), kIteratorPrefix,
      test_case.expected_outputs, test_case.breakpoints));
}

INSTANTIATE_TEST_SUITE_P(
    BatchDatasetOpTest, ParameterizedIteratorSaveAndRestoreTest,
    ::testing::ValuesIn(std::vector<
                        IteratorSaveAndRestoreTestCase<BatchDatasetParams>>(
        {IteratorSaveAndRestoreTestCase1(), IteratorSaveAndRestoreTestCase2(),
         IteratorSaveAndRestoreTestCase3(), IteratorSaveAndRestoreTestCase4(),
         IteratorSaveAndRestoreTestCase5(), IteratorSaveAndRestoreTestCase6(),
         IteratorSaveAndRestoreTestCase7()})));

GetNextTestCase<BatchDatasetParams> InvalidBatchSizeTestCase() {
  return {/*dataset_params=*/InvalidBatchSizeBatchDataset(),
          /*expected_outputs=*/{}};
}

TEST_F(BatchDatasetOpTest, InvalidBatchSize) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  auto test_case = InvalidBatchSizeTestCase();

  std::unique_ptr<OpKernel> batch_dataset_kernel;
  TF_ASSERT_OK(CreateBatchDatasetOpKernel(
      test_case.dataset_params.parallel_copy,
      test_case.dataset_params.output_dtypes,
      test_case.dataset_params.output_shapes, &batch_dataset_kernel));

  TF_ASSERT_OK(MakeRangeDataset(
      test_case.dataset_params.range_dataset_params.start,
      test_case.dataset_params.range_dataset_params.stop,
      test_case.dataset_params.range_dataset_params.step,
      test_case.dataset_params.range_dataset_params.output_dtypes,
      test_case.dataset_params.range_dataset_params.output_shapes,
      &test_case.dataset_params.input_dataset));

  gtl::InlinedVector<TensorValue, 4> inputs;
  TF_ASSERT_OK(test_case.dataset_params.MakeInputs(&inputs));

  std::unique_ptr<OpKernelContext> batch_dataset_context;
  TF_ASSERT_OK(CreateBatchDatasetContext(batch_dataset_kernel.get(), &inputs,
                                         &batch_dataset_context));
  DatasetBase* batch_dataset;
  EXPECT_EQ(CreateDataset(batch_dataset_kernel.get(),
                          batch_dataset_context.get(), &batch_dataset)
                .code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
